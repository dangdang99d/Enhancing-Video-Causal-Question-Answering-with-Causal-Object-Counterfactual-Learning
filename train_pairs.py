"""
Train IPRM with 1-choice pair format.
Each sample = (video, question, single_choice) -> binary label.
No 4-slot padding, no choice_mask.
"""
try:
    import wandb
except ImportError:
    wandb = None
import sys, os, pickle, json, csv, random
from collections import Counter, defaultdict
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset_pairs_v2 import get_pair_dataloader_v2 as get_pair_dataloader
from config import get_default_cfg, config_to_dict, config_to_wandb
from vlm_model import VLM_for_STAR


def _build_egnn_runtime(cfg, device):
    if not bool(getattr(cfg.VLM, "USE_EGNN_DYNAMICS", False)):
        return None
    from egnn_runtime import FrozenEGNN
    ckpt = getattr(cfg.VLM, "EGNN_CKPT_PATH", "")
    hidden = int(getattr(cfg.VLM, "EGNN_DYN_DIM", 128))
    layers = int(getattr(cfg.VLM, "EGNN_LAYERS", 4))
    print(f"[egnn] loading frozen EGNN ckpt={ckpt} hidden={hidden} layers={layers}")
    egnn = FrozenEGNN(ckpt, hidden=hidden, n_layers=layers).to(device)
    egnn.eval()
    return egnn


def _egnn_pack(egnn, bbox_xyxy, labels, num_objs, num_unpadded, cause_mask):
    """Build per-batch valid mask from num_objs and num_unpadded, run EGNN dual."""
    if egnn is None:
        return None
    B, F, N, _ = bbox_xyxy.shape
    valid = torch.zeros(B, F, N, device=bbox_xyxy.device)
    no = torch.as_tensor(num_objs, device=bbox_xyxy.device)  # [B,F]
    for bi in range(B):
        nf = int(num_unpadded[bi])
        for fi in range(F):
            if fi < nf:
                k = int(no[bi, fi].item())
                if k > 0:
                    valid[bi, fi, :k] = 1.0
    return egnn.dual_forward(bbox_xyxy, labels, valid, cause_mask)

seed_val = 200
torch.manual_seed(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)

device = None


def build_text_processors(cfg):
    if cfg.VLM.USE_PRETRAINED_LANG_ENCODER:
        try:
            from lavis.models import load_model_and_preprocess
            _, _, txt_processors = load_model_and_preprocess(
                name="blip_vqa", model_type="vqav2", is_eval=True, device="cpu"
            )
            del _
            return txt_processors
        except (ImportError, Exception):
            pass
    return {"eval": lambda s: s.strip().lower() if isinstance(s, str) else s}


def _distributed_env():
    ws = os.environ.get("WORLD_SIZE", "")
    if not ws:
        return False, 0, 1, 0
    try:
        world_size = int(ws)
    except ValueError:
        return False, 0, 1, 0
    if world_size <= 1:
        return False, 0, 1, 0
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return True, rank, world_size, local_rank


def _ddp_backend():
    return "nccl" if dist.is_nccl_available() else "gloo"


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    m2 = model2.module if getattr(model2, "module", None) is not None else model2
    par2 = dict(m2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1.0 - decay))


def train_epoch(epoch, dataloader, net, net_running, optimizer, cfg,
                txt_processors, is_main=True, accum_steps=1, label_smoothing=0.0,
                egnn_runtime=None, l_sim_weight=0.05,
                focal_gamma=0.0, focal_alpha=0.5):
    pbar = tqdm(iter(dataloader), disable=not is_main)
    moving_acc = None
    moving_loss = None
    accum_steps = max(int(accum_steps), 1)
    net.train(True)
    optimizer.zero_grad(set_to_none=True)

    total_correct = 0
    total_samples = 0
    pred_pos = 0
    pred_neg = 0
    label_pos = 0
    label_neg = 0

    # --- diagnostics accumulators ---
    sum_bce = 0.0
    sum_sim = 0.0
    sum_vnorm = 0.0
    sum_lnorm = 0.0
    sum_vis_attn_ent = 0.0
    sum_grad_lang = 0.0
    sum_grad_vis = 0.0
    sum_grad_other = 0.0
    n_batches = 0
    n_step_batches = 0

    for batch_n, item in enumerate(pbar):
        obj_bbox_x = item["obj_bboxes"].to(device)
        obj_label_x = item["obj_labels"].to(device)
        spatial_rel_x = item["obj_spatial_rels"].to(device)
        contact_rel_x = item["obj_contacting_rels"].to(device)
        frame_res = item["frame_resolutions"].to(device)
        num_objs = item["num_objs"]
        num_unpadded = item["num_nonpadded_frames"]
        labels = item["label"].to(device)

        q_texts = [txt_processors["eval"](t) for t in item["question_text"]]
        c_texts = [txt_processors["eval"](t) for t in item["choice_text"]]

        cause_mask = item.get("cause_obj_mask")
        if cause_mask is not None:
            cause_mask = cause_mask.to(device)

        raw_net = net.module if hasattr(net, "module") else net
        egnn_pack = _egnn_pack(egnn_runtime, obj_bbox_x, obj_label_x,
                               num_objs, num_unpadded, cause_mask)
        output = net(
            (obj_bbox_x, obj_label_x, spatial_rel_x, contact_rel_x),
            (q_texts, c_texts),
            frame_res, num_objs, num_unpadded, save_atts=False,
            cause_obj_mask=cause_mask,
            egnn_pack=egnn_pack,
        )
        logits = output.squeeze(-1)

        if label_smoothing > 0.0:
            labels_target = labels * (1.0 - 2.0 * label_smoothing) + label_smoothing
        else:
            labels_target = labels
        if focal_gamma > 0.0:
            # Focal BCE: alpha_t * (1 - p_t)^gamma * BCE
            # Targets hard positives (label=1, p~0) and hard negatives (label=0, p~1).
            bce_per = F.binary_cross_entropy_with_logits(logits, labels_target, reduction='none')
            p = torch.sigmoid(logits)
            p_t = p * labels + (1.0 - p) * (1.0 - labels)
            focal_w = (1.0 - p_t).clamp(min=1e-8) ** focal_gamma
            alpha_t = focal_alpha * labels + (1.0 - focal_alpha) * (1.0 - labels)
            loss_bce = (alpha_t * focal_w * bce_per).mean()
        else:
            loss_bce = F.binary_cross_entropy_with_logits(logits, labels_target)
        loss = loss_bce
        l_sim_val = 0.0

        # L_sim: sign-conditioned similarity loss (VGCM-inspired)
        # When label=0 (non-causal choice), masking that object should NOT change the representation
        if use_masked_path and hasattr(raw_net, '_last_feat_full') and raw_net._last_feat_full is not None:
            feat_full   = raw_net._last_feat_full    # B x D
            feat_masked = raw_net._last_feat_masked  # B x D
            l_sim_per = F.mse_loss(feat_full, feat_masked, reduction='none').mean(dim=-1)  # B
            non_causal = (1.0 - labels)  # 1.0 for label=0, 0.0 for label=1
            l_sim = (l_sim_per * non_causal).mean()
            loss = loss + l_sim_weight * l_sim
            l_sim_val = float(l_sim.item())


        (loss / accum_steps).backward()

        # --- diagnostics: feat norms, visual attn entropy, grad norms ---
        with torch.no_grad():
            v_n = float(getattr(raw_net, '_last_vis_feat_norm', 0.0) or 0.0)
            l_n = float(getattr(raw_net, '_last_lang_feat_norm', 0.0) or 0.0)
            sum_vnorm += v_n
            sum_lnorm += l_n

            vis_attn_ent = 0.0
            vlm_mod = getattr(raw_net, 'vlm_module', None)
            if vlm_mod is not None:
                atts = getattr(vlm_mod, 'attentions', None)
                if atts is not None and atts.get('image') is not None:
                    # spatial_attentions stacked: T x B x K x N x 1 (raw, pre-softmax +mask-filled)
                    sa = atts['image']
                    sa = sa.squeeze(-1)                         # T x B x K x N
                    probs = F.softmax(sa, dim=-1)                # softmax over N (spatial)
                    ent = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # T x B x K
                    vis_attn_ent = float(ent.mean().item())
            sum_vis_attn_ent += vis_attn_ent

            sum_bce += float(loss_bce.item())
            sum_sim += l_sim_val
            n_batches += 1

        should_step = ((batch_n + 1) % accum_steps == 0) or ((batch_n + 1) == len(dataloader))
        if should_step:
            # grad norms per path (measure BEFORE clip so norms reflect actual raw gradient magnitude)
            with torch.no_grad():
                g_lang_sq = 0.0
                g_vis_sq = 0.0
                g_other_sq = 0.0
                for name, p in net.named_parameters():
                    if p.grad is None:
                        continue
                    g = float(p.grad.detach().data.norm().item())
                    nl = name.lower()
                    if ('lang_enc' in nl or 'option_enc' in nl or 'proj_lang' in nl
                            or 'proj_option' in nl or 'proj_ques' in nl or 'aux_lang' in nl
                            or 'combine_ques_and_option' in nl or 'lang_sa' in nl
                            or 'question_lang_pos' in nl):
                        g_lang_sq += g * g
                    elif ('vis_' in nl or 'visual' in nl or 'stem' in nl or 'proj_vis' in nl
                            or 'obj_' in nl or 'person_' in nl or 'keypoint' in nl
                            or 'bbox' in nl or 'frame_pos' in nl or 'iprm' in nl
                            or 'vlm_module' in nl):
                        g_vis_sq += g * g
                    else:
                        g_other_sq += g * g
                sum_grad_lang += g_lang_sq ** 0.5
                sum_grad_vis += g_vis_sq ** 0.5
                sum_grad_other += g_other_sq ** 0.5
                n_step_batches += 1

            if cfg.SOLVER.GRAD_CLIP:
                nn.utils.clip_grad_norm_(net.parameters(), cfg.SOLVER.GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.5).float()
            correct = pred.eq(labels).sum().item()
            total_correct += correct
            total_samples += labels.shape[0]
            pred_pos += int(pred.sum().item())
            pred_neg += int((1 - pred).sum().item())
            label_pos += int(labels.sum().item())
            label_neg += int((1 - labels).sum().item())

        acc = total_correct / max(total_samples, 1)
        if moving_acc is None:
            moving_acc = correct / labels.shape[0]
            moving_loss = loss.item()
        else:
            moving_acc = moving_acc * 0.95 + (correct / labels.shape[0]) * 0.05
            moving_loss = moving_loss * 0.95 + loss.item() * 0.05

        pbar.set_description(f"Epoch {epoch}; Loss: {loss.item():.5f}; Acc: {acc:.4f}")

        if net_running is not None and should_step:
            accumulate(net_running, net)

    if is_main:
        print(f"  Train Epoch {epoch}: acc={total_correct/max(total_samples,1):.4f} "
              f"pred_pos={pred_pos} pred_neg={pred_neg} "
              f"label_pos={label_pos} label_neg={label_neg}")
        nb = max(n_batches, 1)
        nsb = max(n_step_batches, 1)
        mean_vnorm = sum_vnorm / nb
        mean_lnorm = sum_lnorm / nb
        vl_ratio = mean_vnorm / max(mean_lnorm, 1e-8)
        print(f"  [Diag] L_bce={sum_bce/nb:.4f} L_sim={sum_sim/nb:.4f} "
              f"(0.05*L_sim={0.05*sum_sim/nb:.4f})")
        print(f"  [Diag] |vis|={mean_vnorm:.3f} |lang|={mean_lnorm:.3f} "
              f"vis/lang={vl_ratio:.3f} vis_attn_entropy={sum_vis_attn_ent/nb:.3f}")
        print(f"  [Diag] grad_norm: lang={sum_grad_lang/nsb:.3e} "
              f"vis={sum_grad_vis/nsb:.3e} other={sum_grad_other/nsb:.3e} "
              f"vis/lang={sum_grad_vis/max(sum_grad_lang,1e-12):.3f}")

    return total_correct / max(total_samples, 1)


def validate(epoch, val_dataloaders, net_running, cfg, txt_processors, scheduler,
             is_main=True, egnn_runtime=None):
    results = []
    val_dicts = []
    for val_i, val_dl in enumerate(val_dataloaders):
        net_running.train(False)

        all_preds = []
        with torch.no_grad():
            pbar = tqdm(iter(val_dl), disable=not is_main)
            for item in pbar:
                obj_bbox_x = item["obj_bboxes"].to(device)
                obj_label_x = item["obj_labels"].to(device)
                spatial_rel_x = item["obj_spatial_rels"].to(device)
                contact_rel_x = item["obj_contacting_rels"].to(device)
                frame_res = item["frame_resolutions"].to(device)
                num_objs = item["num_objs"]
                num_unpadded = item["num_nonpadded_frames"]
                labels = item["label"].to(device)

                q_texts = [txt_processors["eval"](t) for t in item["question_text"]]
                c_texts = [txt_processors["eval"](t) for t in item["choice_text"]]

                cause_mask = item.get("cause_obj_mask")
                if cause_mask is not None:
                    cause_mask = cause_mask.to(device)

                egnn_pack = _egnn_pack(egnn_runtime, obj_bbox_x, obj_label_x,
                                       num_objs, num_unpadded, cause_mask)
                output = net_running(
                    (obj_bbox_x, obj_label_x, spatial_rel_x, contact_rel_x),
                    (q_texts, c_texts),
                    frame_res, num_objs, num_unpadded, save_atts=False,
                    cause_obj_mask=cause_mask,
                    egnn_pack=egnn_pack,
                )
                logits = output.squeeze(-1)
                probs = torch.sigmoid(logits)

                for bi in range(labels.shape[0]):
                    all_preds.append({
                        "video_id": item["video_id"][bi],
                        "question_id": item["question_id"][bi],
                        "choice_id": item["choice_id"][bi],
                        "question_type": item["question_type"][bi],
                        "label": int(labels[bi].item()),
                        "prob": float(probs[bi].item()),
                        "logit": float(logits[bi].item()),
                    })

        stats = _compute_pair_stats(all_preds)
        if is_main:
            val_label = f"[val_{val_i}]" if len(val_dataloaders) > 1 else ""
            print(f"  {val_label} {len(all_preds)} pairs evaluated")
            _print_stats(epoch, stats)

        if scheduler:
            scheduler.step(stats["option_acc_0.5"])

        results.append(stats["option_acc_0.5"])
        val_dicts.append(stats)

    return results, val_dicts


def _compute_pair_stats(preds):
    thresholds = [round(x, 2) for x in np.arange(0.30, 0.71, 0.02)]
    stats = {}

    total = len(preds)
    labels = np.array([p["label"] for p in preds])
    probs = np.array([p["prob"] for p in preds])

    stats["total_pairs"] = total
    stats["label_pos"] = int(labels.sum())
    stats["label_neg"] = int((1 - labels).sum())

    for th in thresholds:
        pred_bin = (probs >= th).astype(float)
        correct = (pred_bin == labels).sum()
        stats[f"option_acc_{th:.2f}"] = correct / max(total, 1)

    th05_pred = (probs >= 0.5).astype(float)
    stats["option_acc_0.5"] = stats.get("option_acc_0.50", 0.0)
    stats["pred_pos_0.5"] = int(th05_pred.sum())
    stats["pred_neg_0.5"] = int((1 - th05_pred).sum())

    # per-label accuracy (bias detection)
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() > 0:
        stats["acc_on_correct_labels"] = float((th05_pred[pos_mask] == 1).sum() / pos_mask.sum())
    else:
        stats["acc_on_correct_labels"] = 0.0
    if neg_mask.sum() > 0:
        stats["acc_on_wrong_labels"] = float((th05_pred[neg_mask] == 0).sum() / neg_mask.sum())
    else:
        stats["acc_on_wrong_labels"] = 0.0

    # per question type
    by_qtype = defaultdict(list)
    for p in preds:
        by_qtype[p["question_type"]].append(p)
    stats["by_question_type"] = {}
    for qt, items in sorted(by_qtype.items()):
        qt_labels = np.array([x["label"] for x in items])
        qt_probs = np.array([x["prob"] for x in items])
        qt_pred = (qt_probs >= 0.5).astype(float)
        qt_acc = (qt_pred == qt_labels).sum() / max(len(items), 1)
        stats["by_question_type"][qt] = {
            "count": len(items),
            "acc": float(qt_acc),
            "label_pos": int(qt_labels.sum()),
            "label_neg": int((1 - qt_labels).sum()),
            "pred_pos": int(qt_pred.sum()),
            "pred_neg": int((1 - qt_pred).sum()),
        }

    # question-level accuracy (group by video_id + question_id)
    by_question = defaultdict(list)
    for p in preds:
        qkey = f"{p['video_id']}#{p['question_id']}"
        by_question[qkey].append(p)
    q_correct = 0
    q_total = 0
    for qkey, items in by_question.items():
        q_total += 1
        all_ok = all((item["prob"] >= 0.5) == (item["label"] == 1) for item in items)
        if all_ok:
            q_correct += 1
    stats["question_acc_0.5"] = q_correct / max(q_total, 1)
    stats["num_questions"] = q_total

    # best threshold
    best_th = 0.5
    best_opt_acc = stats["option_acc_0.50"]
    for th in thresholds:
        acc = stats[f"option_acc_{th:.2f}"]
        if acc > best_opt_acc:
            best_opt_acc = acc
            best_th = th
    stats["best_th"] = best_th
    stats["best_option_acc"] = best_opt_acc

    # logit stats (overall)
    logits = np.array([p["logit"] for p in preds])
    stats["logit_mean"] = float(logits.mean())
    stats["logit_std"] = float(logits.std())
    stats["logit_min"] = float(logits.min())
    stats["logit_max"] = float(logits.max())

    # logit stats per label (bias diagnosis)
    logits_pos = logits[pos_mask]
    logits_neg = logits[neg_mask]
    if len(logits_pos) > 0:
        stats["logit_pos_mean"] = float(logits_pos.mean())
        stats["logit_pos_std"] = float(logits_pos.std())
    else:
        stats["logit_pos_mean"] = 0.0
        stats["logit_pos_std"] = 0.0
    if len(logits_neg) > 0:
        stats["logit_neg_mean"] = float(logits_neg.mean())
        stats["logit_neg_std"] = float(logits_neg.std())
    else:
        stats["logit_neg_mean"] = 0.0
        stats["logit_neg_std"] = 0.0

    # probability distribution bins
    prob_bins = {"p<0.2": 0, "0.2<=p<0.4": 0, "0.4<=p<0.6": 0, "0.6<=p<0.8": 0, "p>=0.8": 0}
    for p_val in probs:
        if p_val < 0.2:
            prob_bins["p<0.2"] += 1
        elif p_val < 0.4:
            prob_bins["0.2<=p<0.4"] += 1
        elif p_val < 0.6:
            prob_bins["0.4<=p<0.6"] += 1
        elif p_val < 0.8:
            prob_bins["0.6<=p<0.8"] += 1
        else:
            prob_bins["p>=0.8"] += 1
    stats["prob_distribution"] = prob_bins

    # separation score: ideal = pos logits >> 0, neg logits << 0
    stats["logit_separation"] = stats["logit_pos_mean"] - stats["logit_neg_mean"]

    return stats


def _print_stats(epoch, s):
    print(f"\n{'='*60}")
    print(f"Val Epoch {epoch}: option_acc@0.5={s['option_acc_0.5']:.4f} "
          f"question_acc@0.5={s['question_acc_0.5']:.4f}")
    print(f"  best_th={s['best_th']:.2f} best_option_acc={s['best_option_acc']:.4f}")
    print(f"  pred_pos={s['pred_pos_0.5']} pred_neg={s['pred_neg_0.5']} "
          f"(label: pos={s['label_pos']} neg={s['label_neg']})")
    print(f"  acc_on_correct_labels(label=1→pred=1)={s['acc_on_correct_labels']:.4f} "
          f"acc_on_wrong_labels(label=0→pred=0)={s['acc_on_wrong_labels']:.4f}")
    print(f"  --- Logit Stats ---")
    print(f"  overall: mean={s['logit_mean']:.4f} std={s['logit_std']:.4f} "
          f"min={s['logit_min']:.4f} max={s['logit_max']:.4f}")
    print(f"  label=1: mean={s['logit_pos_mean']:.4f} std={s['logit_pos_std']:.4f}")
    print(f"  label=0: mean={s['logit_neg_mean']:.4f} std={s['logit_neg_std']:.4f}")
    print(f"  separation(pos_mean - neg_mean)={s['logit_separation']:.4f} "
          f"(ideal: >>1.0)")
    print(f"  --- Prob Distribution ---")
    pd = s.get("prob_distribution", {})
    total = max(s["total_pairs"], 1)
    for bin_name, cnt in pd.items():
        print(f"  {bin_name}: {cnt} ({100*cnt/total:.1f}%)")
    print(f"  --- By Question Type ---")
    for qt, info in s.get("by_question_type", {}).items():
        print(f"    {qt}: acc={info['acc']:.4f} n={info['count']} "
              f"(pos={info['label_pos']} neg={info['label_neg']}) "
              f"pred_pos={info['pred_pos']} pred_neg={info['pred_neg']}")
    print(f"{'='*60}\n")


def generate_cfg():
    parser = argparse.ArgumentParser(description="IPRM 1-choice pair training")
    parser.add_argument("--config-file", metavar="FILE")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--masked-path", action="store_true",
                        help="Enable Granger-inspired masked causal path")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="BCE label smoothing eps (e.g. 0.1 -> targets in {0.1, 0.9})")
    parser.add_argument("--swa-start", type=int, default=0,
                        help="1-indexed epoch to start SWA. 0 disables.")
    parser.add_argument("--l-sim-weight", type=float, default=0.05,
                        help="Weight of L_sim similarity loss (default 0.05)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze all params except masked_classifier + masked_indv_head (head-only refine)")
    parser.add_argument("--unfreeze-final-proj", action="store_true",
                        help="With --freeze-backbone, also unfreeze classifier + pre_classifier projs + combine_ques_opt (final projector refine)")
    parser.add_argument("--unfreeze-vlm-module", action="store_true",
                        help="With --freeze-backbone, also unfreeze entire vlm_module (IPRM core: op_states, retrieval blocks, memory attention)")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal BCE focusing exponent (0=off=standard BCE, typical 1-2)")
    parser.add_argument("--focal-alpha", type=float, default=0.5,
                        help="Focal BCE class weight (0.5=balanced, >0.5 upweights positives)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_default_cfg(mode="video")
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.LOAD_PATH in ["", " ", False, None] and not cfg.LOAD:
        cfg.LOAD_PATH = ""
    cfg.freeze()
    return (cfg, args.test_only, args.no_wandb, args.accum_steps,
            args.early_stop_patience, args.masked_path,
            args.label_smoothing, args.swa_start, args.l_sim_weight,
            args.freeze_backbone, args.unfreeze_final_proj, args.unfreeze_vlm_module,
            args.focal_gamma, args.focal_alpha)


if __name__ == "__main__":
    (cfg, test_only, no_wandb, accum_steps, early_stop_patience, use_masked_path,
     label_smoothing, swa_start, l_sim_weight, freeze_backbone, unfreeze_final_proj,
     unfreeze_vlm_module, focal_gamma, focal_alpha) = generate_cfg()
    distributed, rank, world_size, local_rank = _distributed_env()
    is_main = rank == 0

    try:
        if distributed:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=_ddp_backend(), init_method="env://")
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

        if is_main:
            print(f"device={device} distributed={distributed} rank={rank}/{world_size}")
            print("PAIR MODE: 1-choice per sample, no slots, no choice_mask")

        train_dl, val_dls, train_sampler = get_pair_dataloader(
            cfg, distributed=distributed, rank=rank, world_size=world_size)

        if is_main:
            print(f"Train: {len(train_dl.dataset)} pairs, Val: {len(val_dls[0].dataset)} pairs")

        net = VLM_for_STAR(cfg).to(device)
        net_running = None
        if (not distributed) or is_main:
            net_running = VLM_for_STAR(cfg).to(device)

        egnn_runtime = _build_egnn_runtime(cfg, device)

        # When refining head only (--freeze-backbone), we must enable masked_path
        # BEFORE loading, so that _masked_classifier params in the checkpoint load
        # into the newly-created modules. Otherwise strict=False silently drops them.
        if use_masked_path and freeze_backbone:
            if is_main:
                print("Enabling masked causal path BEFORE load (freeze-backbone mode)")
            net.enable_masked_path()
            if net_running is not None:
                net_running.enable_masked_path()

        if cfg.LOAD:
            if is_main:
                print(f"Loading model from {cfg.LOAD_PATH}")
            with open(cfg.LOAD_PATH, "rb") as f:
                try:
                    state = torch.load(f, map_location=device, weights_only=False)
                except TypeError:
                    f.seek(0)
                    state = torch.load(f, map_location=device)
            # Filter size-incompatible tensors (STAR removal changed some dims).
            # Mismatched keys get random-init from current net; FT adapts them.
            _model_state = net.state_dict()
            _skipped = []
            _compat = {}
            for _k, _v in state.items():
                if _k in _model_state and hasattr(_v, "shape") and tuple(_v.shape) == tuple(_model_state[_k].shape):
                    _compat[_k] = _v
                else:
                    _skipped.append((_k, tuple(_v.shape) if hasattr(_v, "shape") else None,
                                     tuple(_model_state[_k].shape) if _k in _model_state else None))
            missing_k, unexpected_k = net.load_state_dict(_compat, strict=False)
            if is_main:
                print(f"  load: missing={len(missing_k)} unexpected={len(unexpected_k)} skipped_size={len(_skipped)}")
                for _k, _cs, _ms in _skipped[:8]:
                    print(f"    SKIP {_k}: ckpt{_cs} vs cur{_ms}")

        if use_masked_path and not freeze_backbone:
            if is_main:
                print("Enabling masked causal path (Granger-inspired)")
            net.enable_masked_path()
            if net_running is not None:
                net_running.enable_masked_path()

        if freeze_backbone:
            # Patterns checked with 'in name' (substring) — head modules
            head_contains = ["_masked_classifier", "_masked_indv_head"]
            # Patterns checked with 'startswith' — final projector path.
            # These produce choice_out in run_individual_choice.
            # Using startswith avoids matching _masked_classifier via 'classifier'.
            extra_startswith = []
            if unfreeze_final_proj or unfreeze_vlm_module:
                extra_startswith = [
                    "classifier.",
                    "lang_summary_rep_pre_classifier_proj.",
                    "option_summary_rep_pre_classifier_proj.",
                    "indv_classifier.",
                    "combine_ques_and_option_summary_proj.",
                ]
            if unfreeze_vlm_module:
                extra_startswith.append("vlm_module.")

            def _is_trainable(name):
                if any(p in name for p in head_contains):
                    return True
                if any(name.startswith(p) for p in extra_startswith):
                    return True
                return False

            trainable_names = []
            for name, p in net.named_parameters():
                if _is_trainable(name):
                    p.requires_grad = True
                    trainable_names.append(name)
                else:
                    p.requires_grad = False
            if net_running is not None:
                for name, p in net_running.named_parameters():
                    p.requires_grad = _is_trainable(name)
            if is_main:
                if unfreeze_vlm_module:
                    mode = "vlm_module+final-proj+head"
                elif unfreeze_final_proj:
                    mode = "final-proj+head"
                else:
                    mode = "head-only"
                print(f"FREEZE-BACKBONE ({mode}): {len(trainable_names)} trainable params")
                for n in trainable_names:
                    print(f"    trainable: {n}")

        if net_running is not None:
            accumulate(net_running, net, 0)

        if distributed:
            net = DDP(net, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=True)

        if freeze_backbone:
            trainable = [p for p in net.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable, lr=cfg.SOLVER.LR)
            if is_main:
                n_trainable = sum(p.numel() for p in trainable)
                n_total = sum(p.numel() for p in net.parameters())
                print(f"FREEZE-BACKBONE optimizer: {n_trainable}/{n_total} params trainable "
                      f"({n_trainable/n_total*100:.4f}%)")
        elif cfg.SOLVER.LANG_ENC_LR:
            lang_enc_params = [p for n, p in net.named_parameters() if "lang_encoder" in n]
            base_params = [p for n, p in net.named_parameters() if "lang_encoder" not in n]
            optimizer = optim.Adam(
                [{"params": base_params}, {"params": lang_enc_params, "lr": cfg.SOLVER.LANG_ENC_LR}],
                lr=cfg.SOLVER.LR,
            )
        else:
            optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.LR)

        scheduler = None
        if cfg.SOLVER.USE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "max", factor=cfg.SOLVER.FACTOR,
                patience=cfg.SOLVER.PATIENCE, threshold=0.001, threshold_mode="rel",
            )

        txt_processors = build_text_processors(cfg)
        best_model_path = os.path.join(cfg.SAVE_DIRECTORY, "best_model.model")
        best_result = -1
        epochs_no_improve = 0

        if is_main:
            os.makedirs(cfg.SAVE_DIRECTORY, exist_ok=True)

        swa_state = None
        swa_count = 0
        if is_main and swa_start > 0:
            print(f"SWA enabled: averaging net_running weights from epoch {swa_start} onwards")

        for epoch in range(1, cfg.SOLVER.EPOCHS + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if epoch == 1 and not test_only and is_main:
                with open(os.path.join(cfg.SAVE_DIRECTORY, "model_config.json"), "w") as f:
                    json.dump(config_to_dict(cfg), f, indent=4)

            if not test_only:
                train_epoch(epoch, train_dl, net, net_running, optimizer, cfg,
                           txt_processors, is_main=is_main, accum_steps=accum_steps,
                           label_smoothing=label_smoothing, egnn_runtime=egnn_runtime,
                           l_sim_weight=l_sim_weight,
                           focal_gamma=focal_gamma, focal_alpha=focal_alpha)

            if distributed:
                dist.barrier()

            val_dicts = None
            if is_main:
                _, val_dicts = validate(epoch, val_dls, net_running, cfg,
                                        txt_processors, scheduler, is_main=True,
                                        egnn_runtime=egnn_runtime)

            if distributed:
                dist.barrier()

            if is_main and val_dicts is not None:
                current = val_dicts[0]["option_acc_0.5"]
                best_from_sweep = val_dicts[0].get("best_option_acc", 0.0)
                current = max(current, best_from_sweep)
                if current > best_result:
                    best_result = current
                    epochs_no_improve = 0
                    print(f"New best: {best_result:.5f}")
                    if net_running is not None:
                        torch.save(net_running.state_dict(), best_model_path)
                        json.dump(val_dicts[0],
                                  open(os.path.join(cfg.SAVE_DIRECTORY, "best_val_stats.json"), "w"),
                                  indent=4, default=str)
                else:
                    epochs_no_improve += 1

                if swa_start > 0 and epoch >= swa_start and net_running is not None:
                    cur_sd = {k: v.detach().cpu().clone()
                              for k, v in net_running.state_dict().items()}
                    if swa_state is None:
                        swa_state = cur_sd
                        swa_count = 1
                    else:
                        swa_count += 1
                        for k in swa_state:
                            if swa_state[k].is_floating_point():
                                swa_state[k].add_(
                                    (cur_sd[k] - swa_state[k]) / swa_count)
                            else:
                                swa_state[k] = cur_sd[k]
                    print(f"  [SWA] updated avg (n={swa_count})")

            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                if is_main:
                    print(f"Early stopping at epoch {epoch}: no improve for {epochs_no_improve} epochs. Best: {best_result:.5f}")
                break

            if test_only:
                if is_main and val_dicts:
                    json.dump(val_dicts[0],
                              open(os.path.join(cfg.SAVE_DIRECTORY, "testonly_val_stats.json"), "w"),
                              indent=4, default=str)
                break

            period = int(getattr(cfg.SOLVER, "CHECKPOINT_PERIOD", 0))
            if is_main and not test_only and period > 0 and epoch % period == 0 and net_running is not None:
                ckpt_path = os.path.join(cfg.SAVE_DIRECTORY, f"checkpoint_epoch_{epoch:04d}.pt")
                torch.save(net_running.state_dict(), ckpt_path)

        if is_main and swa_state is not None and swa_count > 0:
            swa_path = os.path.join(cfg.SAVE_DIRECTORY, "swa_model.model")
            torch.save(swa_state, swa_path)
            print(f"\n=== SWA Evaluation (avg of {swa_count} epochs from ep{swa_start}) ===")
            net_swa = VLM_for_STAR(cfg).to(device)
            if use_masked_path:
                net_swa.enable_masked_path()
            net_swa.load_state_dict(
                {k: v.to(device) for k, v in swa_state.items()}, strict=False)
            _, swa_val = validate(cfg.SOLVER.EPOCHS + 1, val_dls, net_swa, cfg,
                                  txt_processors, None, is_main=True,
                                  egnn_runtime=egnn_runtime)
            if swa_val:
                json.dump(swa_val[0],
                          open(os.path.join(cfg.SAVE_DIRECTORY, "swa_val_stats.json"), "w"),
                          indent=4, default=str)
                swa_opt = swa_val[0].get("option_acc_0.5", 0.0)
                swa_best = swa_val[0].get("best_option_acc", 0.0)
                print(f"  [SWA final] opt_acc@0.5={swa_opt:.4f} best={swa_best:.4f}")
                if max(swa_opt, swa_best) > best_result:
                    print(f"  [SWA] beats best_result ({best_result:.5f}) -> copy to best_model")
                    torch.save({k: v.to(device) for k, v in swa_state.items()},
                               best_model_path)
                    json.dump(swa_val[0],
                              open(os.path.join(cfg.SAVE_DIRECTORY, "best_val_stats.json"), "w"),
                              indent=4, default=str)

    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
