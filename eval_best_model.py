"""Evaluate best_model.model from finetune_from_pretrain_gt_bce_joint on CLEVRER-Humans val."""
import sys, os, json, torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_default_cfg, config_to_dict
from vlm_model import VLM_for_STAR
from dataset import get_dataloader
from tqdm import tqdm


def main():
    cfg = get_default_cfg(mode='video')
    cfg.merge_from_file("configs/clevrer_paper.yaml")

    DATA = "/data/kimin10866/repos/CLEVRER-Humans1.0-main/models/clevrer_monet_latents"
    CLEVRER = "/data/kimin10866/repos/CLEVRER-Humans1.0-main/data/clevrer"
    CKPT = "outputs/finetune_from_pretrain_gt_bce_joint/best_model.model"

    cfg.merge_from_list([
        "DATALOADER.CLEVRER_TRAIN_JSON", f"{DATA}/train_question.json",
        "DATALOADER.CLEVRER_VAL_JSON", f"{DATA}/val_question.json",
        "DATALOADER.CLEVRER_DETECTOR_JSON_TRAIN_ROOT", f"{CLEVRER}/detector_json_export_gt/train",
        "DATALOADER.CLEVRER_DETECTOR_JSON_VAL_ROOT", f"{CLEVRER}/detector_json_export_gt/val",
        "DATALOADER.CLEVRER_ANNOTATION_TRAIN_ROOT", f"{CLEVRER}/annotation_train",
        "DATALOADER.CLEVRER_ANNOTATION_VAL_ROOT", f"{CLEVRER}/annotation_validation",
        "DATALOADER.BATCH_SIZE", "8",
        "DATALOADER.EVAL_BATCH_SIZE", "8",
        "SOLVER.CRITERION", "binary_cross_entropy",
        "LOAD", "True",
        "LOAD_PATH", CKPT,
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_dataloaders, _ = get_dataloader(cfg, distributed=False)
    val_loader = val_dataloaders[0]

    net = VLM_for_STAR(cfg).to(device)
    print(f"Loading checkpoint: {CKPT}")
    state = torch.load(CKPT, map_location=device, weights_only=False)
    net.load_state_dict(state, strict=False)
    net.eval()

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    class TxtProc:
        def __call__(self, text):
            return text
    txt_proc = TxtProc()

    all_results = []

    with torch.no_grad():
        for item in tqdm(val_loader, desc="Evaluating"):
            person_bbox = item['person_bbox'].to(device)
            person_kp = item['person_pose_coords'].to(device)
            obj_bbox = item['obj_bboxes'].to(device)
            obj_label = item['obj_labels'].to(device)
            obj_rels = item['obj_rels'].to(device)
            obj_att = item['obj_att_rels'].to(device)
            obj_spatial = item['obj_spatial_rels'].to(device)
            obj_contact = item['obj_contacting_rels'].to(device)
            frame_res = item['frame_resolutions'].to(device)
            answer = item['answer_ids'].to(device)
            choice_mask = item.get('choice_mask', torch.ones_like(answer)).to(device)
            num_objs = item['num_objs']
            num_unpadded = item['num_nonpadded_frames']

            q_texts = [txt_proc(q) for q in item['questions']]
            c0 = [txt_proc(c) for c in item['choice0_texts']]
            c1 = [txt_proc(c) for c in item['choice1_texts']]
            c2 = [txt_proc(c) for c in item['choice2_texts']]
            c3 = [txt_proc(c) for c in item['choice3_texts']]

            output = net(
                (person_bbox, person_kp),
                (obj_bbox, obj_label, obj_rels, obj_att, obj_spatial, obj_contact),
                (q_texts, c0, c1, c2, c3),
                frame_res, num_objs, num_unpadded, save_atts=False
            )

            probs = torch.sigmoid(output).cpu()
            labels = answer.cpu().float()
            mask = choice_mask.cpu().float()

            qids = item.get('question_ids', list(range(len(q_texts))))
            vnames = item.get('video_filenames', [''] * len(q_texts))
            qtypes = item.get('question_types', ['unknown'] * len(q_texts))

            for b in range(len(q_texts)):
                n_choices = int(mask[b].sum().item())
                choices_detail = []
                for c in range(n_choices):
                    pred = 1 if probs[b, c].item() >= 0.5 else 0
                    label = int(labels[b, c].item())
                    choices_detail.append({
                        "choice_id": c,
                        "prob": round(probs[b, c].item(), 4),
                        "pred": pred,
                        "label": label,
                        "correct": pred == label,
                    })

                q_correct = all(cd["correct"] for cd in choices_detail)
                opt_correct = sum(1 for cd in choices_detail if cd["correct"])

                all_results.append({
                    "video": vnames[b] if isinstance(vnames[b], str) else str(vnames[b]),
                    "question_id": int(qids[b]) if not isinstance(qids[b], str) else qids[b],
                    "question": item['questions'][b],
                    "question_type": qtypes[b] if isinstance(qtypes[b], str) else str(qtypes[b]),
                    "n_choices": n_choices,
                    "choices": choices_detail,
                    "question_correct": q_correct,
                    "option_correct": opt_correct,
                })

    # === Summary stats ===
    total_q = len(all_results)
    q_correct = sum(1 for r in all_results if r["question_correct"])
    total_opt = sum(r["n_choices"] for r in all_results)
    opt_correct = sum(r["option_correct"] for r in all_results)

    print(f"\n{'='*60}")
    print(f"Total questions: {total_q}")
    print(f"Question accuracy: {q_correct}/{total_q} = {q_correct/total_q:.4f}")
    print(f"Option accuracy:   {opt_correct}/{total_opt} = {opt_correct/total_opt:.4f}")

    # Per question type
    by_type = defaultdict(lambda: {"q_total": 0, "q_correct": 0, "opt_total": 0, "opt_correct": 0})
    for r in all_results:
        t = r["question_type"]
        by_type[t]["q_total"] += 1
        by_type[t]["q_correct"] += int(r["question_correct"])
        by_type[t]["opt_total"] += r["n_choices"]
        by_type[t]["opt_correct"] += r["option_correct"]

    print(f"\n--- Per Question Type ---")
    for t, s in sorted(by_type.items()):
        q_acc = s["q_correct"] / max(s["q_total"], 1)
        o_acc = s["opt_correct"] / max(s["opt_total"], 1)
        print(f"  {t}: q_acc={q_acc:.4f} ({s['q_correct']}/{s['q_total']})  opt_acc={o_acc:.4f} ({s['opt_correct']}/{s['opt_total']})")

    # Per choice accuracy (choice 0 vs choice 1)
    print(f"\n--- Per Choice Slot ---")
    for c_idx in range(2):
        c_total = 0
        c_correct = 0
        c_pred_pos = 0
        c_label_pos = 0
        for r in all_results:
            if c_idx < len(r["choices"]):
                cd = r["choices"][c_idx]
                c_total += 1
                c_correct += int(cd["correct"])
                c_pred_pos += cd["pred"]
                c_label_pos += cd["label"]
        print(f"  Choice {c_idx}: acc={c_correct/max(c_total,1):.4f} ({c_correct}/{c_total})  pred_pos={c_pred_pos}  label_pos={c_label_pos}")

    # Label bias
    label_pos = sum(1 for r in all_results for cd in r["choices"] if cd["label"] == 1)
    label_neg = total_opt - label_pos
    pred_pos = sum(1 for r in all_results for cd in r["choices"] if cd["pred"] == 1)
    pred_neg = total_opt - pred_pos
    acc_on_correct = sum(1 for r in all_results for cd in r["choices"] if cd["label"] == 1 and cd["correct"]) / max(label_pos, 1)
    acc_on_wrong = sum(1 for r in all_results for cd in r["choices"] if cd["label"] == 0 and cd["correct"]) / max(label_neg, 1)
    print(f"\n--- Bias ---")
    print(f"  label: pos={label_pos} neg={label_neg}")
    print(f"  pred:  pos={pred_pos} neg={pred_neg}")
    print(f"  acc_on_correct_labels={acc_on_correct:.4f}")
    print(f"  acc_on_wrong_labels={acc_on_wrong:.4f}")

    # Threshold sweep
    print(f"\n--- Threshold Sweep ---")
    for th in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        o_corr = 0
        q_corr = 0
        for r in all_results:
            q_ok = True
            for cd in r["choices"]:
                p = 1 if cd["prob"] >= th else 0
                if p == cd["label"]:
                    o_corr += 1
                else:
                    q_ok = False
            if q_ok:
                q_corr += 1
        print(f"  th={th:.2f}: opt_acc={o_corr/max(total_opt,1):.4f}  q_acc={q_corr/max(total_q,1):.4f}")

    # Sample wrong predictions
    wrong = [r for r in all_results if not r["question_correct"]]
    print(f"\n--- Sample Wrong Predictions ({len(wrong)} total) ---")
    for r in wrong[:10]:
        print(f"  [{r['video']}] Q{r['question_id']} ({r['question_type']}): {r['question']}")
        for cd in r["choices"]:
            mark = "O" if cd["correct"] else "X"
            print(f"    [{mark}] choice{cd['choice_id']}: prob={cd['prob']:.4f} pred={cd['pred']} label={cd['label']}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
