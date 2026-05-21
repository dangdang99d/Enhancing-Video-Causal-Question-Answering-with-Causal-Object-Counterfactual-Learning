"""Standalone EGNN dynamics evaluation on CLEVRER val.

Answers: is the pretrained EGNN itself any good, or is the signal it provides
to IPRM broken at source?

Two heads:
  1) Position: MSE over horizon H rollout (normalized [0,1] coords)
     baseline = constant-velocity extrapolation (copy last velocity)
  2) Collision: BCE / AUROC / AP / F1@0.5 on pairwise coll_logit
     baseline = random (AUC 0.5) + "closer than threshold" geometric baseline

Prints per-epoch-ckpt summary so we can pick which ckpt IPRM loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/data/kimin10866/repos/IPRM_Iterative_and_Parallel_Reasoning_Mechanism-main/videoQA")
from tools.egnn_pretrain import EGNNDynamics, bbox_to_cxcywh, FRAME_W, FRAME_H, ATTR_DIM  # noqa

ANN_ROOT = "/data/kimin10866/repos/CLEVRER-Humans1.0-main/data/clevrer/annotation_validation"
DET_ROOT = "/data/kimin10866/repos/CLEVRER-Humans1.0-main/data/clevrer/detector_json_export_gt/val"

SHAPES = ["cube", "sphere", "cylinder"]
COLORS = ["blue", "cyan", "gray", "brown", "green", "purple", "red", "yellow"]
MATERIALS = ["rubber", "metal"]


def attr_onehot(c, s, m):
    out = np.zeros(ATTR_DIM, np.float32)
    out[SHAPES.index(s)] = 1
    out[len(SHAPES) + COLORS.index(c)] = 1
    out[len(SHAPES) + len(COLORS) + MATERIALS.index(m)] = 1
    return out


def find_ann_path(vid):
    for start in range(0, 15000, 1000):
        sub = f"annotation_{start:05d}-{start+1000:05d}"
        p = os.path.join(ANN_ROOT, sub, f"annotation_{vid:05d}.json")
        if os.path.isfile(p):
            return p
    return None


def process_one(vid):
    ann_p = find_ann_path(vid)
    det_p = os.path.join(DET_ROOT, f"video_{vid:05d}.json")
    if not ann_p or not os.path.isfile(det_p):
        return None
    try:
        ann = json.load(open(ann_p))
        det = json.load(open(det_p))
    except Exception:
        return None

    attr_of = {o["object_id"]: (o["color"], o["shape"], o["material"]) for o in ann["object_property"]}
    n_frames = len(det["per_frame"])
    tracks = [{} for _ in range(n_frames)]
    last_bbox = {}
    attr_to_oids_all = defaultdict(list)
    for oid, a in attr_of.items():
        attr_to_oids_all[a].append(oid)
    for t in range(n_frames):
        dets = det["per_frame"][t]
        attr_to_dets = defaultdict(list)
        for d in dets:
            attr_to_dets[(d["color"], d["shape"], d["material"])].append(d["bbox"])
        for attr, bboxes in attr_to_dets.items():
            oids = attr_to_oids_all.get(attr, [])
            if not oids:
                continue
            if len(bboxes) == 1 and len(oids) == 1:
                oid = oids[0]
                tracks[t][oid] = bboxes[0]
                last_bbox[oid] = bboxes[0]
                continue
            m, n = len(bboxes), len(oids)
            cost = np.full((m, n), 1e6, np.float32)
            for i, b in enumerate(bboxes):
                cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                for j, oid in enumerate(oids):
                    if oid in last_bbox:
                        lb = last_bbox[oid]
                        cl = ((lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2)
                        cost[i, j] = ((cb[0] - cl[0]) ** 2 + (cb[1] - cl[1]) ** 2) ** 0.5
                    else:
                        cost[i, j] = 1000.0
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] >= 1e6 - 1:
                    continue
                oid = oids[j]
                b = bboxes[i]
                tracks[t][oid] = b
                last_bbox[oid] = b
    collisions = [
        (int(c["object_ids"][0]), int(c["object_ids"][1]), int(c["frame_id"]))
        for c in ann.get("collision", [])
    ]
    attrs = {oid: attr_onehot(*attr_of[oid]).tolist() for oid in attr_of}
    return vid, {"attrs": attrs, "tracks": tracks, "collisions": collisions}


def build_val_cache(max_videos, workers):
    t0 = time.time()
    vids = list(range(10000, 10000 + max_videos))
    out = {}
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_one, vids, chunksize=32)):
            if r is not None:
                vid, data = r
                out[vid] = data
            if (i + 1) % 500 == 0:
                print(f"  [cache] {i+1}/{len(vids)} built {len(out)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  [cache] done: {len(out)}/{len(vids)} videos in {time.time()-t0:.0f}s", flush=True)
    return out


def make_windows(db, window=8, horizon=4, stride=8, max_objs=8):
    samples = []
    for vid, e in db.items():
        tracks = e["tracks"]
        attrs_d = e["attrs"]
        n = len(tracks)
        last = n - 1 - horizon
        for s in range(0, max(0, last - window + 1), stride):
            samples.append((vid, s, tracks, attrs_d, e["collisions"]))
    return samples


def batch_samples(samples, batch_size, window, horizon, max_objs):
    N = max_objs
    W = window
    H = horizon
    for b0 in range(0, len(samples), batch_size):
        chunk = samples[b0:b0 + batch_size]
        B = len(chunk)
        bbox_in = np.zeros((B, W, N, 4), np.float32)
        valid_in = np.zeros((B, W, N), np.float32)
        attr_arr = np.zeros((B, N, ATTR_DIM), np.float32)
        pos_tgt = np.zeros((B, H, N, 2), np.float32)
        valid_tgt = np.zeros((B, H, N), np.float32)
        coll_arr = np.zeros((B, N, N), np.float32)
        last_c = np.zeros((B, N, 2), np.float32)   # last-in-frame center (for baselines)
        prev_c = np.zeros((B, N, 2), np.float32)
        for bi, (vid, s, tracks, attrs_d, collisions) in enumerate(chunk):
            for oid_str, a in attrs_d.items():
                oid = int(oid_str)
                if oid < N:
                    attr_arr[bi, oid] = np.asarray(a, np.float32)
            for i, t in enumerate(range(s, s + W)):
                for oid, b in tracks[t].items():
                    if oid >= N:
                        continue
                    cx, cy, w, h = bbox_to_cxcywh(b)
                    bbox_in[bi, i, oid] = [cx, cy, w, h]
                    valid_in[bi, i, oid] = 1.0
            last_in = s + W - 1
            for k in range(1, H + 1):
                ft = last_in + k
                for oid, b in tracks[ft].items():
                    if oid >= N:
                        continue
                    cx, cy, _, _ = bbox_to_cxcywh(b)
                    pos_tgt[bi, k - 1, oid] = [cx, cy]
                    valid_tgt[bi, k - 1, oid] = 1.0
            for a, bb, fr in collisions:
                if a < N and bb < N and last_in < fr <= last_in + H:
                    coll_arr[bi, a, bb] = 1.0
                    coll_arr[bi, bb, a] = 1.0
            last_c[bi] = bbox_in[bi, -1, :, :2]
            prev_c[bi] = bbox_in[bi, -2, :, :2] if W >= 2 else last_c[bi]
        valid = valid_in[:, -1]  # [B,N]
        yield {
            "bbox_in": torch.from_numpy(bbox_in),
            "valid_in": torch.from_numpy(valid_in),
            "attr": torch.from_numpy(attr_arr),
            "pos_tgt": torch.from_numpy(pos_tgt),
            "valid_tgt": torch.from_numpy(valid_tgt),
            "valid": torch.from_numpy(valid),
            "coll": torch.from_numpy(coll_arr),
            "last_c": torch.from_numpy(last_c),
            "prev_c": torch.from_numpy(prev_c),
        }


def eval_ckpt(ckpt_path, db, device, window, horizon, max_objs, batch_size, hidden, layers):
    model = EGNNDynamics(d_hid=hidden, n_layers=layers).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    samples = make_windows(db, window=window, horizon=horizon, stride=window, max_objs=max_objs)
    # Position accumulators (per step)
    pos_se_sum = torch.zeros(horizon, device=device)
    pos_se_n = torch.zeros(horizon, device=device)
    # Constant-velocity baseline
    cv_se_sum = torch.zeros(horizon, device=device)
    # Collision accumulators
    coll_logits = []
    coll_labels = []
    coll_geom_dist = []   # last-frame pairwise distance as baseline feature
    coll_bce_sum = 0.0
    coll_bce_n = 0

    with torch.no_grad():
        for batch in batch_samples(samples, batch_size, window, horizon, max_objs):
            bbox_in = batch["bbox_in"].to(device)
            attr = batch["attr"].to(device)
            pos_tgt = batch["pos_tgt"].to(device)
            valid_tgt = batch["valid_tgt"].to(device)
            valid = batch["valid"].to(device)
            coll = batch["coll"].to(device)
            last_c = batch["last_c"].to(device)
            prev_c = batch["prev_c"].to(device)

            pred, coll_logit, _ = model(attr, bbox_in, valid, horizon=horizon)
            # pred: [B,H,N,2]
            vmask = valid_tgt * valid.unsqueeze(1)   # [B,H,N]
            se = ((pred - pos_tgt) ** 2).sum(-1)     # [B,H,N]
            for k in range(horizon):
                pos_se_sum[k] += (se[:, k] * vmask[:, k]).sum()
                pos_se_n[k] += vmask[:, k].sum()
            # Constant-velocity rollout
            vel = last_c - prev_c
            cv_pred = []
            cur = last_c
            for k in range(horizon):
                cur = cur + vel
                cv_pred.append(cur)
            cv_pred = torch.stack(cv_pred, dim=1)
            cv_se = ((cv_pred - pos_tgt) ** 2).sum(-1)
            for k in range(horizon):
                cv_se_sum[k] += (cv_se[:, k] * vmask[:, k]).sum()
            # Collision (upper triangle only, both endpoints valid)
            B, N, _ = coll.shape
            pm = valid.unsqueeze(1) * valid.unsqueeze(2)
            triu = torch.triu(torch.ones(N, N, device=device), diagonal=1).unsqueeze(0).expand(B, -1, -1)
            pm = pm * triu
            mask = pm > 0.5
            if mask.any():
                logits_v = coll_logit[mask].float()
                labels_v = coll[mask].float()
                xd = last_c.unsqueeze(2) - last_c.unsqueeze(1)
                d2 = (xd * xd).sum(-1)
                dist_v = d2.sqrt()[mask].float()
                coll_logits.append(logits_v.cpu())
                coll_labels.append(labels_v.cpu())
                coll_geom_dist.append(dist_v.cpu())
                bce = F.binary_cross_entropy_with_logits(logits_v, labels_v, reduction="sum")
                coll_bce_sum += bce.item()
                coll_bce_n += logits_v.numel()

    pos_mse = (pos_se_sum / pos_se_n.clamp_min(1)).cpu().numpy()
    cv_mse = (cv_se_sum / pos_se_n.clamp_min(1)).cpu().numpy()
    logits_all = torch.cat(coll_logits).numpy()
    labels_all = torch.cat(coll_labels).numpy()
    dist_all = torch.cat(coll_geom_dist).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_all))

    # Metrics
    def auroc(y, s):
        # rank-based
        order = np.argsort(-s)
        y = y[order]
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        if pos == 0 or neg == 0:
            return float("nan")
        tp = np.cumsum(y == 1)
        fp = np.cumsum(y == 0)
        tpr = tp / pos
        fpr = fp / neg
        return float(np.trapz(tpr, fpr))

    def ap(y, s):
        order = np.argsort(-s)
        y = y[order]
        pos = (y == 1).sum()
        if pos == 0:
            return float("nan")
        tp = np.cumsum(y == 1)
        prec = tp / (np.arange(len(y)) + 1)
        rec = tp / pos
        d = np.diff(np.concatenate([[0], rec]))
        return float((prec * d).sum())

    auc = auroc(labels_all, probs)
    ap_v = ap(labels_all, probs)
    pred_pos = (probs >= 0.5).astype(np.float32)
    tp = ((pred_pos == 1) & (labels_all == 1)).sum()
    fp = ((pred_pos == 1) & (labels_all == 0)).sum()
    fn = ((pred_pos == 0) & (labels_all == 1)).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    bce = coll_bce_sum / max(1, coll_bce_n)

    # Geometric baseline (closer pairs = more collision likely): use -dist as score
    auc_geom = auroc(labels_all, -dist_all)
    ap_geom = ap(labels_all, -dist_all)

    return {
        "pos_mse_per_k": pos_mse.tolist(),
        "pos_rmse_px_per_k": [float(np.sqrt(m) * FRAME_W) for m in pos_mse],  # approx (cx space)
        "cv_mse_per_k": cv_mse.tolist(),
        "cv_rmse_px_per_k": [float(np.sqrt(m) * FRAME_W) for m in cv_mse],
        "coll_bce": bce,
        "coll_auc": auc,
        "coll_ap": ap_v,
        "coll_f1@0.5": float(f1),
        "coll_prec@0.5": float(prec),
        "coll_rec@0.5": float(rec),
        "coll_pos_rate": float(labels_all.mean()),
        "coll_n_pairs": int(labels_all.size),
        "coll_auc_geom_baseline": auc_geom,
        "coll_ap_geom_baseline": ap_geom,
        "n_samples": len(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-videos", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--max-objs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"Building val cache ({args.max_videos} videos)...", flush=True)
    db = build_val_cache(args.max_videos, args.workers)

    print(f"Running EGNN eval on ckpt={args.ckpt}", flush=True)
    device = torch.device(args.device)
    res = eval_ckpt(
        args.ckpt, db, device,
        window=args.window, horizon=args.horizon, max_objs=args.max_objs,
        batch_size=args.batch_size, hidden=args.hidden, layers=args.layers,
    )
    print("==========================================")
    print("  EGNN STANDALONE EVAL")
    print("==========================================")
    print(f"  ckpt         : {args.ckpt}")
    print(f"  n_videos     : {len(db)}  n_windows: {res['n_samples']}")
    print(f"  --- POSITION ---")
    for k, (m, mb) in enumerate(zip(res['pos_rmse_px_per_k'], res['cv_rmse_px_per_k'])):
        print(f"  k={k+1:d}: EGNN RMSE(px,cx)={m:.1f}  const-vel baseline={mb:.1f}")
    print(f"  --- COLLISION ---")
    print(f"  n_pairs      : {res['coll_n_pairs']}  pos_rate: {res['coll_pos_rate']*100:.3f}%")
    print(f"  BCE          : {res['coll_bce']:.4f}")
    print(f"  AUROC        : {res['coll_auc']:.4f}  (geom-dist baseline: {res['coll_auc_geom_baseline']:.4f})")
    print(f"  AP           : {res['coll_ap']:.4f}  (geom-dist baseline: {res['coll_ap_geom_baseline']:.4f})")
    print(f"  P/R/F1@0.5   : {res['coll_prec@0.5']:.4f} / {res['coll_rec@0.5']:.4f} / {res['coll_f1@0.5']:.4f}")
    print("==========================================")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
