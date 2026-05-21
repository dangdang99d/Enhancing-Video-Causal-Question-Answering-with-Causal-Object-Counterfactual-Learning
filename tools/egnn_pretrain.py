"""EGNN dynamics pretraining on GT tracks.

Inputs: pickle cache built by build_dynamics_cache.py

Node features:
  - attr one-hot (13 = 3 shapes + 8 colors + 2 materials)
  - last-frame velocity (2) from bbox-center finite difference
    (NOT GT velocity — computed from bbox centers to match inference path)

Equivariant coords: 2D bbox center (cx, cy) in [0,1].

Targets (full trajectory over horizon H):
  - positions @ t+1 .. t+H: MSE on (cx,cy), autoregressive rollout
  - collision in (t, t+H]: pairwise BCE using final-step embedding

Loss = mean_k L_pos(k) + lambda_coll * L_coll
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

FRAME_W, FRAME_H = 480.0, 320.0  # CLEVRER frames are 480x320 (confirmed via dataset)

ATTR_DIM = 13  # 3 shape + 8 color + 2 material


def bbox_to_cxcywh(b):
    cx = (b[0] + b[2]) / 2.0 / FRAME_W
    cy = (b[1] + b[3]) / 2.0 / FRAME_H
    w = (b[2] - b[0]) / FRAME_W
    h = (b[3] - b[1]) / FRAME_H
    return cx, cy, w, h


class DynamicsDataset(Dataset):
    def __init__(self, cache_path, window=8, horizon=4, stride=8, max_objs=8):
        t0 = time.time()
        with open(cache_path, "rb") as f:
            self.db = pickle.load(f)
        print(f"[ds] loaded {len(self.db)} videos in {time.time()-t0:.1f}s")
        self.vids = sorted(self.db.keys())
        self.window = window
        self.horizon = horizon
        self.max_objs = max_objs
        self.index = []
        for vid in self.vids:
            n = len(self.db[vid]["tracks"])
            last = n - 1 - horizon
            for s in range(0, max(0, last - window + 1), stride):
                self.index.append((vid, s))
        print(f"[ds] {len(self.index)} samples (W={window}, H={horizon}, stride={stride})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vid, s = self.index[idx]
        e = self.db[vid]
        tracks = e["tracks"]
        attrs = e["attrs"]
        N = self.max_objs
        W = self.window
        H = self.horizon

        bbox_in = np.zeros((W, N, 4), np.float32)
        valid_in = np.zeros((W, N), np.float32)
        attr_arr = np.zeros((N, ATTR_DIM), np.float32)
        for oid, a in attrs.items():
            if oid < N:
                attr_arr[oid] = np.asarray(a, np.float32)

        for i, t in enumerate(range(s, s + W)):
            for oid, b in tracks[t].items():
                if oid >= N:
                    continue
                cx, cy, w, h = bbox_to_cxcywh(b)
                bbox_in[i, oid] = [cx, cy, w, h]
                valid_in[i, oid] = 1.0

        # Trajectory targets: positions at t+1..t+H
        pos_tgt = np.zeros((H, N, 2), np.float32)
        valid_tgt = np.zeros((H, N), np.float32)
        last_in = s + W - 1
        for k in range(1, H + 1):
            ft = last_in + k
            for oid, b in tracks[ft].items():
                if oid >= N:
                    continue
                cx, cy, _, _ = bbox_to_cxcywh(b)
                pos_tgt[k - 1, oid] = [cx, cy]
                valid_tgt[k - 1, oid] = 1.0

        # Object-level validity for starting rollout: need last input frame
        valid = valid_in[-1]

        coll = np.zeros((N, N), np.float32)
        for a, b, fr in e["collisions"]:
            if a < N and b < N and last_in < fr <= last_in + H:
                coll[a, b] = 1.0
                coll[b, a] = 1.0

        return {
            "bbox_in": torch.from_numpy(bbox_in),
            "valid_in": torch.from_numpy(valid_in),
            "attr": torch.from_numpy(attr_arr),
            "pos_tgt": torch.from_numpy(pos_tgt),
            "valid_tgt": torch.from_numpy(valid_tgt),
            "valid": torch.from_numpy(valid),
            "coll": torch.from_numpy(coll),
        }


class EGNNLayer(nn.Module):
    def __init__(self, d_node, d_hid):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_node + 1, d_hid), nn.SiLU(),
            nn.Linear(d_hid, d_hid), nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(d_hid, d_hid), nn.SiLU(),
            nn.Linear(d_hid, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(d_node + d_hid, d_hid), nn.SiLU(),
            nn.Linear(d_hid, d_node),
        )

    def forward(self, h, x, mask):
        # h: [B,N,D]  x: [B,N,2]  mask: [B,N]
        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        dx = x.unsqueeze(2) - x.unsqueeze(1)  # [B,N,N,2]
        d2 = (dx * dx).sum(-1, keepdim=True)
        e = self.edge_mlp(torch.cat([hi, hj, d2], -1))
        pm = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        eye = torch.eye(N, device=h.device).unsqueeze(0).unsqueeze(-1)
        pm = pm * (1.0 - eye)
        e = e * pm
        denom = pm.sum(dim=2).clamp_min(1.0)
        cw = self.coord_mlp(e) * pm
        x_new = x + (dx * cw).sum(dim=2) / denom
        m_agg = e.sum(dim=2) / denom
        h_new = h + self.node_mlp(torch.cat([h, m_agg], -1))
        return h_new, x_new


class EGNNDynamics(nn.Module):
    def __init__(self, attr_dim=ATTR_DIM, d_hid=128, n_layers=4):
        super().__init__()
        self.in_dim = attr_dim + 2  # attr + vel(2); size not used
        self.node_in = nn.Linear(self.in_dim, d_hid)
        self.layers = nn.ModuleList([EGNNLayer(d_hid, d_hid) for _ in range(n_layers)])
        self.coll_head = nn.Sequential(
            nn.Linear(2 * d_hid + 1, d_hid), nn.SiLU(),
            nn.Linear(d_hid, 1),
        )

    def forward(self, attr, bbox_in, valid, horizon=4):
        # bbox_in: [B,W,N,4]=cxcywh  valid: [B,N]  -> autoregressive rollout of H steps
        B, W, N, _ = bbox_in.shape
        cur_c = bbox_in[:, -1, :, :2]
        prev_c = bbox_in[:, -2, :, :2] if W >= 2 else cur_c
        preds = []
        h_final = None
        for k in range(horizon):
            vel = cur_c - prev_c
            h = self.node_in(torch.cat([attr, vel], -1))
            x = cur_c
            for layer in self.layers:
                h, x = layer(h, x, valid)
            preds.append(x)
            prev_c = cur_c
            cur_c = x
            h_final = h
        pred_traj = torch.stack(preds, dim=1)  # [B,H,N,2]
        hi = h_final.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h_final.unsqueeze(1).expand(-1, N, -1, -1)
        xd = cur_c.unsqueeze(2) - cur_c.unsqueeze(1)
        d2 = (xd * xd).sum(-1, keepdim=True)
        coll_logit = self.coll_head(torch.cat([hi, hj, d2], -1)).squeeze(-1)
        return pred_traj, coll_logit, h_final


def train_run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-objs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lambda-coll", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = DynamicsDataset(
        args.cache, window=args.window, horizon=args.horizon,
        stride=args.stride, max_objs=args.max_objs,
    )
    # simple split by index (deterministic, same-video samples may land in both; fine for pretrain)
    n = len(ds)
    n_val = int(n * args.val_frac)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_idx = [i for i in perm[n_val:]]
    val_idx = list(val_idx)
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    print(f"[split] train={len(train_ds)} val={len(val_ds)}")

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device(args.device)
    model = EGNNDynamics(d_hid=args.hidden, n_layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_params:,}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val = float("inf")

    for ep in range(args.epochs):
        model.train()
        ep_start = time.time()
        run = defaultdict(float)
        for batch in loader:
            bbox_in = batch["bbox_in"].to(device, non_blocking=True)
            attr = batch["attr"].to(device, non_blocking=True)
            pos_tgt = batch["pos_tgt"].to(device, non_blocking=True)     # [B,H,N,2]
            valid_tgt = batch["valid_tgt"].to(device, non_blocking=True) # [B,H,N]
            valid = batch["valid"].to(device, non_blocking=True)         # [B,N]
            coll = batch["coll"].to(device, non_blocking=True)

            pred, coll_logit, _ = model(attr, bbox_in, valid, horizon=args.horizon)
            vmask = valid_tgt * valid.unsqueeze(1)                       # [B,H,N]
            pos_err = ((pred - pos_tgt) ** 2).sum(-1)                    # [B,H,N]
            pos_loss = (pos_err * vmask).sum() / vmask.sum().clamp_min(1)
            pm = valid.unsqueeze(1) * valid.unsqueeze(2)
            eye = torch.eye(coll.shape[-1], device=device).unsqueeze(0)
            pm = pm * (1 - eye)
            coll_bce = F.binary_cross_entropy_with_logits(coll_logit, coll, reduction="none")
            coll_loss = (coll_bce * pm).sum() / pm.sum().clamp_min(1)

            loss = pos_loss + args.lambda_coll * coll_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run["pos"] += pos_loss.item()
            run["coll"] += coll_loss.item()
            # last-step pos err separately for diagnostic
            last_err = (pos_err[:, -1] * vmask[:, -1]).sum() / vmask[:, -1].sum().clamp_min(1)
            run["pos_last"] += last_err.item()
            run["n"] += 1
        sched.step()

        # val
        model.eval()
        v = defaultdict(float)
        with torch.no_grad():
            for batch in val_loader:
                bbox_in = batch["bbox_in"].to(device)
                attr = batch["attr"].to(device)
                pos_tgt = batch["pos_tgt"].to(device)
                valid_tgt = batch["valid_tgt"].to(device)
                valid = batch["valid"].to(device)
                coll = batch["coll"].to(device)
                pred, coll_logit, _ = model(attr, bbox_in, valid, horizon=args.horizon)
                vmask = valid_tgt * valid.unsqueeze(1)
                pos_err = ((pred - pos_tgt) ** 2).sum(-1)
                pos_loss = (pos_err * vmask).sum() / vmask.sum().clamp_min(1)
                pm = valid.unsqueeze(1) * valid.unsqueeze(2)
                eye = torch.eye(coll.shape[-1], device=device).unsqueeze(0)
                pm = pm * (1 - eye)
                coll_bce = F.binary_cross_entropy_with_logits(coll_logit, coll, reduction="none")
                coll_loss = (coll_bce * pm).sum() / pm.sum().clamp_min(1)
                last_err = (pos_err[:, -1] * vmask[:, -1]).sum() / vmask[:, -1].sum().clamp_min(1)
                v["pos"] += pos_loss.item()
                v["coll"] += coll_loss.item()
                v["pos_last"] += last_err.item()
                v["n"] += 1

        pos_tr = run["pos"] / max(1, run["n"])
        last_tr = run["pos_last"] / max(1, run["n"])
        co_tr = run["coll"] / max(1, run["n"])
        total_tr = pos_tr + args.lambda_coll * co_tr
        pos_v = v["pos"] / max(1, v["n"])
        last_v = v["pos_last"] / max(1, v["n"])
        co_v = v["coll"] / max(1, v["n"])
        total_v = pos_v + args.lambda_coll * co_v
        dt = time.time() - ep_start
        print(
            f"[ep {ep+1}/{args.epochs}] "
            f"tr pos={pos_tr:.4f} (last={last_tr:.4f}) co={co_tr:.4f} tot={total_tr:.4f} | "
            f"val pos={pos_v:.4f} (last={last_v:.4f}) co={co_v:.4f} tot={total_v:.4f} "
            f"({dt:.1f}s lr={sched.get_last_lr()[0]:.2e})",
            flush=True,
        )
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"epoch_{ep+1:02d}.pt"))
        if total_v < best_val:
            best_val = total_v
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    train_run()
