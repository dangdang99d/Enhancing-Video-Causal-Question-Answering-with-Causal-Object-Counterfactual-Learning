"""Frozen EGNN runtime for in-graph dynamics features.

Loads EGNNDynamics best.pt and exposes a per-frame forward producing:
  dyn_embd:   [B, F, N, D=128]  per-object dynamics hidden (post-encoder)
  coll_logit: [B, F, N, N]      per-object pairwise collision logit
For masked-path counterfactuals, call twice — once with full input, once with
cause_obj coords/attrs zeroed at the input level (no leakage via msg passing).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tools.egnn_pretrain import EGNNDynamics, ATTR_DIM, FRAME_W, FRAME_H


# 48-class label index → 13d (3 shape + 8 color + 2 material) one-hot.
# Mirrors clevrer_label_vocab.build_clevrer_attr_vocab() iteration order:
#   for c in COLORS_RAW: for s in SHAPES: for m in MATERIALS  (innermost = material)
# COLORS_RAW has 10 entries (gold/silver collapse) → 48 unique post-norm.
_COLORS_RAW = ("gray", "red", "blue", "green", "brown", "cyan",
               "purple", "yellow", "gold", "silver")
_COLORS = ("blue", "cyan", "gray", "brown", "green", "purple", "red", "yellow")
_SHAPES = ("cube", "sphere", "cylinder")
_MATERIALS = ("metal", "rubber")
_MAT_ATTR = ("rubber", "metal")  # EGNN attr_onehot order: rubber index 0, metal 1


def _norm_color(c: str) -> str:
    if c == "gold":
        return "yellow"
    if c == "silver":
        return "gray"
    return c


def _build_label_to_attr_lookup() -> torch.Tensor:
    """Returns [48, 13] float lookup: label idx -> attr one-hot."""
    table = torch.zeros(48, ATTR_DIM, dtype=torch.float32)
    seen = {}
    i = 0
    for c in _COLORS_RAW:
        for s in _SHAPES:
            for m in _MATERIALS:
                key = (_norm_color(c), s, m)
                if key in seen:
                    continue
                seen[key] = i
                vec = torch.zeros(ATTR_DIM)
                vec[_SHAPES.index(s)] = 1.0
                vec[len(_SHAPES) + _COLORS.index(_norm_color(c))] = 1.0
                vec[len(_SHAPES) + len(_COLORS) + _MAT_ATTR.index(m)] = 1.0
                table[i] = vec
                i += 1
    return table


class FrozenEGNN(nn.Module):
    """Frozen EGNN dynamics encoder for IPRM in-graph use.

    Always runs no_grad (params have requires_grad=False). The encoder is invoked
    with horizon=1 so we get one h_final per frame — no rollout needed at inference.
    """

    def __init__(self, ckpt_path: str, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        self.model = EGNNDynamics(d_hid=hidden, n_layers=n_layers)
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[FrozenEGNN] missing={len(missing)} unexpected={len(unexpected)}")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.register_buffer("label_to_attr", _build_label_to_attr_lookup(),
                             persistent=False)
        self.hidden = hidden

    def label_idx_to_attr_oh(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: [..., ] int -> [..., 13] one-hot attr."""
        return self.label_to_attr[labels.clamp(min=0, max=47)]

    @torch.no_grad()
    def forward_per_frame(
        self,
        bbox_xyxy: torch.Tensor,
        attr_oh: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run EGNN encoder per frame using a 2-frame window (prev, cur).

        Args:
          bbox_xyxy: [B, F, N, 4] in pixel xyxy
          attr_oh:   [B, F, N, 13] one-hot attr (per-frame in case label changes)
          valid:     [B, F, N] {0,1}
        Returns:
          dyn_embd:   [B, F, N, hidden]
          coll_logit: [B, F, N, N]
        """
        B, F, N, _ = bbox_xyxy.shape
        cx = (bbox_xyxy[..., 0] + bbox_xyxy[..., 2]) / 2.0 / FRAME_W
        cy = (bbox_xyxy[..., 1] + bbox_xyxy[..., 3]) / 2.0 / FRAME_H
        w = (bbox_xyxy[..., 2] - bbox_xyxy[..., 0]).clamp(min=0) / FRAME_W
        h = (bbox_xyxy[..., 3] - bbox_xyxy[..., 1]).clamp(min=0) / FRAME_H
        cxcywh = torch.stack([cx, cy, w, h], dim=-1)  # [B,F,N,4]

        prev = torch.cat([cxcywh[:, :1], cxcywh[:, :-1]], dim=1)  # [B,F,N,4]
        bbox_in = torch.stack([prev, cxcywh], dim=2)  # [B,F,W=2,N,4]
        bbox_in = bbox_in.reshape(B * F, 2, N, 4)
        attr_flat = attr_oh.reshape(B * F, N, ATTR_DIM)
        valid_flat = valid.reshape(B * F, N)

        _, coll, h_final = self.model(attr_flat, bbox_in, valid_flat, horizon=1)
        dyn_embd = h_final.reshape(B, F, N, self.hidden)
        coll_logit = coll.reshape(B, F, N, N)
        return dyn_embd, coll_logit

    @torch.no_grad()
    def dual_forward(
        self,
        bbox_xyxy: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
        cause_obj_mask: torch.Tensor | None,
    ) -> dict:
        """Run EGNN twice: full input, and cause-zeroed input.

        Args:
          bbox_xyxy: [B, F, N, 4] pixel xyxy
          labels:    [B, F, N] int 48-class
          valid:     [B, F, N] {0,1}
          cause_obj_mask: [B, F, N] {0,1} where 1 = cause obj to MASK
        Returns dict with dyn_embd_full / coll_full / dyn_embd_masked / coll_masked.
          If cause_obj_mask is None, the masked outputs equal the full outputs.
        """
        attr_oh = self.label_idx_to_attr_oh(labels)  # [B,F,N,13]
        dyn_full, coll_full = self.forward_per_frame(bbox_xyxy, attr_oh, valid)
        if cause_obj_mask is None:
            return {
                "dyn_embd_full": dyn_full, "coll_full": coll_full,
                "dyn_embd_masked": dyn_full, "coll_masked": coll_full,
            }
        keep = (1.0 - cause_obj_mask).unsqueeze(-1)  # [B,F,N,1]
        bbox_m = bbox_xyxy * keep
        attr_m = attr_oh * keep
        valid_m = valid * (1.0 - cause_obj_mask)
        dyn_m, coll_m = self.forward_per_frame(bbox_m, attr_m, valid_m)
        return {
            "dyn_embd_full": dyn_full, "coll_full": coll_full,
            "dyn_embd_masked": dyn_m, "coll_masked": coll_m,
        }
