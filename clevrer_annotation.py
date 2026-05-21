"""Official CLEVRER annotation JSON — schema and IPRM/paper-aligned field usage.

Full MIT files contain (example ``annotation_00000.json``):

- ``object_property``: static attrs per object — **USED** (color, shape, material).
- ``motion_trajectory``: list of frames, each with ``frame_id`` and ``objects`` — **USED**:
  ``object_id``, ``location`` (→ 2D bbox heuristic), ``inside_camera_view``.
- ``collision``: event list — **NOT** fed to IPRM (paper: no event / program graph in the model).
- ``scene_index``, ``video_filename`` — metadata; **NOT** used in the visual tensor pipeline.
- Per-object kinematics ``orientation``, ``velocity``, ``angular_velocity`` — **NOT** used
  (paper visual stream: Faster R-CNN bbox + discrete attrs only; we mirror that with GT).

After load we keep only ``object_property`` and ``motion_trajectory`` to save RAM in the LRU cache.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np


def video_stem_to_index(video_filename: str) -> int:
    base = os.path.basename(video_filename)
    return int(base.replace("video_", "").replace(".mp4", ""))


def clevrer_annotation_subpath(video_index: int) -> tuple[str, str]:
    bucket_lo = (video_index // 1000) * 1000
    bucket_hi = bucket_lo + 1000
    subdir = f"annotation_{bucket_lo:05d}-{bucket_hi:05d}"
    fname = f"annotation_{video_index:05d}.json"
    return subdir, fname


def resolve_annotation_path(annotation_root: str, video_filename: str) -> str | None:
    if not annotation_root:
        return None
    idx = video_stem_to_index(video_filename)
    subdir, fname = clevrer_annotation_subpath(idx)
    p = os.path.normpath(os.path.join(annotation_root, subdir, fname))
    if os.path.isfile(p):
        return os.path.abspath(p)
    flat = os.path.normpath(os.path.join(annotation_root, fname))
    if os.path.isfile(flat):
        return os.path.abspath(flat)
    return None


def validate_clevrer_annotation_for_iprm(raw: dict[str, Any], path: str = "") -> None:
    """Ensure JSON has the slices IPRM uses; ignore ``collision`` and kinematics."""
    hint = f" ({path})" if path else ""
    if "object_property" not in raw or "motion_trajectory" not in raw:
        raise ValueError(
            f"CLEVRER annotation must contain 'object_property' and 'motion_trajectory'{hint}"
        )
    if not isinstance(raw["object_property"], list) or not raw["object_property"]:
        raise ValueError(f"'object_property' must be a non-empty list{hint}")
    traj = raw["motion_trajectory"]
    if not isinstance(traj, list) or not traj:
        raise ValueError(f"'motion_trajectory' must be a non-empty list{hint}")
    frame0 = traj[0]
    if "objects" not in frame0 or not isinstance(frame0["objects"], list):
        raise ValueError(f"motion_trajectory[0] must have 'objects' list{hint}")
    if frame0["objects"]:
        o0 = frame0["objects"][0]
        for k in ("object_id", "location"):
            if k not in o0:
                raise ValueError(f"each frame object needs '{k}'{hint}")


def slim_annotation_for_iprm(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop fields not used by the VLM (saves memory when caching many videos)."""
    return {
        "object_property": raw["object_property"],
        "motion_trajectory": raw["motion_trajectory"],
    }


@lru_cache(maxsize=512)
def load_annotation_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    validate_clevrer_annotation_for_iprm(raw, path)
    return slim_annotation_for_iprm(raw)


# CLEVRER 3×4 camera projection matrix (from Blender scene used to render 480×320 videos).
# Source: VRDP (Ding et al.) — dynamics/physics_simulation.py
_CLEVRER_CAM_MAT = np.array([
    [-207.8461456298828,  525.0000610351562, -120.00001525878906, 1200.0003662109375],
    [ 123.93595886230469,   1.832598354667425e-05, -534.663330078125, 799.9999389648438],
    [  -0.866025447845459, -3.650024282819686e-08,   -0.4999999701976776, 5.000000476837158],
], dtype=np.float64)


def _project_3d_to_2d(x3d: float, y3d: float, z3d: float = 0.2) -> tuple[float, float]:
    """Project a CLEVRER 3D world coordinate to 2D pixel space (480x320).

    Returns (px, py) where px is in [0, 480) and py is in [0, 320).
    Uses the camera projection matrix extracted from the Blender scene.
    """
    pos = np.array([x3d, y3d, z3d, 1.0], dtype=np.float64)
    uv = _CLEVRER_CAM_MAT @ pos          # [u, v, w]
    px = uv[0] / uv[2]                   # perspective divide -> pixel x
    py = uv[1] / uv[2]                   # perspective divide -> pixel y
    return float(px), float(py)


def _project_location_to_bbox_xyxy(
    location: list[float] | tuple[float, ...],
    frame_w: float = 480.0,
    frame_h: float = 320.0,
) -> list[float]:
    """Project simulator ``location`` to 2D xyxy in pixel space (CLEVRER 480×320)."""
    px, py = _project_3d_to_2d(float(location[0]), float(location[1]),
                                float(location[2]) if len(location) > 2 else 0.2)
    bw, bh = 25, 28
    x1 = max(0.0, px - bw)
    y1 = max(0.0, py - bh)
    x2 = min(frame_w, px + bw)
    y2 = min(frame_h, py + bh)
    return [x1, y1, x2, y2]


def objects_visible_at_frame(
    ann: dict[str, Any],
    frame_idx: int,
    frame_w: float = 480.0,
    frame_h: float = 320.0,
) -> list[tuple[int, list[float], str, str, str]]:
    """
    Per-frame objects for the paper-style stream: bbox + (color, shape, material).
    Skips ``inside_camera_view`` false. Does not use velocity/orientation/collision.
    """
    traj = ann["motion_trajectory"]
    frame = traj[frame_idx]
    id_to_prop = {int(o["object_id"]): o for o in ann["object_property"]}
    out: list[tuple[int, list[float], str, str, str]] = []
    for o in frame["objects"]:
        oid = int(o["object_id"])
        if not o.get("inside_camera_view", True):
            continue
        if oid not in id_to_prop:
            continue
        prop = id_to_prop[oid]
        bbox = _project_location_to_bbox_xyxy(o["location"], frame_w, frame_h)
        out.append(
            (
                oid,
                bbox,
                str(prop["color"]),
                str(prop["shape"]),
                str(prop["material"]),
            )
        )
    out.sort(key=lambda t: t[0])
    return out
