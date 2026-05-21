"""
Precomputed *detector-style* visual input for CLEVRER (paper: Faster R-CNN bbox + predicted attrs).

Each video one JSON file. This matches the paper’s interface (bbox + discrete attributes per object),
not the raw MIT ``annotation_*.json`` schema.

File layout (either is accepted):

1) Flat next to root::

    DETECTOR_JSON_ROOT/video_00000.json

2) Same bucket folders as MIT annotations::

    DETECTOR_JSON_ROOT/annotation_00000-01000/video_00000.json

JSON schema::

    {
      "per_frame": [
        [ {"bbox": [x1, y1, x2, y2], "color": "blue", "shape": "sphere", "material": "rubber"}, ... ],
        ... 128 lists ...
      ]
    }

``bbox`` is xyxy in **pixel** coordinates on 480×320, same as dataloader ``frame_resolutions``.

Obtain files by running your Faster R-CNN (CLEVRER-trained) pipeline and exporting to this schema.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from clevrer_annotation import video_stem_to_index, clevrer_annotation_subpath


def resolve_detector_json_path(root: str, video_filename: str) -> str | None:
    if not root or not video_filename:
        return None
    stem = video_filename.replace(".mp4", "").replace(".MP4", "")
    flat = os.path.normpath(os.path.join(root, f"{stem}.json"))
    if os.path.isfile(flat):
        return os.path.abspath(flat)
    idx = video_stem_to_index(video_filename)
    subdir, _ = clevrer_annotation_subpath(idx)
    nested = os.path.normpath(os.path.join(root, subdir, f"{stem}.json"))
    if os.path.isfile(nested):
        return os.path.abspath(nested)
    return None


@lru_cache(maxsize=512)
def load_detector_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "per_frame" not in raw or not isinstance(raw["per_frame"], list):
        raise ValueError(f"detector JSON must contain non-empty list 'per_frame': {path}")
    return raw


def objects_at_frame_from_detector(
    det: dict[str, Any], frame_idx: int, max_objs: int
) -> list[tuple[int, list[float], str, str, str]]:
    """Return (dummy_id, bbox, color, shape, material) compatible with dataset stacking."""
    frames = det["per_frame"]
    fi = min(max(0, frame_idx), len(frames) - 1)
    objs = frames[fi]
    if not isinstance(objs, list):
        return []
    out: list[tuple[int, list[float], str, str, str]] = []
    for oi, o in enumerate(objs[:max_objs]):
        if not isinstance(o, dict):
            continue
        bb = o["bbox"]
        bbox = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
        out.append(
            (
                oi,
                bbox,
                str(o["color"]),
                str(o["shape"]),
                str(o["material"]),
            )
        )
    return out
