"""
1-choice pair dataset v2: collision + velocity encoded into relation tensors.

obj_contacting_rels (19 dims) encoding per object per frame:
  [0]    : 1.0 if this object is involved in a collision at/near this frame
  [1:3]  : collision partner's normalized location (x, y)
  [3:5]  : collision location (x, y) normalized
  [5]    : number of frames since/until nearest collision (signed, / 128)
  [6:18] : reserved (zero)
  [18]   : 1.0 if object is inside camera view

obj_spatial_rels (8 dims) encoding per object per frame:
  [0:2]  : object velocity (vx, vy) normalized
  [2:4]  : object location (x, y) from motion_trajectory, normalized
  [4:6]  : object angular velocity (wx, wy) normalized
  [6:8]  : reserved (zero)
"""
from __future__ import annotations
import glob, json, os, pickle, warnings
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from clevrer_annotation import load_annotation_json, objects_visible_at_frame, resolve_annotation_path
from clevrer_detector_json import load_detector_json, objects_at_frame_from_detector, resolve_detector_json_path
from clevrer_label_vocab import attr_tuple_to_label_index


def _num_obj_label_embeddings(cfg) -> int:
    path = cfg.VLM.OBJ_LABEL_EMBD_LAYER_PATH
    here = os.path.dirname(os.path.abspath(__file__))
    path = path if os.path.isabs(path) else os.path.join(here, path)
    try:
        w = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError, RuntimeError):
        w = torch.load(path, map_location="cpu")
    if isinstance(w, torch.Tensor):
        return int(w.shape[0])
    return int(w.weight.shape[0])


def _paper_stride_frame_indices(n_given, num_target_frames, stride):
    raw = list(range(0, max(n_given, 1), stride))
    raw = raw[:num_target_frames]
    while len(raw) < num_target_frames:
        raw.append(raw[-1] if raw else 0)
    num_unpadded = min(num_target_frames, len(list(range(0, n_given, stride))))
    return raw, num_unpadded


def _detector_dir_has_json(det_root):
    if not det_root or not os.path.isdir(det_root):
        return False
    return len(glob.glob(os.path.join(det_root, "*.json"))) > 0


def _build_collision_index(ann):
    """Build per-object collision lookup: obj_id -> list of (frame_id, partner_id, location)."""
    collisions = ann.get("collision", [])
    obj_collisions = {}
    for c in collisions:
        ids = c["object_ids"]
        fid = c["frame_id"]
        loc = c.get("location", [0, 0, 0])
        for i, oid in enumerate(ids):
            partner = ids[1 - i] if len(ids) == 2 else -1
            obj_collisions.setdefault(oid, []).append((fid, partner, loc))
    return obj_collisions


def _get_motion_at_frame(ann, frame_id, obj_id):
    """Get velocity and location from motion_trajectory for an object at a frame."""
    mt = ann.get("motion_trajectory", [])
    if frame_id >= len(mt):
        return [0, 0], [0, 0], [0, 0], False
    frame_data = mt[frame_id]
    for obj in frame_data.get("objects", []):
        if obj["object_id"] == obj_id:
            vel = obj.get("velocity", [0, 0, 0])[:2]
            loc = obj.get("location", [0, 0, 0])[:2]
            ang_vel = obj.get("angular_velocity", [0, 0, 0])[:2]
            inside = obj.get("inside_camera_view", True)
            return vel, loc, ang_vel, inside
    return [0, 0], [0, 0], [0, 0], False


VEL_SCALE = 5.0
LOC_SCALE = 5.0
COLLISION_WINDOW = 4  # frames within stride to count as "at this frame"

_CLEVRER_COLORS = {"gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow", "gold", "silver"}
_CLEVRER_SHAPES = {"cube", "sphere", "cylinder", "ball"}
_SHAPE_NORM = {"ball": "sphere"}


def _parse_cause_object(choice_text: str):
    """Extract the subject (cause) object's (color, shape) from choice text.

    Choices follow pattern: 'the <color> <shape> <verb> ...'
    Returns (color, shape) or None if parsing fails.
    """
    words = choice_text.lower().strip().split()
    for i, w in enumerate(words):
        if w == "the" and i + 2 < len(words):
            c, s = words[i + 1], words[i + 2]
            if c in _CLEVRER_COLORS and s in _CLEVRER_SHAPES:
                return c, _SHAPE_NORM.get(s, s)
    return None


class ClevrerPairDatasetV2(Dataset):
    def __init__(self, cfg, pairs_json_path, annotation_root, num_label_embeddings,
                 split_name="train", visual_source="annotation", detector_root=""):
        super().__init__()
        self.cfg = cfg
        self.annotation_root = annotation_root
        self.detector_root = detector_root
        self.visual_source = visual_source.lower().strip()
        self.num_label_embeddings = num_label_embeddings
        self.split_name = split_name
        self.num_sample_frames = int(cfg.DATALOADER.NUM_SAMPLE_FRAMES)
        self.frame_stride = int(getattr(cfg.DATALOADER, "CLEVRER_FRAME_STRIDE", 4))
        self.max_objs = int(getattr(cfg.DATALOADER, "CLEVRER_MAX_OBJECTS", 16))
        self.num_spatial = int(cfg.VLM.NUM_OBJ_SPATIAL_RELS)
        self.num_contact = int(cfg.VLM.NUM_OBJ_CONTACT_RELS)

        with open(pairs_json_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)
        if not self.pairs:
            raise RuntimeError(f"No pairs in {pairs_json_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        p = self.pairs[index]
        video_filename = p["video_filename"]

        # Always load annotation for collision + motion data
        ann_path = resolve_annotation_path(self.annotation_root, video_filename)
        if not ann_path:
            raise FileNotFoundError(f"Annotation missing for {video_filename}")
        ann = load_annotation_json(ann_path)
        traj_len = len(ann["motion_trajectory"])

        # Detector JSON for bbox (if available)
        det = None
        if self.visual_source == "detector_json":
            det_path = resolve_detector_json_path(self.detector_root, video_filename)
            if det_path:
                det = load_detector_json(det_path)

        frame_idxs, num_unpadded_frames = _paper_stride_frame_indices(
            traj_len, self.num_sample_frames, self.frame_stride)

        # Build collision index
        obj_collisions = _build_collision_index(ann)
        obj_props = {o["object_id"]: o for o in ann.get("object_property", [])}

        cause_parsed = _parse_cause_object(p["choice"])

        # Find cause_obj_id for leakage-free contact
        cause_obj_id = None
        if cause_parsed is not None:
            for oid, prop in obj_props.items():
                nc = prop.get("color", "").lower().strip()
                if nc == "gold": nc = "yellow"
                if nc == "silver": nc = "gray"
                ns = prop.get("shape", "").lower().strip()
                if ns == "ball": ns = "sphere"
                if nc == cause_parsed[0] and ns == cause_parsed[1]:
                    cause_obj_id = oid
                    break

        # (obj_id, real_frame) where contact partner is cause_obj -> leakage when masking
        leakage_key_set = set()
        if cause_obj_id is not None:
            for oid, col_list in obj_collisions.items():
                if oid == cause_obj_id:
                    continue
                for (fid, partner_id, _) in col_list:
                    if partner_id == cause_obj_id:
                        for w in range(-COLLISION_WINDOW, COLLISION_WINDOW + 1):
                            leakage_key_set.add((oid, fid + w))

        f = self.num_sample_frames
        max_objs = self.max_objs
        cause_obj_mask = torch.zeros(f, max_objs, dtype=torch.float32)
        obj_bboxes = torch.zeros(f, max_objs, 4, dtype=torch.float32)
        obj_labels = torch.zeros(f, max_objs, dtype=torch.long)
        obj_spatial = torch.zeros(f, max_objs, self.num_spatial, dtype=torch.float32)
        obj_contact = torch.zeros(f, max_objs, self.num_contact, dtype=torch.float32)
        frame_res = torch.zeros(f, 2, dtype=torch.float32)
        frame_res[:, 0] = 320.0
        frame_res[:, 1] = 480.0
        num_objs = torch.zeros(f, dtype=torch.long)

        for fi, real_fi in enumerate(frame_idxs):
            real_fi = min(int(real_fi), traj_len - 1)

            # Get visible objects (bbox + attributes)
            if det is not None:
                objs = objects_at_frame_from_detector(det, real_fi, max_objs)
            else:
                objs = objects_visible_at_frame(ann, real_fi)
            n_vis = min(len(objs), max_objs)
            num_objs[fi] = n_vis

            for oi in range(n_vis):
                obj_id, bbox, col, shp, mat = objs[oi]
                obj_bboxes[fi, oi] = torch.tensor(bbox, dtype=torch.float32)
                obj_labels[fi, oi] = attr_tuple_to_label_index(col, shp, mat, self.num_label_embeddings)

                if cause_parsed is not None:
                    nc = col.lower().strip()
                    if nc == "gold": nc = "yellow"
                    if nc == "silver": nc = "gray"
                    ns = shp.lower().strip()
                    if ns == "ball": ns = "sphere"
                    if nc == cause_parsed[0] and ns == cause_parsed[1]:
                        cause_obj_mask[fi, oi] = 1.0

                # === Spatial rels: velocity + location (kept) ===
                vel, loc, ang_vel, inside = _get_motion_at_frame(ann, real_fi, obj_id)
                obj_spatial[fi, oi, 0] = vel[0] / VEL_SCALE
                obj_spatial[fi, oi, 1] = vel[1] / VEL_SCALE
                obj_spatial[fi, oi, 2] = loc[0] / LOC_SCALE
                obj_spatial[fi, oi, 3] = loc[1] / LOC_SCALE
                obj_spatial[fi, oi, 4] = ang_vel[0] / VEL_SCALE
                obj_spatial[fi, oi, 5] = ang_vel[1] / VEL_SCALE
                # === Contact rels: collision info (OLD semantics from bak_before_chain, 71233 repro) ===
                if obj_id in obj_collisions:
                    nearest_dist = 999
                    nearest_coll = None
                    for coll_frame, partner_id, coll_loc in obj_collisions[obj_id]:
                        dist = abs(coll_frame - real_fi)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_coll = (coll_frame, partner_id, coll_loc)

                    if nearest_coll is not None and nearest_dist <= COLLISION_WINDOW:
                        coll_frame, partner_id, coll_loc = nearest_coll
                        obj_contact[fi, oi, 0] = 1.0

                        p_vel, p_loc, _, _ = _get_motion_at_frame(ann, coll_frame, partner_id)
                        obj_contact[fi, oi, 1] = p_loc[0] / LOC_SCALE
                        obj_contact[fi, oi, 2] = p_loc[1] / LOC_SCALE

                        obj_contact[fi, oi, 3] = coll_loc[0] / LOC_SCALE if len(coll_loc) > 0 else 0
                        obj_contact[fi, oi, 4] = coll_loc[1] / LOC_SCALE if len(coll_loc) > 1 else 0

                        obj_contact[fi, oi, 5] = (coll_frame - real_fi) / 128.0

                    obj_contact[fi, oi, 6] = nearest_dist / 128.0

                obj_contact[fi, oi, 18] = 1.0 if inside else 0.0

        return {
            "obj_bboxes": obj_bboxes,
            "obj_labels": obj_labels,
            "obj_spatial_rels": obj_spatial,
            "obj_contacting_rels": obj_contact,
            "cause_obj_mask": cause_obj_mask,
            "frame_resolutions": frame_res,
            "num_objs": num_objs,
            "num_nonpadded_frames": num_unpadded_frames,
            "label": torch.tensor(p["label"], dtype=torch.float32),
            "question_text": p["question"],
            "choice_text": p["choice"],
            "question_type": p.get("question_type", "unknown"),
            "question_id": int(p.get("question_id", index)),
            "choice_id": int(p.get("choice_id", 0)),
            "video_filename": video_filename,
            "video_id": str(p.get("video_id", "")),
        }


def _pair_collate(batch):
    out = {}
    stack_keys = [
        "obj_bboxes", "obj_labels",
        "obj_spatial_rels", "obj_contacting_rels",
        "cause_obj_mask", "frame_resolutions", "num_objs", "label",
    ]
    for k in stack_keys:
        out[k] = torch.stack([b[k] for b in batch], 0)
    out["num_nonpadded_frames"] = [int(b["num_nonpadded_frames"]) for b in batch]
    for k in ("question_text", "choice_text", "question_type",
              "question_id", "choice_id", "video_filename", "video_id"):
        out[k] = [b[k] for b in batch]
    return out


def _resolve_path(base_dir, rel):
    if not rel:
        return ""
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(base_dir, rel))


def get_pair_dataloader_v2(cfg, distributed=False, rank=0, world_size=1):
    here = os.path.dirname(os.path.abspath(__file__))
    train_json = _resolve_path(here, getattr(cfg.DATALOADER, "CLEVRER_TRAIN_JSON", "").strip())
    val_json = _resolve_path(here, getattr(cfg.DATALOADER, "CLEVRER_VAL_JSON", "").strip())

    ann_train = _resolve_path(here, getattr(cfg.DATALOADER, "CLEVRER_ANNOTATION_TRAIN_ROOT", ""))
    ann_val = _resolve_path(here, getattr(cfg.DATALOADER, "CLEVRER_ANNOTATION_VAL_ROOT", "") or ann_train)

    visual_source = getattr(cfg.DATALOADER, "CLEVRER_VISUAL_SOURCE", "detector_json").lower().strip()
    det_train = _resolve_path(here, getattr(cfg.DATALOADER, "CLEVRER_DETECTOR_JSON_TRAIN_ROOT", ""))
    det_val = _resolve_path(here, getattr(cfg.DATALOADER, "CLEVRER_DETECTOR_JSON_VAL_ROOT", "") or det_train)

    if visual_source == "detector_json":
        if not _detector_dir_has_json(det_train) or not _detector_dir_has_json(det_val):
            warnings.warn("Detector JSON missing; falling back to annotation.", UserWarning, stacklevel=2)
            visual_source = "annotation"

    num_emb = _num_obj_label_embeddings(cfg)

    train_ds = ClevrerPairDatasetV2(cfg, train_json, ann_train, num_emb,
                                     split_name="train", visual_source=visual_source, detector_root=det_train)
    val_ds = ClevrerPairDatasetV2(cfg, val_json, ann_val, num_emb,
                                   split_name="val", visual_source=visual_source, detector_root=det_val)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=200)
        train_loader = DataLoader(train_ds, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=False,
                                   sampler=train_sampler, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                   collate_fn=_pair_collate, pin_memory=True)
    else:
        g = torch.Generator()
        g.manual_seed(200)
        train_loader = DataLoader(train_ds, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                   num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=_pair_collate,
                                   pin_memory=True, generator=g)

    val_loader = DataLoader(val_ds, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, shuffle=False,
                             num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=_pair_collate, pin_memory=True)
    val_loaders = [val_loader]

    extra_val_jsons = getattr(cfg.DATALOADER, "CLEVRER_EXTRA_VAL_JSONS", "").strip()
    if extra_val_jsons:
        for ev_path in extra_val_jsons.split(","):
            ev_path = _resolve_path(here, ev_path.strip())
            if os.path.isfile(ev_path):
                ev_ds = ClevrerPairDatasetV2(cfg, ev_path, ann_val, num_emb,
                                              split_name="val", visual_source=visual_source, detector_root=det_val)
                ev_dl = DataLoader(ev_ds, batch_size=cfg.DATALOADER.EVAL_BATCH_SIZE, shuffle=False,
                                    num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=_pair_collate, pin_memory=True)
                val_loaders.append(ev_dl)

    return train_loader, val_loaders, train_sampler
