# Enhancing Video Causal Question Answering with Causal Object Counterfactual Learning

**SOTA 78.70% option accuracy on CLEVRER-Humans** | Kyung Hee University

---

## Overview

This repository contains the code for our EGNN-augmented IPRM model for causal video question answering on the [CLEVRER-Humans](https://clevrer-humans.github.io/) benchmark.

We extend the IPRM (Iterative and Parallel Reasoning Mechanism) with a frozen **E(n)-Equivariant Graph Neural Network (EGNN)** that provides physics-aware object dynamics features, combined with a **masked-path counterfactual** strategy to improve causal reasoning.

### Key Results

| Method | Option Acc@0.5 | Question Acc@0.5 |
|--------|:--------------:|:----------------:|
| IPRM (baseline) | ~74% | ~62% |
| **Ours (EGNN + masked-path)** | **78.70%** | **62.50%** |

---

## Architecture

```
Video frames
    │
    ▼
Object detector (GT boxes)
    │
    ├──► BiLSTM + IPRM reasoning (trainable)
    │         │
    │    EGNN Dynamics Encoder (frozen)
    │         │  dyn_embd  [B, F, N, 128]  ──► Linear proj ──┐
    │         │  coll_logit [B, F, N, N]   ──► Linear proj ──┤
    │         └─────────────────────────────────────────────►─┤
    │                                                         │
    └─────────────────────────────────────────────────────────┘
                              │
                    IPRM iterative reasoning
                    (11 computation steps)
                              │
                    Per-option binary classifier
```

**Masked-path counterfactual:** for causal questions, the model performs two forward passes — one with full object features, one with the cause-object coordinates zeroed — enabling explicit counterfactual reasoning.

### EGNN Pretrain

The EGNN is pretrained on CLEVRER dynamics prediction (position forecasting + collision detection) before being frozen for the QA stage:
- Position RMSE: 5.3px
- Collision AUC: **0.988**

### Fine-tuning Hyperparameters (SOTA config)

| Hyperparameter | Value |
|---|---|
| LR (BiLSTM + IPRM) | 1e-5 |
| IPRM computation steps | 11 |
| Focal loss γ | 2.0 |
| Focal loss α | 0.65 |
| L_sim weight | 0.01 |
| Batch size (eff.) | 64 (4 GPU × 4 × accum 4) |
| Training epochs | 150 (early stop patience 15) |

---

## Repository Structure

```
.
├── vlm_model.py            # Main model: IPRM + EGNN integration
├── egnn_runtime.py         # Frozen EGNN runtime (FrozenEGNN)
├── iprm_v2_module.py       # IPRM v2 reasoning module
├── iprm_module.py          # IPRM base module
├── basic_model_blocks.py   # Shared building blocks
├── train_pairs.py          # Training script (pair-format BCE)
├── dataset_pairs_v2.py     # Dataset loader
├── eval_utils.py           # Evaluation utilities
├── eval_best_model.py      # Best-checkpoint evaluator
├── clevrer_annotation.py   # CLEVRER annotation parser
├── clevrer_detector_json.py
├── clevrer_label_vocab.py
├── data_utils.py
├── config/                 # Config defaults
├── configs/
│   ├── clevrer_paper.yaml          # Base config
│   └── clevrer_humans_finetune.yaml
├── tools/
│   ├── egnn_pretrain.py    # EGNNDynamics model + pretrain script
│   └── egnn_eval_standalone.py
└── run_sota.sbatch         # SLURM script to reproduce SOTA
```

---

## Setup

```bash
conda create -n iprm python=3.10
conda activate iprm
pip install -r requirements.txt
```

---

## Data

Download CLEVRER-Humans data from the official benchmark:

- **CLEVRER-Humans**: [https://clevrer-humans.github.io/](https://clevrer-humans.github.io/)
- **CLEVRER** (videos + annotations): [http://clevrer.csail.mit.edu/](http://clevrer.csail.mit.edu/)

After downloading, prepare the pair-format QA files:

```
data/
├── train_pairs_humans_only.json   # Humans-only balanced train split
├── val_pairs_balanced.json        # Balanced val split
└── clevrer/
    ├── annotation_train/
    ├── annotation_validation/
    └── detector_json_export_gt/
        ├── train/
        └── val/
```

**Note:** We train exclusively on the humans-only balanced split (NOT CEG-merged) to match the CLEVRER-Humans evaluation protocol.

---

## Training

### Step 1 — Pretrain EGNN dynamics encoder

```bash
python tools/egnn_pretrain.py \
  --data-root /path/to/clevrer \
  --output outputs/egnn_pretrain_v1
```

### Step 2 — Pretrain BiLSTM + IPRM on CLEVRER (GT boxes)

```bash
torchrun --standalone --nproc_per_node=4 train_pairs.py \
  --config-file configs/clevrer_paper.yaml \
  --masked-path \
  VLM.USE_EGNN_DYNAMICS True \
  VLM.EGNN_CKPT_PATH outputs/egnn_pretrain_v1/best.pt \
  DATALOADER.CLEVRER_TRAIN_JSON /path/to/clevrer_train_pairs.json \
  SAVE_DIRECTORY outputs/pretrain_egnn_v1
```

### Step 3 — Fine-tune on CLEVRER-Humans (SOTA config)

Edit paths in `run_sota.sbatch` then:

```bash
sbatch run_sota.sbatch
# or directly:
torchrun --standalone --nproc_per_node=4 train_pairs.py \
  --config-file configs/clevrer_paper.yaml \
  --masked-path --accum-steps 4 \
  --early-stop-patience 15 \
  --l-sim-weight 0.01 --focal-gamma 2.0 --focal-alpha 0.65 \
  VLM.USE_EGNN_DYNAMICS True \
  VLM.EGNN_CKPT_PATH outputs/egnn_pretrain_v1/best.pt \
  VLM.EGNN_DYN_DIM 128 VLM.EGNN_LAYERS 4 \
  IPRM.NUM_COMPUTATION_STEPS 11 \
  SOLVER.LR 1.0e-5 SOLVER.LANG_ENC_LR 1.0e-5 \
  DATALOADER.BATCH_SIZE 4 \
  DATALOADER.CLEVRER_TRAIN_JSON /path/to/train_pairs_humans_only.json \
  DATALOADER.CLEVRER_VAL_JSON /path/to/val_pairs_balanced.json \
  LOAD True LOAD_PATH outputs/pretrain_egnn_v1/best_model.model \
  SAVE_DIRECTORY outputs/sota_repro
```

---

## Evaluation

```bash
python eval_best_model.py \
  --ckpt outputs/sota_repro/best_model.model \
  --config-file configs/clevrer_paper.yaml \
  --val-json /path/to/val_pairs_balanced.json
```

Expected: **option_acc@0.5 ≈ 0.787, question_acc@0.5 ≈ 0.625**

---

## Citation

If you use this code, please cite the CLEVRER-Humans benchmark and the IPRM paper:

```bibtex
@inproceedings{clevrer_humans,
  title={CLEVRER-Humans: Describing Physical and Causal Events the Human Way},
  booktitle={NeurIPS},
  year={2022}
}

@article{iprm,
  title={Iterative and Parallel Reasoning Mechanism for Visual Question Answering},
  year={2022}
}
```
