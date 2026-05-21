#!/usr/bin/bash
#SBATCH -J sota-egnn
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH -p batch_ugrad
#SBATCH -t 12:00:00
#SBATCH -o slurm/slurm-%A.out

# ==============================================================================
# SOTA reproduction: EGNN-frozen + thin Linear adapter + v7 trial 1 params
# Best val option_acc = 0.7870 (th=0.5), q_acc@0.5 = 0.625
#
# Architecture (vlm_model.py current state = SOTA):
#   - obj_spatial_rel_proj : single Linear(egnn_dyn_dim=128 -> OBJ_REL_LATENT_DIMS=128) + act + dropout
#   - obj_contact_rel_proj : single Linear(egnn_coll_dim=16 -> OBJ_REL_LATENT_DIMS=128) + act + dropout
#   - EGNN frozen (loaded from EGNN_CKPT_PATH), BiLSTM+IPRM trainable from pretrain ckpt
#
# Ancestor ckpts required:
#   - videoQA/outputs/egnn_pretrain_v1/best.pt                          (EGNN pretrain, 92% coll AUC 0.988)
#   - videoQA/outputs/bilstm_pretrain_expl_egnn_v1/best_model.model     (BiLSTM+IPRM+EGNN pretrain, 92.98%)
#
# Expected runtime: ~80 min (150 epoch cap, ES patience=15, usual ES ~E45-60)
# ==============================================================================
set -euo pipefail

source /data/kimin10866/anaconda3/etc/profile.d/conda.sh && conda activate iprm

REPO=/data/kimin10866/repos/IPRM_Iterative_and_Parallel_Reasoning_Mechanism-main
CLEVRER=/data/kimin10866/repos/CLEVRER-Humans1.0-main/data/clevrer
DATA=/data/kimin10866/repos/CLEVRER-Humans1.0-main/models/clevrer_monet_latents

PRETRAIN_CKPT="${REPO}/videoQA/outputs/bilstm_pretrain_expl_egnn_v1/best_model.model"
EGNN_CKPT="${REPO}/videoQA/outputs/egnn_pretrain_v1/best.pt"
TRAIN_JSON="${DATA}/train_pairs_humans_only.json"
VAL_JSON="${DATA}/val_pairs_balanced.json"

OUT="${REPO}/videoQA/outputs/sota_egnn_repro"
mkdir -p "${OUT}"

cd "${REPO}/videoQA"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT_MS=7200000
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

pwd; which python; hostname
nvidia-smi -L || true

# SOTA hyperparams (v7 trial 1: best_opt=0.7870 @ th=0.5)
#   lr=1e-5, l_sim=0.01, focal_g=2.0, focal_a=0.65, n_steps=11
torchrun --standalone --nproc_per_node=4 \
  train_pairs.py \
  --config-file configs/clevrer_paper.yaml \
  --no-wandb \
  --masked-path \
  --accum-steps 4 \
  --early-stop-patience 15 \
  --l-sim-weight 0.01 \
  --focal-gamma 2.0 \
  --focal-alpha 0.65 \
  --label-smoothing 0.0 \
  DATALOADER.BATCH_SIZE 4 \
  DATALOADER.EVAL_BATCH_SIZE 4 \
  DATALOADER.NUM_WORKERS 4 \
  DATALOADER.CLEVRER_TRAIN_JSON "${TRAIN_JSON}" \
  DATALOADER.CLEVRER_VAL_JSON "${VAL_JSON}" \
  DATALOADER.CLEVRER_DETECTOR_JSON_TRAIN_ROOT "${CLEVRER}/detector_json_export_gt/train" \
  DATALOADER.CLEVRER_DETECTOR_JSON_VAL_ROOT "${CLEVRER}/detector_json_export_gt/val" \
  DATALOADER.CLEVRER_ANNOTATION_TRAIN_ROOT "${CLEVRER}/annotation_train" \
  DATALOADER.CLEVRER_ANNOTATION_VAL_ROOT "${CLEVRER}/annotation_validation" \
  SOLVER.CRITERION binary_cross_entropy \
  SOLVER.LR 1.0e-5 \
  SOLVER.LANG_ENC_LR 1.0e-5 \
  SOLVER.EPOCHS 150 \
  SOLVER.CHECKPOINT_PERIOD 0 \
  SOLVER.PATIENCE 5 \
  VLM.USE_PRETRAINED_LANG_ENCODER False \
  VLM.LSTM_APPLY_AUX_PROJS False \
  VLM.INDEPENDENT_OPTION_CLASSIFIER_TYPE individual \
  VLM.USE_EGNN_DYNAMICS True \
  VLM.EGNN_CKPT_PATH "${EGNN_CKPT}" \
  VLM.EGNN_DYN_DIM 128 \
  VLM.EGNN_LAYERS 4 \
  IPRM.NUM_COMPUTATION_STEPS 11 \
  IPRM.REDUCTION_R 2 \
  IPRM.MEMORY_ATT_WINDOW_LEN 2 \
  LOAD True \
  LOAD_PATH "${PRETRAIN_CKPT}" \
  SAVE_DIRECTORY "${OUT}"

echo ""
echo "=== SOTA reproduction done (target: 0.7870 best_option_acc) ==="
