#!/bin/bash
# Train Spamo_MV (3-view attention-pool fusion, LlamaSLT3ViewAttnPool) on 3-view CSV data.
# True 3-view per forward pass — eval same as train.
#
# Released config: 7B (videollama3_7b). Sibling 2B/3B configs are not shipped in this
# release; if you regenerate them locally, this script accepts LLM_SIZE=videollama3_2b
# or LLM_SIZE=qwen2_5_3b transparently.
#
# Required env vars:
#   VIDEOLLAMA3_7B_PATH  → path to VideoLLaMA3-7B weights
#   CLIP_FEAT_ROOT       → directory holding pre-extracted CLIP-ViT-L/14 features
#   MAE_FEAT_ROOT        → directory holding pre-extracted VideoMAE features
#   FRAME_ROOT           → directory of decoded frames (used for metadata only)
#
# Optional:
#   REPO_ROOT            → repo root (default: auto-detect from this script's location)
#   SPLITS_DIR           → split CSVs (default: $REPO_ROOT/splits/three_view/e1_sentence_progressive_lab)
#   PYTHON               → python interpreter (default: python)
#   LOG_ROOT             → logging root (default: $REPO_ROOT/logs/$USER)
#   BATCH_SIZE           → override the per-LLM-size default
#   CUDA_VISIBLE_DEVICES → GPU id (default: 0)
#
# Usage:
#   LLM_SIZE=videollama3_7b bash experiments/llm_size_3view_v2/scripts/train_3view_attn.sh
#   LLM_SIZE=videollama3_7b BATCH_SIZE=4 bash .../train_3view_attn.sh    # OOM override

set -euo pipefail
trap 'echo ">>> interrupted, stopping..."; kill 0' SIGINT SIGTERM

: "${LLM_SIZE:?LLM_SIZE must be set (videollama3_7b for the released config)}"

# Auto-detect REPO_ROOT = three levels up from this script (scripts/ → llm_size_3view_v2/ → experiments/ → REPO).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXPDIR="${REPO_ROOT}/experiments/llm_size_3view_v2"
PYTHON="${PYTHON:-python}"

CONFIG="${EXPDIR}/configs/finetune_3view_attn_${LLM_SIZE}.yaml"
RUN_NAME="e1_3view_attn_${LLM_SIZE}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/${USER:-user}}"
RUN_TAG="${SLURM_JOB_ID:-$(date +%Y%m%dT%H%M%S)}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}/${RUN_TAG}"

# Per-LLM-size default batch (override via env BATCH_SIZE)
case "${LLM_SIZE}" in
    videollama3_2b) DEFAULT_BS=16 ;;
    qwen2_5_3b)     DEFAULT_BS=12 ;;
    videollama3_7b) DEFAULT_BS=8  ;;
    *)              DEFAULT_BS=8  ;;
esac
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BS}}"

# CSV splits (shipped under splits/ in this release)
SPLITS_DIR="${SPLITS_DIR:-${REPO_ROOT}/splits/three_view/e1_sentence_progressive_lab}"
TRAIN_CSV="${SPLITS_DIR}/train.csv"
VAL_CSV="${SPLITS_DIR}/val.csv"
TEST_CSV="${SPLITS_DIR}/test.csv"

# Pre-extracted features (must be set by user; see README → "Feature extraction")
: "${CLIP_FEAT_ROOT:?set CLIP_FEAT_ROOT to your extracted CLIP features dir}"
: "${MAE_FEAT_ROOT:?set MAE_FEAT_ROOT to your extracted MAE features dir}"
FRAME_ROOT="${FRAME_ROOT:-${REPO_ROOT}/frames}"

mkdir -p "${LOG_ROOT}"
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Train CSV not found: ${TRAIN_CSV}" >&2; exit 1; }

echo "======================================================"
echo "  LLM_SIZE     : ${LLM_SIZE}"
echo "  Config       : ${CONFIG}"
echo "  Run name     : ${RUN_NAME}"
echo "  Train rows   : $(wc -l < "${TRAIN_CSV}")"
echo "  Val rows     : $(wc -l < "${VAL_CSV}")"
echo "  Test rows    : $(wc -l < "${TEST_CSV}")"
echo "  Batch size   : ${BATCH_SIZE}"
echo "  CLIP feats   : ${CLIP_FEAT_ROOT}"
echo "  MAE feats    : ${MAE_FEAT_ROOT}"
echo "  Frame root   : ${FRAME_ROOT}"
echo "  Log dir      : ${LOG_DIR}"
echo "======================================================"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${EXPDIR}/scripts:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" "${EXPDIR}/scripts/main_3view_csv.py" \
    -c "${CONFIG}" \
    --train_csv "${TRAIN_CSV}" \
    --val_csv   "${VAL_CSV}" \
    --test_csv  "${TEST_CSV}" \
    --clip_feat_root "${CLIP_FEAT_ROOT}" \
    --mae_feat_root  "${MAE_FEAT_ROOT}" \
    --frame_root     "${FRAME_ROOT}" \
    --label_column   Sign_sentence \
    --batch_size     "${BATCH_SIZE}" \
    -e bleu \
    -n "${RUN_NAME}" \
    --logdir "${LOG_DIR}"

echo ">>> Train done."
