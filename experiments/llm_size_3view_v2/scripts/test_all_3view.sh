#!/bin/bash
# Test the released 3-view 7B ckpt (LlamaSLT3ViewAttnPool) on val / test / test_unseen
# / test_real_group1. Single combined log + per-run logs.
#
# Required env vars:
#   CKPT_PATH       → trained 7B checkpoint (epoch=...-step=...-bleu4=...ckpt)
#   CLIP_FEAT_ROOT  → directory holding pre-extracted CLIP-ViT-L/14 features
#   MAE_FEAT_ROOT   → directory holding pre-extracted VideoMAE features
#   FRAME_ROOT      → directory of decoded frames (used for metadata only)
#
# Optional:
#   REPO_ROOT       → repo root (default: auto-detect from this script's location)
#   SPLITS_DIR      → split CSVs (default: $REPO_ROOT/splits/three_view/e1_sentence_progressive_lab)
#   PYTHON          → python interpreter (default: python)
#   SPLITS          → space-separated list of splits to run
#                     (default: "val test test_unseen test_real_group1")
#   CUDA_VISIBLE_DEVICES → GPU id (default: 0)
#
# Usage:
#   CKPT_PATH=/path/to/best.ckpt \
#   CLIP_FEAT_ROOT=... MAE_FEAT_ROOT=... FRAME_ROOT=... \
#   bash experiments/llm_size_3view_v2/scripts/test_all_3view.sh
#
# Run a subset of splits:
#   SPLITS="val test_unseen" bash .../test_all_3view.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP="${REPO_ROOT}/experiments/llm_size_3view_v2"
PY="${PYTHON:-python}"

SPLITS_DIR="${SPLITS_DIR:-${REPO_ROOT}/splits/three_view/e1_sentence_progressive_lab}"
: "${CKPT_PATH:?set CKPT_PATH to the 7B checkpoint to evaluate}"
: "${CLIP_FEAT_ROOT:?set CLIP_FEAT_ROOT to your extracted CLIP features dir}"
: "${MAE_FEAT_ROOT:?set MAE_FEAT_ROOT to your extracted MAE features dir}"
FRAME_ROOT="${FRAME_ROOT:-${REPO_ROOT}/frames}"

CFG_7B="${EXP}/configs/finetune_3view_attn_videollama3_7b.yaml"

# Default test splits (override via SPLITS="...")
read -r -a SPLITS <<<"${SPLITS:-val test test_unseen test_real_group1}"

RESULTS_DIR="${EXP}/results"
mkdir -p "${RESULTS_DIR}"

TS=$(date +%Y%m%d_%H%M%S)
COMBINED_LOG="${RESULTS_DIR}/test_all_3view_${TS}.log"
SUMMARY_LOG="${RESULTS_DIR}/test_all_3view_${TS}.summary"

# Sanity checks
[[ -f "$CKPT_PATH" ]] || { echo "[FATAL] missing CKPT_PATH: $CKPT_PATH" | tee -a "$COMBINED_LOG"; exit 1; }
[[ -f "$CFG_7B" ]] || { echo "[FATAL] missing config: $CFG_7B" | tee -a "$COMBINED_LOG"; exit 1; }
for s in "${SPLITS[@]}"; do
    [[ -f "${SPLITS_DIR}/${s}.csv" ]] || { echo "[FATAL] missing: ${SPLITS_DIR}/${s}.csv" | tee -a "$COMBINED_LOG"; exit 1; }
done

run_test() {
    local TAG=$1; local CFG=$2; local CKPT=$3; local SPLIT=$4; local BS=$5
    local PER_LOG=${RESULTS_DIR}/${TAG}__${SPLIT}__${TS}.log

    {
        echo
        echo "============================================================"
        echo "  MODEL : ${TAG}"
        echo "  SPLIT : ${SPLIT}"
        echo "  CKPT  : ${CKPT}"
        echo "  CFG   : ${CFG}"
        echo "  BS    : ${BS}"
        echo "  TIME  : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
    } | tee -a "$COMBINED_LOG"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" "$EXP/scripts/test_3view_csv.py" \
        --config         "$CFG" \
        --ckpt           "$CKPT" \
        --csv            "${SPLITS_DIR}/${SPLIT}.csv" \
        --clip_feat_root "$CLIP_FEAT_ROOT" \
        --mae_feat_root  "$MAE_FEAT_ROOT" \
        --frame_root     "$FRAME_ROOT" \
        --label_column   Sign_sentence \
        --batch_size     "$BS" \
        --split_name     "${TAG}__${SPLIT}" \
        2>&1 | tee "$PER_LOG" | tee -a "$COMBINED_LOG"
    local rc=${PIPESTATUS[0]}

    if [[ $rc -ne 0 ]]; then
        echo "[ERROR] ${TAG} ${SPLIT} exited with code $rc" | tee -a "$COMBINED_LOG"
    fi

    # Extract BLEU/ROUGE lines for summary
    {
        echo "[${TAG} | ${SPLIT}]"
        grep -E "test/(bleu[1-4]|rouge|wer|loss)" "$PER_LOG" | sed 's/^/    /' || echo "    (no metrics matched)"
    } >> "$SUMMARY_LOG"
}

echo ">>> Combined log : $COMBINED_LOG"
echo ">>> Summary log  : $SUMMARY_LOG"
echo ">>> Per-run logs : ${RESULTS_DIR}/<tag>__<split>__${TS}.log"
echo

BS="${BATCH_SIZE:-8}"
for SPLIT in "${SPLITS[@]}"; do
    run_test videollama3_7b "$CFG_7B" "$CKPT_PATH" "$SPLIT" "$BS"
done

{
    echo
    echo "============================================================"
    echo "  ALL DONE : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo
    echo ">>> Final summary:"
    cat "$SUMMARY_LOG"
} | tee -a "$COMBINED_LOG"

echo
echo ">>> View summary : cat $SUMMARY_LOG"
echo ">>> View full    : less $COMBINED_LOG"
