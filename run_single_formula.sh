#!/usr/bin/env bash
set -euo pipefail

# Run enumerate_rxn_worker.py for one line of smiles_by_formula.txt
# Usage: ./run_single_formula.sh <line_number>

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <line_number>" >&2
    exit 1
fi

LINE_NUM="$1"
if ! [[ "$LINE_NUM" =~ ^[0-9]+$ ]]; then
    echo "Error: line_number must be a positive integer." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMILES_FILE="${SCRIPT_DIR}/smiles_by_formula_cleaned.txt"

if [[ ! -f "$SMILES_FILE" ]]; then
    echo "Error: SMILES-by-formula file not found at $SMILES_FILE" >&2
    exit 1
fi

LINE_CONTENT="$(sed -n "${LINE_NUM}p" "$SMILES_FILE")"
if [[ -z "${LINE_CONTENT// }" ]]; then
    echo "Error: no content on line ${LINE_NUM} in $SMILES_FILE" >&2
    exit 1
fi

# First token is the label/formula, remaining tokens are SMILES
read -r -a TOKENS <<<"$LINE_CONTENT"
if (( ${#TOKENS[@]} < 2 )); then
    echo "Error: line ${LINE_NUM} has no SMILES entries (only label: ${TOKENS[*]:-})" >&2
    exit 1
fi

LABEL="${TOKENS[0]}"
SMILES_LIST=("${TOKENS[@]:1}")

OUT_DIR="${SCRIPT_DIR}/t1x-yarp-0.5/line${LINE_NUM}"
OUTPUT_PREFIX="${OUT_DIR}/formula_line${LINE_NUM}"
OUTPUT_PKL="${OUTPUT_PREFIX}.pkl"
LOG_FILE="${OUTPUT_PREFIX}.log"
MAX_WORKERS=54
SCORE_THRESHOLD=0.15

# 如果已有日志且最后一行包含 Done，则认为任务完成，直接退出
if [[ -f "$LOG_FILE" ]]; then
    LAST_LINE="$(tail -n 1 "$LOG_FILE")"
    if [[ "$LAST_LINE" == *"Done"* ]]; then
        echo "Log indicates completion (last line has 'Done'): ${LOG_FILE}"
        echo "Skipping calculation for line ${LINE_NUM} (label=${LABEL})."
        exit 0
    fi
    echo "Existing log found but not completed; rerunning line ${LINE_NUM} (label=${LABEL})."
fi

# 准备输出目录
mkdir -p "${OUT_DIR}"

echo "Running bnfm_iterator_batch on line ${LINE_NUM} (label=${LABEL}) with ${#SMILES_LIST[@]} SMILES | max-workers=${MAX_WORKERS}"

python "${SCRIPT_DIR}/bnfm_iterator_batch.py" \
    --smiles "${SMILES_LIST[@]}" \
    --output "${OUTPUT_PKL}" \
    --log-file "${LOG_FILE}" \
    --max-workers ${MAX_WORKERS} \
    --score-threshold ${SCORE_THRESHOLD}

echo "Done. Results: ${OUTPUT_PKL}"
echo "Log: ${LOG_FILE}"
