#!/usr/bin/env bash
set -euo pipefail

# Main controls: edit these defaults or set them in your terminal for one run.
LEARNING_RATE="${LEARNING_RATE:-0.000001}"
SPEED="${SPEED:-0.5}"
BASELINE_MODEL="${BASELINE_MODEL:-baseline.zip}"
STEPS="${STEPS:-30000}"

# The launcher skips occupied offsets automatically.
PORT_OFFSET="${PORT_OFFSET:-11}"
TRAIN_MINUTES="${TRAIN_MINUTES:-25}"
BUDGET_MINUTES="${BUDGET_MINUTES:-30}"
MATRIX_MAP="${MATRIX_MAP:-loop}"
ENTITY="${ENTITY:-map_0/vehicle_0}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-120}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "$PYTHON_BIN" -u scripts/launch_training.py \
    --learning-rate "$LEARNING_RATE" --speed "$SPEED" \
    --baseline-model "$BASELINE_MODEL" --steps "$STEPS" \
    --port-offset "$PORT_OFFSET" --train-minutes "$TRAIN_MINUTES" \
    --budget-minutes "$BUDGET_MINUTES" --map "$MATRIX_MAP" \
    --entity "$ENTITY" --startup-timeout "$STARTUP_TIMEOUT" "$@"
