#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="data/internal/conf_discrete_synth"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/live_run_seed0.txt"

echo "Starting synthetic discrete ConfMDP run..."
echo "Live log: $LOG_FILE"

uv run python -u scripts/run_conf_discrete_synthetic.py \
	--output-dir "$OUT_DIR" \
	--seed 0 \
	--n-iterations 100 \
	2>&1 | tee "$LOG_FILE"

echo "Done."
