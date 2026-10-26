#!/bin/bash
# Run all 2PMG Continuous experiments in parallel.
# Usage: bash sh/run_2pmg_continuous.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BENEFIT_BASELINE_MODE="${BENEFIT_BASELINE_MODE:-online-online}"
DEBUG_TRACE="${DEBUG_TRACE:-0}"
DEBUG_TRACE_EVERY="${DEBUG_TRACE_EVERY:-10}"
DEBUG_TRACE_JSONL="${DEBUG_TRACE_JSONL:-1}"
FULL_BATCH_CUMULATIVE="${FULL_BATCH_CUMULATIVE:-0}"
LOG_DIR="data/internal-test/2pmg_continuous/test7-1gate1e-4critic10epsode4lr0.05weights0.50.5beta0.01sigmaleader1e-3_${BENEFIT_BASELINE_MODE}"
N_SEEDS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            LOG_DIR="$2"; shift 2 ;;
        --n-seeds)
            N_SEEDS="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash sh/run_2pmg_continuous.sh [--out <OUTPUT_DIR>] [--n-seeds <N>]"
            exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

EXTRA_ARGS=(--benefit-baseline-mode "$BENEFIT_BASELINE_MODE")
if [ "$DEBUG_TRACE" = "1" ]; then
    EXTRA_ARGS+=(--debug-trace --debug-trace-every "$DEBUG_TRACE_EVERY")
fi
if [ "$DEBUG_TRACE_JSONL" = "1" ]; then
    EXTRA_ARGS+=(--debug-trace-jsonl)
fi
if [ "$FULL_BATCH_CUMULATIVE" = "1" ]; then
    EXTRA_ARGS+=(--full-batch-cumulative)
fi

SEEDS=$(seq 0 $((N_SEEDS - 1)))

echo "Starting 2PMG Continuous experiments..."
echo "  Output dir : $LOG_DIR"
echo "  N seeds    : $N_SEEDS  (seeds 0 .. $((N_SEEDS - 1)))"
echo "  Benefit baseline:  $BENEFIT_BASELINE_MODE"
echo "  Working dir: $(pwd)"
echo ""

PIDS=()

for seed in $SEEDS; do
    for mode in softvi irl_known_dynamics irl_unknown_dynamics non_irl irl_recursive_rls irl_unknown_recursive_rls; do
        log_file="$LOG_DIR/logs_${mode}_seed${seed}.txt"
        nohup uv run python -u scripts/run_2pmg_continuous.py \
            --mode "$mode" \
            --output-dir "$LOG_DIR" \
            --seed "$seed" \
            "${EXTRA_ARGS[@]}" \
            > "$log_file" 2>&1 &
        PID=$!
        PIDS+=($PID)
        echo "  Started mode=$mode seed=$seed  [PID: $PID]  -> $log_file"
    done
done

echo ""
echo "All $((N_SEEDS * 6)) processes started in background."
echo "Monitor progress (example):"
echo "  tail -f $LOG_DIR/logs_softvi_seed0.txt"
echo ""

trap 'echo "Detached. Processes continue in background."; exit 0' INT TERM
wait "${PIDS[@]}"

echo "All experiments finished! Aggregating results..."
uv run python scripts/run_2pmg_continuous.py --mode aggregate --output-dir "$LOG_DIR" --n-seeds "$N_SEEDS"

echo "Generating plots..."
uv run python scripts/run_2pmg_continuous.py --mode plot --output-dir "$LOG_DIR"
echo "Done! Check $LOG_DIR/comparison_curves.pdf"
