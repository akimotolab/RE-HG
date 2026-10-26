#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="data/internal-test/conf_continuous/next1/test20secondthirdbaseQ30"
N_SEEDS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            LOG_DIR="$2"; shift 2 ;;
        --n-seeds)
            N_SEEDS="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash sh/run_conf_continuous.sh --out <OUTPUT_DIR> [--n-seeds <N>]"
            exit 1 ;;
    esac
done

if [[ -z "$LOG_DIR" ]]; then
    echo "Error: --out <OUTPUT_DIR> is required."
    echo "Usage: bash sh/run_conf_continuous.sh --out <OUTPUT_DIR> [--n-seeds <N>]"
    exit 1
fi

SEEDS=$(seq 0 $((N_SEEDS - 1)))
mkdir -p "$LOG_DIR"

echo "Starting Conf Continuous (BuildingThermalControl) experiments"
echo "  Output dir : $LOG_DIR"
echo "  N seeds    : $N_SEEDS  (seeds 0 .. $((N_SEEDS - 1)))"
echo "  Working dir: $(pwd)"
echo ""

PIDS=()

for seed in $SEEDS; do
    for mode in softvi irl_known_dynamics irl_recursive_rls non_irl; do
        log_file="$LOG_DIR/logs_${mode}_seed${seed}.txt"
        nohup uv run python -u scripts/run_conf_continuous.py \
            --mode "$mode" \
            --output-dir "$LOG_DIR" \
            --seed "$seed" \
            > "$log_file" 2>&1 &
        PID=$!
        PIDS+=($PID)
        echo "  Started mode=$mode seed=$seed  [PID: $PID]  -> $log_file"
    done
done

echo ""
echo "All $((N_SEEDS * 4)) processes started in background."
echo "Monitor progress (example):"
echo "  tail -f $LOG_DIR/logs_softvi_seed0.txt"
echo ""

trap 'echo "Detached. Processes continue in background."; exit 0' INT TERM
wait "${PIDS[@]}"

echo ""
echo "All experiments finished! Aggregating results..."
uv run python scripts/run_conf_continuous.py \
    --mode aggregate \
    --output-dir "$LOG_DIR" \
    --n-seeds "$N_SEEDS"

echo ""
echo "Generating plots..."
uv run python scripts/run_conf_continuous.py \
    --mode plot \
    --output-dir "$LOG_DIR" \
    --n-seeds "$N_SEEDS"

echo "Done! Check $LOG_DIR/"
