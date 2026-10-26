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
LOG_DIR="data/internal/2pmg_continuous/testcritic7epsode7lr0.05weights0.50.5beta0.01sigmaleader1e-3_${BENEFIT_BASELINE_MODE}"
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

echo "Starting 2PMG Continuous experiments..."
echo "Working directory: $(pwd)"
echo "Log directory:     $LOG_DIR"
echo "Benefit baseline:  $BENEFIT_BASELINE_MODE"
echo "Debug trace:       $DEBUG_TRACE (every=$DEBUG_TRACE_EVERY, jsonl=$DEBUG_TRACE_JSONL)"
echo "Full batch cum.:   $FULL_BATCH_CUMULATIVE"
echo ""

nohup uv run python -u scripts/run_2pmg_continuous.py \
    --mode softvi --output-dir "$LOG_DIR" "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/logs_softvi.txt" 2>&1 &
PID1=$!
echo "Started softvi            [PID: $PID1]"

nohup uv run python -u scripts/run_2pmg_continuous.py \
    --mode irl_known_dynamics --output-dir "$LOG_DIR" "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/logs_irl_known_dynamics.txt" 2>&1 &
PID2=$!
echo "Started irl_known_dynamics [PID: $PID2]"

nohup uv run python -u scripts/run_2pmg_continuous.py \
    --mode non_irl --output-dir "$LOG_DIR" "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/logs_non_irl.txt" 2>&1 &
PID3=$!
echo "Started non_irl            [PID: $PID3]"

nohup uv run python -u scripts/run_2pmg_continuous.py \
    --mode irl_recursive_rls --output-dir "$LOG_DIR" "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/logs_irl_recursive_rls.txt" 2>&1 &
PID4=$!
echo "Started irl_recursive_rls    [PID: $PID4]"

nohup uv run python -u scripts/run_2pmg_continuous.py \
    --mode irl_unknown_dynamics --output-dir "$LOG_DIR" "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/logs_irl_unknown_dynamics.txt" 2>&1 &
PID5=$!
echo "Started irl_unknown_dynamics [PID: $PID5]"

nohup uv run python -u scripts/run_2pmg_continuous.py \
    --mode irl_unknown_recursive_rls --output-dir "$LOG_DIR" "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/logs_irl_unknown_recursive_rls.txt" 2>&1 &
PID6=$!
echo "Started irl_unknown_recursive_rls [PID: $PID6]"

echo "$PID1" > "$LOG_DIR/.pid_softvi"
echo "$PID2" > "$LOG_DIR/.pid_irl_known_dynamics"
echo "$PID3" > "$LOG_DIR/.pid_non_irl"
echo "$PID4" > "$LOG_DIR/.pid_irl_recursive_rls"
echo "$PID5" > "$LOG_DIR/.pid_irl_unknown_dynamics"
echo "$PID6" > "$LOG_DIR/.pid_irl_unknown_recursive_rls"

echo ""
echo "All 6 experiments started in background."
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/logs_softvi.txt"
echo "  tail -f $LOG_DIR/logs_irl_known_dynamics.txt"
echo "  tail -f $LOG_DIR/logs_irl_unknown_dynamics.txt"
echo "  tail -f $LOG_DIR/logs_non_irl.txt"
echo "  tail -f $LOG_DIR/logs_irl_recursive_rls.txt"
echo "  tail -f $LOG_DIR/logs_irl_unknown_recursive_rls.txt"
echo "Check status:"
echo "  ps -p $PID1 $PID2 $PID3 $PID4 $PID5 $PID6"
echo ""

trap 'echo "Detached. Processes continue in background."; exit 0' INT TERM
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6

echo "All experiments finished! Generating plots..."
uv run python scripts/run_2pmg_continuous.py --mode plot --output-dir "$LOG_DIR"
echo "Done! Check $LOG_DIR/comparison_curves.pdf"
