#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting Parallel Experiments..."
echo "Working directory: $(pwd)"

LOG_DIR="data/internal-test/discrete_toyenv/test30-1sq50gamma0.9beta0.25randomstart"
mkdir -p "$LOG_DIR"

SEEDS="1,7,8"
MAX_SEED_WORKERS=3

nohup uv run python -u scripts/runrun.py --mode irl --output-dir "$LOG_DIR" --seeds "$SEEDS" --seed-parallel --max-seed-workers "$MAX_SEED_WORKERS" > "$LOG_DIR/logs_irl.txt" 2>&1 &
PID1=$!
echo "Started Proposed (IRL) [PID: $PID1]"

nohup uv run python -u scripts/runrun.py --mode expert_surrogate_no_second_term --output-dir "$LOG_DIR" --seeds "$SEEDS" --seed-parallel --max-seed-workers "$MAX_SEED_WORKERS" > "$LOG_DIR/logs_expert_surrogate_no_second_term.txt" 2>&1 &
PID2=$!
echo "Started Baseline (Expert surrogate, no 2nd term) [PID: $PID2]"

nohup uv run python -u scripts/runrun.py --mode irl_known_dynamics --output-dir "$LOG_DIR" --seeds "$SEEDS" --seed-parallel --max-seed-workers "$MAX_SEED_WORKERS" > "$LOG_DIR/logs_irl_known_dynamics.txt" 2>&1 &
PID3=$!
echo "Started Proposed (IRL, known dynamics) [PID: $PID3]"

nohup uv run python -u scripts/runrun.py --mode expert_surrogate_known_dynamics_no_second_term --output-dir "$LOG_DIR" --seeds "$SEEDS" --seed-parallel --max-seed-workers "$MAX_SEED_WORKERS" > "$LOG_DIR/logs_expert_surrogate_known_dynamics_no_second_term.txt" 2>&1 &
PID6=$!
echo "Started Baseline (Expert surrogate, known dynamics, no 2nd term) [PID: $PID6]"

nohup uv run python -u scripts/runrun.py --mode softql --output-dir "$LOG_DIR" --seeds "$SEEDS" --seed-parallel --max-seed-workers "$MAX_SEED_WORKERS" > "$LOG_DIR/logs_softql.txt" 2>&1 &
PID4=$!
echo "Started Proposed (SoftQL) [PID: $PID4]"

nohup uv run python -u scripts/runrun.py --mode softvi_known_dynamics --output-dir "$LOG_DIR" --seeds "$SEEDS" --seed-parallel --max-seed-workers "$MAX_SEED_WORKERS" > "$LOG_DIR/logs_softvi_known_dynamics.txt" 2>&1 &
PID7=$!
echo "Started Oracle (SoftVI, known dynamics) [PID: $PID7]"

nohup uv run python -u scripts/runrun.py --mode irl_known_dynamics_every_iter --output-dir "$LOG_DIR" --seeds "$SEEDS" --seed-parallel --max-seed-workers "$MAX_SEED_WORKERS" > "$LOG_DIR/logs_irl_known_dynamics_every_iter.txt" 2>&1 &
PID8=$!
echo "Started Proposed (IRL, known dynamics, every iter) [PID: $PID8]"

nohup uv run python -u scripts/runrun.py --mode irl_recursive_rls --output-dir "$LOG_DIR" --seeds "$SEEDS" --seed-parallel --max-seed-workers "$MAX_SEED_WORKERS" > "$LOG_DIR/logs_irl_recursive_rls.txt" 2>&1 &
PID9=$!
echo "Started Proposed (IRL aggregate) [PID: $PID9]"


echo "$PID1" > "$LOG_DIR/.pids_irl.txt"
echo "$PID2" > "$LOG_DIR/.pids_expert_surrogate_no_second_term.txt"
echo "$PID3" > "$LOG_DIR/.pids_irl_known_dynamics.txt"
echo "$PID6" > "$LOG_DIR/.pids_expert_surrogate_known_dynamics_no_second_term.txt"
echo "$PID4" > "$LOG_DIR/.pids_softql.txt"
echo "$PID7" > "$LOG_DIR/.pids_softvi_known_dynamics.txt"
echo "$PID8" > "$LOG_DIR/.pids_irl_known_dynamics_every_iter.txt"
echo "$PID9" > "$LOG_DIR/.pids_irl_recursive_rls.txt"

echo ""
echo "All experiments started in background!"
echo "You can safely close the terminal. Processes will continue running."
echo ""
echo "PID files saved: $LOG_DIR/.pids_irl.txt, $LOG_DIR/.pids_expert_surrogate_no_second_term.txt, $LOG_DIR/.pids_irl_known_dynamics.txt, $LOG_DIR/.pids_expert_surrogate_known_dynamics_no_second_term.txt, $LOG_DIR/.pids_softql.txt, $LOG_DIR/.pids_softvi.txt, $LOG_DIR/.pids_softvi_known_dynamics.txt, $LOG_DIR/.pids_irl_known_dynamics_every_iter.txt"
echo "To check progress: tail -f $LOG_DIR/logs_irl.txt (or logs_expert_surrogate_no_second_term.txt, logs_irl_known_dynamics.txt, logs_expert_surrogate_known_dynamics_no_second_term.txt, logs_softql.txt, logs_softvi.txt, logs_softvi_known_dynamics.txt, logs_irl_known_dynamics_every_iter.txt)"
echo "To check if still running: ps -p $PID1 $PID2 $PID3 $PID4 $PID6 $PID7 $PID8 $PID9"
echo ""
echo "Waiting for all experiments to finish..."
echo "(Press Ctrl+C to detach - processes will continue in background)"

trap 'echo ""; echo "Detached from processes. They will continue running in background."; exit 0' INT TERM

wait $PID1 $PID2 $PID3 $PID4 $PID6 $PID7 $PID8 $PID9

echo "All experiments finished!"

echo "Generating plots..."
uv run python scripts/runrun.py --mode plot --output-dir "$LOG_DIR"

echo "Done! Check $LOG_DIR/comparison_curves.pdf"