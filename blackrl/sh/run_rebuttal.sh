#!/bin/bash
# =============================================================================
# Rebuttal experiments for the Building Thermal Control (steep) task.
#
#   Exp 1a Myopic follower     : T-step Riccati behavior, T in {12, 16, 20, 30}.
#                                Interpretable, but the severity fades as the
#                                leader stabilizes the building, so the
#                                perturbation shrinks exactly as the leader
#                                converges (measured TV drops from ~0.34 to
#                                ~0.02 over training at T=12).
#   Exp 1b Under-reacting      : behavioral gain (1-alpha) K*, alpha in
#          follower              {0.05, 0.1, 0.2, 0.35}, giving a PERSISTENT,
#                                configuration-independent TV of about
#                                0.09 / 0.15 / 0.27 / 0.43. This is the primary
#                                robustness curve.
#                                (TReH's explicit condition; ZqQ3-Q1)
#   Exp 2a Span mismatch       : true Q gains normalized off-diagonal coupling,
#                                delta in {0.2, 0.4, 0.6, 0.8} (PD for |d|<1),
#                                giving irreducible misspecification of about
#                                0.10 / 0.20 / 0.30 / 0.37 -- the same
#                                occupancy-weighted scale as Exp 2b.
#   Exp 2b Under-specification : freeze state-cost features in the IRL model.
#                                Severity is swept so the measured irreducible
#                                misspecification is roughly evenly spaced
#                                (~0.06 / 0.14 / 0.26 / 0.52); the two drop
#                                orders (cost vs excitation) contrast which
#                                directions actually matter.
#   Exp 2c Over-specification  : add 2/5 redundant cross-term features (true
#                                coefficients are zero). EXCLUDED from "all":
#                                additive cross-term parameters and
#                                multiplicative diagonal ones have Jacobian
#                                scales ~2 orders apart, so Sigma_k is dominated
#                                by the cross directions and the truncation
#                                discards the diagonal ones. Run explicitly
#                                (--exp 2c) for diagnostics only.
#                                (TReH's explicit condition; ZqQ3-W1, Xcsj-W1)
#   Exp 3  Oracle LR sweep     : lr in {0.01, 0.03, 0.3} (clip fixed at 1.0)
#                                (Xcsj-Q2: "same regularization" Oracle)
#   Exp 4  eps sensitivity     : eig threshold in {1e-6, 1e-5, 1e-4, 3e-4, 1e-3}.
#                                Sigma_k's largest eigenvalue is ~2e-4..8e-4 on
#                                this task, so the sweep must bracket that
#                                range: at eps >= 1e-3 every direction is cut,
#                                U_k = 0, and no reward is recovered at all
#                                (a useful endpoint, but only one is needed).
#                                (TReH-W2: role of truncation)
#   Exp 5  Joint pooled IRL    : one shared reward from ALL accumulated
#                                demonstrations, re-solved every M=10 leader
#                                iterations (chunked). Three arms: without
#                                regularization (the formulation the reviewer
#                                names), with a ridge penalty, and with early
#                                stopping. The regularized arms matter because
#                                the unregularized one differs from RE-HG in
#                                two ways at once; they separate "pooled vs
#                                sequential" from "regularized vs not".
#                                (Xcsj-Q3 / meta-review)
#
# All other hyperparameters identical to the paper (Table 6).
# Usage:
#   bash sh/run_rebuttal.sh [--n-seeds N] [--out DIR] [--exp 1a|1b|1c|2a|2b|2c|3|3b|3c|4|5|all]
#                           [--modes "irl_recursive_rls ..."]
#
# --modes restricts which methods are launched. Only the modes that run the
# local IRL (irl_recursive_rls, irl_known_dynamics) depend on the reward-model
# solver; softvi and non_irl do not, so after a change confined to the IRL you
# can re-run just the former and keep the existing results for the latter.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_DIR="data/internal-test/rebuttal"
N_SEEDS=3
EXP="all"
MODES=""          # empty = every mode; otherwise a space-separated whitelist

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)     BASE_DIR="$2"; shift 2 ;;
        --n-seeds) N_SEEDS="$2";  shift 2 ;;
        --exp)     EXP="$2";      shift 2 ;;
        --modes)   MODES="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SEEDS=$(seq 0 $((N_SEEDS - 1)))
mkdir -p "$BASE_DIR"
PIDS=()

launch () {
    # launch <out_dir> <mode> <seed> [extra args...]
    local out_dir="$1"; local mode="$2"; local seed="$3"; shift 3
    # --modes lets you re-run only the modes that were affected by a change,
    # e.g. --modes "irl_recursive_rls" after touching the local IRL. Results
    # for the untouched modes stay on disk and are picked up by the plots.
    if [[ -n "$MODES" && " $MODES " != *" $mode "* ]]; then
        return 0
    fi
    mkdir -p "$out_dir"
    local log="$out_dir/logs_${mode}_seed${seed}.txt"
    nohup uv run python -u scripts/run_conf_continuous.py \
        --mode "$mode" --output-dir "$out_dir" --seed "$seed" "$@" \
        > "$log" 2>&1 &
    PIDS+=($!)
    echo "  [$!] $out_dir mode=$mode seed=$seed $*"
}

# ---- Exp 1: suboptimal follower ---------------------------------------------
if [[ "$EXP" == "1" || "$EXP" == "1a" || "$EXP" == "all" ]]; then
    echo "=== Exp 1a: myopic follower (early-stopped Riccati) ==="
    for T in 12 16 20 30; do
        for seed in $SEEDS; do
            for mode in softvi irl_recursive_rls; do
                launch "$BASE_DIR/exp1a_myopic_T${T}" "$mode" "$seed" \
                    --follower-riccati-iters "$T"
            done
        done
    done
fi

# ---- Exp 1b: under-reacting follower (persistent suboptimality) --------------
if [[ "$EXP" == "1b" || "$EXP" == "all" ]]; then
    echo "=== Exp 1b: under-reacting follower, gain (1-alpha) K* ==="
    for AL in 0.05 0.1 0.2 0.35; do
        for seed in $SEEDS; do
            for mode in softvi irl_recursive_rls; do
                launch "$BASE_DIR/exp1b_shrink_a${AL}" "$mode" "$seed" \
                    --follower-gain-shrink "$AL"
            done
        done
    done
fi

# ---- Exp 1c: suboptimal follower on the DISCRETE 2-player MG ----------------
# One-hot features make Assumption 1 exact on a finite state-action space, so
# this variant tests the optimal-follower assumption WITHOUT confounding it
# with reward-model misspecification -- something the continuous runs cannot
# do, since both assumptions are in play there. Truncating soft value
# iteration at N sweeps gives TV distances from the exact best response of
# about 0.23 / 0.12 / 0.04 / 0.006 (exact here: discrete actions, no MC).
if [[ "$EXP" == "1c" || "$EXP" == "all" ]]; then
    echo "=== Exp 1c: suboptimal follower, discrete 2-player MG ==="
    for N in 1 2 4 7; do
        for mode in softvi irl_recursive_rls; do
            if [[ -n "$MODES" && " $MODES " != *" $mode "* ]]; then continue; fi
            out="$BASE_DIR/exp1c_discrete_sweeps${N}"
            mkdir -p "$out"
            log="$out/logs_${mode}.txt"
            nohup uv run python -u scripts/runrun.py \
                --mode "$mode" --output-dir "$out" \
                --seeds "$(seq -s, 0 $((N_SEEDS - 1)))" \
                --follower-softvi-iters "$N" \
                > "$log" 2>&1 &
            PIDS+=($!)
            echo "  [$!] $out mode=$mode sweeps=$N"
        done
    done
fi

# ---- Exp 2: feature misspecification ----------------------------------------
if [[ "$EXP" == "2" || "$EXP" == "2a" || "$EXP" == "all" ]]; then
    echo "=== Exp 2a: span mismatch (off-diagonal true Q) ==="
    for D in 0.2 0.4 0.6 0.8; do
        for seed in $SEEDS; do
            for mode in softvi irl_recursive_rls; do
                launch "$BASE_DIR/exp2_misspec_d${D}" "$mode" "$seed" \
                    --misspec-delta "$D"
            done
        done
    done
fi

# ---- Exp 2b: under-specification (features removed) -------------------------
if [[ "$EXP" == "2b" || "$EXP" == "all" ]]; then
    echo "=== Exp 2b: under-specification (frozen state-cost features) ==="
    # Main curve: freeze the dominant cost feature, sweeping severity so the
    # measured misspecification amount is evenly spaced.
    for SEV in 0.02 0.05 0.10 0.25; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp2b_lowspec_cost_sev${SEV}" "irl_recursive_rls" "$seed" \
                --lowspec-drop 3 --lowspec-order cost --lowspec-severity "$SEV"
        done
    done
    # Contrast: freezing the least-excited features entirely is nearly harmless
    # (the directions truncation itself would discard).
    for seed in $SEEDS; do
        launch "$BASE_DIR/exp2b_lowspec_excitation_full" "irl_recursive_rls" "$seed" \
            --lowspec-drop 3 --lowspec-order excitation --lowspec-severity 1.0
    done
fi

# ---- Exp 2c: over-specification (redundant features added) -------------------
if [[ "$EXP" == "2c" ]]; then
    echo "=== Exp 2c: over-specification (redundant cross-term features) ==="
    for NP in 2 5; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp2c_overspec_p${NP}" "irl_recursive_rls" "$seed" \
                --overspec-pairs "$NP"
        done
    done
fi

# ---- Exp 3: Oracle learning-rate sweep --------------------------------------
if [[ "$EXP" == "3" || "$EXP" == "all" ]]; then
    echo "=== Exp 3: Oracle LR sweep (clip fixed at 1.0, as all methods) ==="
    for LR in 0.01 0.03 0.3; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp3_oracle_lr${LR}" "softvi" "$seed" \
                --lr "$LR"
        done
    done
fi

# ---- Exp 3b: the remaining magnitude-only controls for the Oracle -----------
# Xcsj asks whether RE-HG's advantage is reproducible by conditioning the
# Oracle's update alone. Exp 3 covers the learning rate; this adds the two
# remaining knobs.
#
#   grad-norm : g <- g/||g||, so every step has exactly the same length.
#               Stricter than clipping, which leaves small gradients alone.
#               Swept over lr because after normalization lr IS the step size.
#   lr + more : the low-lr arms of Exp 3 had large seed variance (about +-16),
#     episodes   so a slower Oracle may simply be gradient-noise limited.
#               Same learning rates with 300 episodes per iteration instead of
#               100 tests that directly.
#
# Both are positively homogeneous rescalings and therefore cannot change the
# update direction; ``cos_pre_post`` in the logs records this (it must stay 1).
if [[ "$EXP" == "3b" ]]; then
    echo "=== Exp 3b-i: Oracle with gradient normalization ==="
    for LR in 0.003 0.01 0.03 0.1; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp3b_oracle_gnorm_lr${LR}" "softvi" "$seed" \
                --grad-normalize --lr "$LR"
        done
    done

    echo "=== Exp 3b-ii: Oracle at low LR with 3x episodes ==="
    for LR in 0.01 0.03; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp3b_oracle_lr${LR}_ep800" "softvi" "$seed" \
                --lr "$LR" --n-episodes 800
        done
    done

    echo "=== Exp 3b-iii: RE-HG reference under the same episode budget ==="
    for seed in $SEEDS; do
        launch "$BASE_DIR/exp3b_rehg_ep800" "irl_recursive_rls" "$seed" \
            --n-episodes 800
    done
fi

# ---- Exp 3c: matched regularization, done properly --------------------------
# Xcsj's third requested control ("matched regularization") is not operationally
# defined as stated. The realization that IS well defined here scales the
# hypergradient's influence term by beta, because that is precisely the quantity
# we claim truncation suppresses: the measured ratio ||infl||/||dir|| is
# 0.03-0.15 for RE-HG against 0.2-0.5 for the Oracle, so beta around 0.1-0.3
# matches the Oracle's ratio to RE-HG's.
#
# Note this is NOT a homogeneous rescaling of the full gradient: dir + beta*infl
# changes the direction, so it can in principle reproduce truncation's effect.
# That makes it a real test of our own claim rather than a strawman.
#
#   beta = 1.0 : unmodified Oracle (the arm already in the paper)
#   beta -> 0  : direct term only
# If some beta reproduces RE-HG, the mechanism is a scalar suppression of the
# second term. If no beta does, truncation acts subspace-selectively and cannot
# be reduced to a scalar -- the stronger outcome for us.
if [[ "$EXP" == "3c" ]]; then
    echo "=== Exp 3c: Oracle with the influence term scaled by beta ==="
    for BETA in 0.03 0.1 0.3 0.5; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp3c_oracle_beta${BETA}" "softvi" "$seed" \
                --influence-scale "$BETA"
        done
    done
fi

# ---- Exp 4: eigenvalue-threshold sensitivity --------------------------------
if [[ "$EXP" == "4" || "$EXP" == "all" ]]; then
    echo "=== Exp 4: epsilon sensitivity (RE-HG truncation threshold) ==="
    # Sigma_k is in (r_F/beta)-units (beta=0.1): 1e-2 here equals the former 1e-4.
    for EPS in 1e-2; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp4_eps${EPS}" "irl_recursive_rls" "$seed" \
                --eig-threshold "$EPS"
        done
    done
fi

# ---- Exp 5: joint pooled IRL baseline ---------------------------------------
if [[ "$EXP" == "5" || "$EXP" == "all" ]]; then
    echo "=== Exp 5: joint pooled multi-environment IRL (chunk M=10) ==="
    # Unregularized arm: the "existing multi-task IRL formulation" the reviewer
    # names. Differs from RE-HG in two ways at once (pooled vs sequential, and
    # unregularized vs truncated), so it cannot attribute the difference.
    for seed in $SEEDS; do
        launch "$BASE_DIR/exp5_joint_irl" "irl_joint_pooled" "$seed" \
            --joint-chunk 10
    done
    # Regularized arms: same pooling, now WITH shrinkage. Ridge and eigenvalue
    # truncation are the soft and hard members of the same family of spectral
    # filters, so this isolates the remaining effect of the sequential spectral
    # form. Report the baseline at its best setting.
    for RG in 0.3 0.6 0.9; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp5_joint_ridge${RG}" "irl_joint_pooled" "$seed" \
                --joint-chunk 10 --joint-ridge "$RG"
        done
    done
    # Early stopping is itself a spectral filter of the same family and costs
    # nothing to implement: fewer solver iterations = stronger regularization.
    for IT in 25 50 150; do
        for seed in $SEEDS; do
            launch "$BASE_DIR/exp5_joint_earlystop${IT}" "irl_joint_pooled" "$seed" \
                --joint-chunk 10 --joint-iters "$IT"
        done
    done
    # Reference RE-HG runs in the same directory tree for a like-for-like plot.
    for seed in $SEEDS; do
        launch "$BASE_DIR/exp5_joint_irl" "irl_recursive_rls" "$seed"
    done
fi

echo ""
echo "Launched ${#PIDS[@]} runs. Monitor: tail -f $BASE_DIR/*/logs_*.txt"
trap 'echo "Detached. Processes continue in background."; exit 0' INT TERM
wait "${PIDS[@]}"

echo ""
echo "Identifiability diagnostic (excitation vs cost vs controllability)..."
uv run python scripts/diagnose_excitation.py \
    --json "$BASE_DIR/excitation_diagnostic.json" \
    > "$BASE_DIR/excitation_diagnostic.txt" 2>&1 || true
echo "  -> $BASE_DIR/excitation_diagnostic.txt"

echo ""
echo "All runs finished. Aggregating each config dir..."
for d in "$BASE_DIR"/exp*/; do
    uv run python scripts/run_conf_continuous.py \
        --mode aggregate --output-dir "$d" --n-seeds "$N_SEEDS" || true
done

echo ""
echo "Drawing rebuttal figures..."
uv run python scripts/plot_rebuttal.py --dir "$BASE_DIR" --n-seeds "$N_SEEDS" || true

echo "Done. Results under $BASE_DIR/, figures under $BASE_DIR/figures/"