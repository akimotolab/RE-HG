# blackrl

Main code for RE-HG (Reward-Estimated Hypergradient). This project covers the
two-player Markov game experiments (discrete and continuous), the Configurable MDP
(Hard) experiment, the synthetic task, and the additional ablation experiments.
The Configurable MDP (Easy) experiment is in
[`../blackrlforconfeasycodeonly`](../blackrlforconfeasycodeonly).

## Setup

Requires Python 3.14 and uv.

```bash
uv sync
```

Run every command below from this directory (`blackrl/`).

## Experiments

Each shell script starts all methods and seeds as background `nohup` processes and
writes one log file per run to the output directory. When every run has finished,
the script aggregates the results and saves the figures (for example
`comparison_curves.pdf`) in the same directory. Press `Ctrl+C` to detach from a
script; the runs keep going in the background.

### 2-player Markov game (discrete)

```bash
nohup bash sh/run.sh &
```

- Script: `scripts/runrun.py`
- Output: `data/internal-test/discrete_toyenv/...` (set by `LOG_DIR` in `sh/run.sh`)
- The seeds (`SEEDS`) and the number of parallel workers (`MAX_SEED_WORKERS`) are set
  at the top of `sh/run.sh`.

To run a single method:

```bash
uv run python scripts/runrun.py --mode irl_recursive_rls --output-dir <OUTPUT_DIR> --seeds 0,1,2
uv run python scripts/runrun.py --mode plot --output-dir <OUTPUT_DIR>
```

### 2-player Markov game (continuous)

```bash
BENEFIT_BASELINE_MODE=online-online bash sh/run_2pmg_continuous.sh [--out <OUTPUT_DIR>] [--n-seeds <N>]
```

- Script: `scripts/run_2pmg_continuous.py`
- `BENEFIT_BASELINE_MODE` is `online-online` (default) or `online-target`.
- Optional environment variables: `DEBUG_TRACE=1`, `DEBUG_TRACE_EVERY=<k>`,
  `DEBUG_TRACE_JSONL=0|1`, `FULL_BATCH_CUMULATIVE=1`.
- Default: 3 seeds, output under `data/internal-test/2pmg_continuous/`.

### Configurable MDP (Hard)

```bash
nohup bash sh/run_conf_continuous.sh --out <OUTPUT_DIR> [--n-seeds <N>] &
```

- Script: `scripts/run_conf_continuous.py` (building thermal control task)
- Default: 3 seeds.

To run a single method and seed, then aggregate and plot:

```bash
uv run python scripts/run_conf_continuous.py --mode irl_recursive_rls --output-dir <OUTPUT_DIR> --seed 0
uv run python scripts/run_conf_continuous.py --mode aggregate --output-dir <OUTPUT_DIR> --n-seeds 3
uv run python scripts/run_conf_continuous.py --mode plot --output-dir <OUTPUT_DIR> --n-seeds 3
```

### Synthetic task

```bash
uv run python scripts/run_conf_discrete_synthetic.py --output-dir <OUTPUT_DIR> --seed 0
```

Run `uv run python scripts/run_conf_discrete_synthetic.py --help` for all options
(`--n-iterations`, `--theta0`, `--theta-lr`, ...).

### Additional experiments (ablations)

`sh/run_rebuttal.sh` runs the additional experiments on the Configurable MDP task:
suboptimal followers, reward-feature misspecification, Oracle learning-rate and
regularization sweeps, eigenvalue-threshold sensitivity, and a pooled joint-IRL
baseline. The header of the script describes each experiment.

```bash
bash sh/run_rebuttal.sh [--exp 1a|1b|1c|2a|2b|2c|3|3b|3c|4|5|all] [--n-seeds <N>] [--out <DIR>] [--modes "<mode> ..."]
```

Results go to `data/internal-test/rebuttal/` by default, and figures to its
`figures/` subdirectory.

## Methods

| `--mode` | Method |
| --- | --- |
| `irl_recursive_rls` | RE-HG (proposed) |
| `irl_known_dynamics` | RE-HG (Single) |
| `irl_unknown_recursive_rls`, `irl_unknown_dynamics` | RE-HG / RE-HG (Single) with unknown dynamics (continuous 2-player MG only) |
| `softvi` | Oracle (white-box follower) |
| `non_irl` | No-Influence (hypergradient without the influence term) |
| `irl_joint_pooled` | Pooled joint-IRL baseline (Configurable MDP only) |

The discrete 2-player MG script (`scripts/runrun.py`) uses its own mode names. See
`--help` for the full list.

## Experiment data

`data/internal/` contains the outputs of our experiment runs: per-seed logs
(`logs_*.txt`), per-seed and aggregated statistics (`stats_*.pkl`,
`aggregated_*.pkl`), and figures (`*.pdf`).

| Directory | Experiment |
| --- | --- |
| `data/internal/discrete_toyenv/` | 2-player Markov game (discrete) |
| `data/internal/discrete_joint_cao/` | 2-player Markov game (discrete), pooled joint-IRL baseline |
| `data/internal/2pmg_continuous/` | 2-player Markov game (continuous), `online-online` and `online-target` |
| `data/internal/conf_continuous/next1/` | Configurable MDP (Hard) |
| `data/internal/conf_continuous/easy/` | Configurable MDP (Easy) |
| `data/internal/conf_discrete_synth2/` | Synthetic task |
| `data/internal/rebuttal/` | Additional experiments (ablations) |

New runs write to `data/internal-test/` by default, so they do not overwrite this
data.
