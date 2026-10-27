# blackrl: Configurable MDP (Easy)

Code for the Configurable MDP (Easy) experiment with RE-HG (Reward-Estimated
Hypergradient), run on the building thermal control task. The other experiments are in
[`../blackrl`](../blackrl).

## Setup

Requires Python 3.14 and uv.

```bash
uv sync
```

Run every command below from this directory (`blackrlforconfeasycodeonly/`).

## Running

```bash
nohup bash sh/run_conf_continuous.sh --out <OUTPUT_DIR> [--n-seeds <N>] &
```

The script runs every method (`softvi`, `irl_known_dynamics`, `irl_recursive_rls`,
`non_irl`) for each seed in parallel (default: 3 seeds). Each run writes a log to
`<OUTPUT_DIR>/logs_<mode>_seed<seed>.txt`. When every run has finished, the script
aggregates the results and saves the figures in `<OUTPUT_DIR>`. Press `Ctrl+C` to
detach; the runs keep going in the background.

To run a single method and seed, then aggregate and plot:

```bash
uv run python scripts/run_conf_continuous.py --mode irl_recursive_rls --output-dir <OUTPUT_DIR> --seed 0
uv run python scripts/run_conf_continuous.py --mode aggregate --output-dir <OUTPUT_DIR> --n-seeds 3
uv run python scripts/run_conf_continuous.py --mode plot --output-dir <OUTPUT_DIR> --n-seeds 3
```

## Methods

| `--mode` | Method |
| --- | --- |
| `irl_recursive_rls` | RE-HG (proposed) |
| `irl_known_dynamics` | RE-HG (Single) |
| `softvi` | Oracle (white-box follower) |
| `non_irl` | No-Influence (hypergradient without the influence term) |
