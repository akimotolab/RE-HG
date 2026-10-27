# Reward-Estimated Hypergradient for Bilevel Reinforcement Learning with Black-Box Follower

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

RE-HG (Reward-Estimated Hypergradient) lets a leader optimize its policy against a
black-box follower whose reward is unknown. It recovers the follower's reward from
observed trajectories with inverse reinforcement learning (IRL) and uses the estimate
to compute the hypergradient.

## Repository layout

The repository has two self-contained `uv` projects. Each one has its own environment
and lock file.

| Directory | Contents |
| --- | --- |
| [`blackrl/`](blackrl/) | Main code: discrete and continuous two-player Markov games, Configurable MDP (Hard), the synthetic task, and the additional ablation experiments. |
| [`blackrlforconfeasycodeonly/`](blackrlforconfeasycodeonly/) | Code for the Configurable MDP (Easy) experiment. |

The outputs of our experiment runs (logs, statistics and figures) for all experiments,
including Configurable MDP (Easy), are in
[`blackrl/data/internal/`](blackrl/data/internal/). See
[`blackrl/README.md`](blackrl/README.md#experiment-data) for which directory holds which
experiment.

Both projects share the same package layout:

```
blackrl/            # Python package
  agents/           #   leader policy, follower models, IRL (MDCE-IRL), soft Q-learning
  algos/            #   bilevel RL loop, hypergradient estimators, leader actor / critic
  envs/             #   environments (discrete toy MG, 2-player MG, Configurable MDP, ...)
scripts/            # entry points for each experiment and plotting
sh/                 # shell scripts that launch every method and seed in parallel
```

## Requirements

- Python 3.14
- uv

Install the dependencies inside the project you want to run:

```bash
cd blackrl          # or: cd blackrlforconfeasycodeonly
uv sync
```

## Running the experiments

Run every command from inside the project directory (`blackrl/` or
`blackrlforconfeasycodeonly/`). Each shell script launches all methods and seeds as
background processes. When they finish, it aggregates the results and writes the plots
to the output directory.

The project READMEs give the commands and options for each experiment:

- [`blackrl/README.md`](blackrl/README.md): 2-player Markov game (discrete and continuous),
  Configurable MDP (Hard), synthetic task, ablations
- [`blackrlforconfeasycodeonly/README.md`](blackrlforconfeasycodeonly/README.md):
  Configurable MDP (Easy)

### Methods

The scripts select a method with `--mode`. The main methods are:

| `--mode` | Method |
| --- | --- |
| `irl_recursive_rls` | RE-HG (proposed) |
| `irl_known_dynamics` | RE-HG (Single) |
| `softvi` | Oracle (white-box follower) |
| `non_irl` | No-Influence (hypergradient without the influence term) |
| `aggregate` / `plot` | Aggregate per-seed results / draw figures |

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
