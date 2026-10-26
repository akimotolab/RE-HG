# blackrl


## setup

```bash
uv sync
```

## execute

### 2PMG (discrete)
```bash
nohup bash sh/run.sh &
```

### Configurable MDP(Hard)
```bash
nohup bash sh/run_conf_continuous.sh &
```

### 2PMG (continuous)
```bash
BENEFIT_BASELINE_MODE=online-online(or online-target) bash sh/run_2pmg_continuous.sh
```

### Synthetic task
```bash
uv run scripts/run_conf_discrete_synthetic.py
```