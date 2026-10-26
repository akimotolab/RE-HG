"""Learning curves for the discrete pooled-IRL comparison (Xcsj W2/Q3).

RE-HG aggregates per-environment IRL estimates by projection and truncation
(Algorithm 2); the pooled baseline fits one shared reward to all accumulated
demonstrations at once, in the formulation of Cao, Cohen and Szpruch (2021,
Sec. 7.2). Both see exactly the same datasets D_1..D_k, so any difference is
attributable to the aggregation rule rather than to having multi-environment
data at all.

The comparison is most informative early: once K is large both estimators
approach the same solution set, so a plot of tail performance alone can show
"no difference" while the transient differs. Hence curves, not endpoints.

Usage:
    uv run python scripts/plot_joint_cao.py \
        --run "RE-HG:data/internal/discrete_rehg_ref" \
        --run "Pooled (Cao 7.2):data/internal/discrete_joint_cao" \
        --out figures_joint_cao/discrete_joint_cao.pdf

Each --run is "LABEL:DIRECTORY". Seeds are discovered from the filenames, so
runs with different seed sets still plot (they are simply averaged over
whatever is present -- check the printed seed lists before drawing conclusions
from a comparison across mismatched seeds).
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Panels: (stats key, axis label, panel title)
_PANELS = [
    ("leader_return", "Leader return", "performance"),
    ("hg_cosine_similarity", r"cos(HG$_{\rm est}$, HG$_{\rm oracle}$)",
     "hypergradient alignment"),
    ("irl_reward_info_rank", r"rank($\bar\Sigma_K$)", "identified dimensions"),
]

_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def load_dir(d: Path, mode: str | None = None) -> tuple[dict[int, dict], str]:
    """Load the per-seed stats pickles for ONE mode in ``d``.

    A directory may hold several modes (the legacy result folders do), and
    silently averaging across them would compare a mixture rather than a
    method. So the mode must be unambiguous: either there is exactly one, or
    the caller names it.
    """
    found: dict[str, dict[int, dict]] = {}
    for p in sorted(d.glob("stats_*__seed_*.pkl")):
        m = re.search(r"stats_(.+)__seed_(\d+)\.pkl$", p.name)
        if not m:
            continue
        found.setdefault(m.group(1), {})[int(m.group(2))] = p

    if not found:
        return {}, ""
    if mode is None:
        if len(found) > 1:
            raise SystemExit(
                f"{d} holds several modes: {sorted(found)}.\n"
                f"  Disambiguate with LABEL:DIRECTORY:MODE"
            )
        mode = next(iter(found))
    elif mode not in found:
        raise SystemExit(f"{d} has no stats for mode {mode!r}; available: {sorted(found)}")

    out = {}
    for seed, p in found[mode].items():
        with open(p, "rb") as f:
            out[seed] = pickle.load(f)
    return out, mode


def series(stats_by_seed: dict[int, dict], key: str):
    """Mean and std across seeds, truncated to the shortest run."""
    arrs = [np.asarray(s[key], dtype=np.float64)
            for s in stats_by_seed.values()
            if s.get(key) is not None and len(s.get(key, [])) > 0]
    if not arrs:
        return None
    n = min(len(a) for a in arrs)
    A = np.stack([a[:n] for a in arrs])
    return A.mean(0), A.std(0), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help='"LABEL:DIRECTORY", repeatable')
    ap.add_argument("--out", default="figures_joint_cao/discrete_joint_cao.pdf")
    ap.add_argument("--smooth", type=int, default=1,
                    help="moving-average window over iterations (1 = raw)")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        parts = spec.split(":")
        if len(parts) == 2:
            label, path, mode_req = parts[0], parts[1], None
        elif len(parts) == 3:
            label, path, mode_req = parts
        else:
            raise SystemExit(
                f"--run must be LABEL:DIRECTORY[:MODE], got {spec!r}")
        d = Path(path)
        if not d.is_dir():
            raise SystemExit(f"not a directory: {d}")
        stats, mode = load_dir(d, mode_req)
        if not stats:
            raise SystemExit(f"no stats_*__seed_*.pkl found in {d}")
        print(f"{label:24s} mode={mode:34s} seeds={sorted(stats)} "
              f"n_iters={min(len(s.get('leader_return', [])) for s in stats.values())}")
        runs.append((label, stats))

    seed_sets = {tuple(sorted(s)) for _, s in runs}
    if len(seed_sets) > 1:
        print("\n  WARNING: the runs do not share the same seeds "
              f"({seed_sets}). Differences may be seed effects rather than "
              "method effects.\n")

    def smooth(x):
        w = max(1, int(args.smooth))
        if w == 1:
            return x
        k = np.ones(w) / w
        return np.convolve(x, k, mode="valid")

    present = [p for p in _PANELS if any(series(s, p[0]) for _, s in runs)]
    if not present:
        raise SystemExit("none of the expected keys are present in the stats")

    fig, axes = plt.subplots(1, len(present), figsize=(5.4 * len(present), 4.3),
                             squeeze=False, layout="constrained")
    for pi, (key, ylab, title) in enumerate(present):
        ax = axes[0][pi]
        for ri, (label, stats) in enumerate(runs):
            got = series(stats, key)
            if got is None:
                continue
            m, s, n = got
            m, s = smooth(m), smooth(s)
            x = np.arange(len(m))
            col = _COLORS[ri % len(_COLORS)]
            ax.plot(x, m, label=label, color=col, linewidth=2.0,
                    linestyle="--" if ri == 0 else "-")
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.15)
        ax.set_xlabel("leader iteration")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Discrete: projection-and-truncation vs pooled multi-environment IRL",
                 y=1.04)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # Tail summary, so the numbers can be quoted without reading the figure.
    print("\ntail-20 mean:")
    for key, _, _ in present:
        print(f"  {key}")
        for label, stats in runs:
            got = series(stats, key)
            if got is None:
                continue
            m, _, _ = got
            print(f"    {label:24s} {m[-20:].mean():.4f}")


if __name__ == "__main__":
    main()