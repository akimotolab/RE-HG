"""Draw every rebuttal figure from the raw stats pickles.

Scans a rebuttal output directory (default ``data/internal/rebuttal``), works out
which experiment each sub-directory belongs to from its name, and writes the
corresponding figures. Safe to re-run: experiments with no data are skipped.

    uv run python scripts/plot_rebuttal.py [--dir DIR] [--out DIR] [--n-seeds N]

Figures produced
----------------
exp1_robustness.pdf      performance / HG cosine / reward corr vs follower TV
exp1_tv_trace.pdf        follower TV against iteration (shows 1a decays, 1b flat)
exp2_misspec.pdf         same three metrics vs measured misspecification amount
exp2c_overspec.pdf       redundant cross-term coefficients over iterations
exp3_terms.pdf           influence/direct norm ratio, clipping activity
exp3_oracle_lr.pdf       final performance vs Oracle learning rate
exp4_eps.pdf             performance and surviving rank vs truncation threshold
exp5_joint_cost.pdf      RE-HG vs joint pooled IRL: performance and solve time
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
import scienceplots  # noqa: F401

_STYLE = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "lines.linewidth": 2.0, "lines.markersize": 6.0, "figure.dpi": 130,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}
_C = {
    "RE-HG": "#0072B2",
    "Oracle": "black",
    "Joint-IRL": "#D55E00",
    "contrast": "#009E73",
}
_MODE_LABEL = {
    "irl_recursive_rls": "RE-HG",
    "softvi": "Oracle",
    "irl_joint_pooled": "Joint-IRL",
    "irl_known_dynamics": "RE-HG (Single)",
    "non_irl": "No-Influence",
}


# ---------------------------------------------------------------- loading ----

def _baseline_dir(base: Path) -> Path:
    """Directory holding the unmodified (default-hyperparameter) runs.

    Prefer the epsilon sweep's default point; fall back to any directory that
    contains RE-HG runs. Note eps=1e-3 exceeds Sigma_k's largest eigenvalue on
    this task and recovers no reward at all, so it must never serve as the
    baseline.
    """
    for name in ("exp4_eps1e-4", "exp4_eps3e-4", "exp4_eps1e-5"):
        d = base / name
        if d.exists():
            return d
    return base / "exp4_eps1e-4"


def load_runs(cfg_dir: Path, mode: str, n_seeds: int) -> list[dict]:
    out = []
    for s in range(n_seeds):
        p = cfg_dir / f"stats_{mode}_seed{s}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                out.append(pickle.load(f))
    return out


def tail_mean(run: dict, key: str, frac: float = 0.1) -> float | None:
    """Mean of the last ``frac`` of a per-iteration series (final performance)."""
    v = run.get(key)
    if not v:
        return None
    a = np.asarray([x for x in v if x is not None], dtype=np.float64)
    if a.size == 0:
        return None
    k = max(1, int(round(a.size * frac)))
    return float(np.mean(a[-k:]))


def series_mean(run: dict, key: str) -> float | None:
    v = run.get(key)
    if not v:
        return None
    a = np.asarray([x for x in v if x is not None], dtype=np.float64)
    return float(np.mean(a)) if a.size else None


def agg(runs: list[dict], key: str, how="tail") -> tuple[float, float] | None:
    """Aggregate a metric across seeds -> (mean, std)."""
    f = tail_mean if how == "tail" else series_mean
    vals = [f(r, key) for r in runs]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def cfg_value(runs: list[dict], key: str) -> float | None:
    for r in runs:
        v = r.get("rebuttal_config", {}).get(key)
        if v is not None:
            return float(v)
    return None


def _errbar(ax, xs, means, stds, label, color, marker="o"):
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    means = np.asarray(means)[order]
    stds = np.asarray(stds)[order]
    ax.errorbar(xs, means, yerr=stds, label=label, color=color,
                marker=marker, capsize=3)


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------- Exp 1: suboptimal follower -

def plot_exp1(base: Path, out: Path, n_seeds: int) -> None:
    variants = {
        "Myopic follower: finite-horizon Riccati updates": sorted(base.glob("exp1a_myopic_T*")),
        "Under-reacting follower: gain shrinkage": sorted(base.glob("exp1b_shrink_a*")),
    }
    if not any(variants.values()):
        return

    metrics = [
        ("eval_leader_return", "Leader return (eval)"),
        ("hg_cosine_similarity", "HG cosine vs Oracle"),
        ("reward_pearson_corr", "Reward correlation"),
    ]
    fig, axes = plt.subplots(len(variants), 3, figsize=(15, 4.6 * len(variants)),
                             squeeze=False, layout="constrained")

    for row, (vname, dirs) in enumerate(variants.items()):
        if not dirs:
            for c in range(3):
                axes[row][c].axis("off")
            continue
        for col, (key, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            for mode in ("softvi", "irl_recursive_rls"):
                xs, ms, ss = [], [], []
                for d in dirs:
                    runs = load_runs(d, mode, n_seeds)
                    if not runs:
                        continue
                    # x-axis: TV averaged over training (the honest summary when
                    # the perturbation is not constant, as in 1a)
                    tv = agg(runs, "follower_tv", how="mean")
                    met = agg(runs, key, how="tail")
                    if tv is None or met is None:
                        continue
                    xs.append(tv[0]); ms.append(met[0]); ss.append(met[1])
                if xs:
                    lab = _MODE_LABEL[mode]
                    _errbar(ax, xs, ms, ss, lab, _C.get(lab, None),
                            "o" if mode == "irl_recursive_rls" else "s")
            ax.set_xlabel("follower TV (mean over training)", fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_title(vname if col == 1 else "", fontsize=26)
            ax.grid(alpha=0.3)
            if col == 2:
                ax.legend(fontsize=17)
    fig.suptitle("Robustness to a suboptimal follower", fontsize=26)
    _save(fig, out / "exp1_robustness.pdf")

    # TV traces: 1a decays as the leader converges, 1b stays flat.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False,
                             layout="constrained")
    for col, (vname, dirs) in enumerate(variants.items()):
        ax = axes[0][col]
        for d in dirs:
            runs = load_runs(d, "irl_recursive_rls", n_seeds)
            if not runs:
                continue
            arrs = [np.asarray(r["follower_tv"], dtype=np.float64)
                    for r in runs if r.get("follower_tv")]
            if not arrs:
                continue
            n = min(len(a) for a in arrs)
            A = np.stack([a[:n] for a in arrs])
            m, s = A.mean(0), A.std(0)
            ax.plot(m, label=d.name.split("_")[-1])
            ax.fill_between(np.arange(n), m - s, m + s, alpha=0.15)
        ax.set_xlabel("leader iteration", fontsize=20); ax.set_ylabel("follower TV", fontsize=20)
        ax.set_title(vname, fontsize=26); ax.grid(alpha=0.3); ax.legend(fontsize=17)
    fig.suptitle("Persistence of follower suboptimality", fontsize=26)
    _save(fig, out / "exp1_tv_trace.pdf")


# ------------------------------------------------ Exp 2: misspecification ----

def plot_exp2(base: Path, out: Path, n_seeds: int) -> None:
    groups = {
        "Basis mismatch": (sorted(base.glob("exp2a_misspec_d*"))
                             or sorted(base.glob("exp2_misspec_d*")), "misspec_amount"),
        "Representation-limited reward model": (sorted(base.glob("exp2b_lowspec_cost_sev*")),
                                   "lowspec_amount"),
    }
    contrast = sorted(base.glob("exp2b_lowspec_excitation*"))
    if not any(g[0] for g in groups.values()):
        return

    metrics = [
        ("eval_leader_return", "Leader return (eval)"),
        ("hg_cosine_similarity", "HG cosine vs Oracle"),
        ("reward_pearson_corr", "Reward correlation"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), squeeze=False,
                             layout="constrained")
    for col, (key, ylabel) in enumerate(metrics):
        ax = axes[0][col]
        for gi, (gname, (dirs, amount_key)) in enumerate(groups.items()):
            xs, ms, ss = [], [], []
            for d in dirs:
                runs = load_runs(d, "irl_recursive_rls", n_seeds)
                if not runs:
                    continue
                amt = cfg_value(runs, amount_key)
                met = agg(runs, key, how="tail")
                if amt is None or met is None:
                    continue
                xs.append(amt); ms.append(met[0]); ss.append(met[1])
            if xs:
                _errbar(ax, xs, ms, ss, gname,
                        ["#0072B2", "#D55E00"][gi], ["o", "s"][gi])
        # contrast point: least-excited features frozen entirely
        for d in contrast:
            runs = load_runs(d, "irl_recursive_rls", n_seeds)
            if not runs:
                continue
            amt = cfg_value(runs, "lowspec_amount")
            met = agg(runs, key, how="tail")
            if amt is None or met is None:
                continue
            ax.errorbar([amt], [met[0]], yerr=[met[1]], color=_C["contrast"],
                        marker="*", markersize=14, linestyle="none",
                        label="2b least-excited (contrast)")
        ax.set_xlabel("irreducible misspecification")
        if col == 0:
            ax.set_title("Reward-model misspecification", fontsize=16)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        if col == 2:
            ax.legend(fontsize=9)
    _save(fig, out / "exp2_misspec.pdf")


def plot_exp2c(base: Path, out: Path, n_seeds: int) -> None:
    dirs = sorted(base.glob("exp2c_overspec_p*"))
    if not dirs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False,
                             layout="constrained")
    ax = axes[0][0]
    for d in dirs:
        runs = load_runs(d, "irl_recursive_rls", n_seeds)
        arrs = [np.asarray(r["overspec_cross_absmax"], dtype=np.float64)
                for r in runs if r.get("overspec_cross_absmax")]
        if not arrs:
            continue
        n = min(len(a) for a in arrs)
        A = np.stack([a[:n] for a in arrs])
        m, s = A.mean(0), A.std(0)
        ax.plot(m, label=d.name.split("_")[-1])
        ax.fill_between(np.arange(n), m - s, m + s, alpha=0.15)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.5,
               label="true value = 0")
    ax.set_xlabel("leader iteration")
    ax.set_ylabel(r"max$_{ij}$ |recovered cross coeff.|")
    ax.set_title("Redundant coefficients $\\to$ 0", fontsize=16)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0][1]
    labels, ms, ss = [], [], []
    for d in [None] + dirs:
        if d is None:
            runs = load_runs(_baseline_dir(base), "irl_recursive_rls", n_seeds) or []
            lab = "baseline"
        else:
            runs = load_runs(d, "irl_recursive_rls", n_seeds)
            lab = d.name.split("_")[-1]
        met = agg(runs, "eval_leader_return", how="tail") if runs else None
        if met is None:
            continue
        labels.append(lab); ms.append(met[0]); ss.append(met[1])
    if labels:
        ax.bar(labels, ms, yerr=ss, capsize=4, color="#0072B2")
        ax.set_ylabel("Leader return (eval)")
        ax.set_title("performance unaffected", fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        _save(fig, out / "exp2c_overspec.pdf")


# ------------------------------------------- Exp 3: terms, clipping, Oracle --

def plot_exp3(base: Path, out: Path, n_seeds: int) -> None:
    # Term-wise diagnostics come from any run that logs them; prefer the
    # baseline epsilon run, else the first available RE-HG directory.
    cand = ([_baseline_dir(base)] + sorted(base.glob("exp*"))) if base.exists() else []
    runs_re, runs_or = [], []
    for d in cand:
        if not d.is_dir():
            continue
        runs_re = runs_re or load_runs(d, "irl_recursive_rls", n_seeds)
        runs_or = runs_or or load_runs(d, "softvi", n_seeds)
        if runs_re and runs_or:
            break
    if runs_re or runs_or:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False,
                                 layout="constrained")

        ax = axes[0][0]
        # Each method's OWN ratio on its OWN trajectories. Do not substitute
        # `hg_oracle_term_ratio` from an IRL run: that field applies the
        # exact-reward formula to surrogate-follower states and is off-policy,
        # which inflates the influence term by orders of magnitude.
        for runs, lab, col in ((runs_re, "RE-HG", _C["RE-HG"]),
                               (runs_or, "Oracle", _C["Oracle"])):
            arrs = [np.asarray(r["hg_term_ratio"], dtype=np.float64)
                    for r in runs if r.get("hg_term_ratio")]
            if not arrs:
                continue
            n = min(len(a) for a in arrs)
            A = np.stack([a[:n] for a in arrs])
            ax.plot(A.mean(0), label=lab, color=col,
                    linestyle="--" if lab == "RE-HG" else "-",
                    linewidth=2.6 if lab == "RE-HG" else 1.8)
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.5,
                   label="terms balanced")
        ax.set_yscale("log")
        ax.set_xlabel("leader iteration")
        ax.set_ylabel(r"$\|$influence$\|\,/\,\|$direct$\|$")
        ax.set_title("Exp 3 — term imbalance sets the direction", fontsize=12)
        ax.grid(alpha=0.3); ax.legend(fontsize=10)

        ax = axes[0][1]
        for runs, lab, col in ((runs_re, "RE-HG", _C["RE-HG"]),
                               (runs_or, "Oracle", _C["Oracle"])):
            arrs = [np.asarray(r["grad_norm_preclip"], dtype=np.float64)
                    for r in runs if r.get("grad_norm_preclip")]
            if not arrs:
                continue
            n = min(len(a) for a in arrs)
            ax.plot(np.stack([a[:n] for a in arrs]).mean(0), label=f"{lab} pre-clip",
                    color=col, linestyle="--" if lab == "RE-HG" else "-",
                    linewidth=2.6 if lab == "RE-HG" else 1.8)
            post = [np.asarray(r["grad_norm_postclip"], dtype=np.float64)
                    for r in runs if r.get("grad_norm_postclip")]
            if post:
                n2 = min(len(a) for a in post)
                ax.plot(np.stack([a[:n2] for a in post]).mean(0),
                        linestyle=":" if lab == "RE-HG" else "-.",
                        label=f"{lab} post-clip", color=col, alpha=0.75,
                        linewidth=2.4)
        ax.set_yscale("log"); ax.set_xlabel("leader iteration")
        ax.set_ylabel("gradient norm")
        ax.set_title("same clipped step for every method", fontsize=12)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        ax = axes[0][2]
        for runs, lab, col in ((runs_re, "RE-HG", _C["RE-HG"]),
                               (runs_or, "Oracle", _C["Oracle"])):
            arrs = [np.asarray(r["hg_influence_cosine"], dtype=np.float64)
                    for r in runs if r.get("hg_influence_cosine")]
            if not arrs:
                continue
            n = min(len(a) for a in arrs)
            ax.plot(np.stack([a[:n] for a in arrs]).mean(0), label=lab, color=col,
                    linestyle="--" if lab == "RE-HG" else "-",
                    linewidth=2.6 if lab == "RE-HG" else 1.8)
        ax.set_xlabel("leader iteration")
        ax.set_ylabel("cos(influence est., influence Oracle)")
        ax.set_title("influence-term direction", fontsize=12)
        ax.grid(alpha=0.3); ax.legend(fontsize=10)
        fig.suptitle("Exp 3 — why rescaling the update cannot fix the direction",
                     y=1.02)
        _save(fig, out / "exp3_terms.pdf")

    # Learning curves first: they must survive the early return below, which
    # fires whenever the lr sweep itself was not run.
    _plot_exp3_curves(base, out, n_seeds)

    # Oracle learning-rate / clip sweep
    lr_dirs = sorted(base.glob("exp3_oracle_lr*"))
    clip_dirs = sorted(base.glob("exp3_oracle_clip*"))
    if not (lr_dirs or clip_dirs):
        return
    fig, axes = plt.subplots(1, 2 if clip_dirs else 1,
                             figsize=(11 if clip_dirs else 6, 4.2), squeeze=False,
                             layout="constrained")
    for ci, (dirs, xname) in enumerate([(lr_dirs, "learning rate"),
                                        (clip_dirs, "clip norm")]):
        if not dirs:
            continue
        ax = axes[0][ci if clip_dirs else 0]
        xs, ms, ss = [], [], []
        for d in dirs:
            m = re.search(r"(?:lr|clip)([0-9.eE+-]+)$", d.name)
            runs = load_runs(d, "softvi", n_seeds)
            met = agg(runs, "eval_leader_return", how="tail") if runs else None
            if not m or met is None:
                continue
            xs.append(float(m.group(1))); ms.append(met[0]); ss.append(met[1])
        if not xs:
            continue
        _errbar(ax, xs, ms, ss, "Oracle", _C["Oracle"], "s")
        # RE-HG reference line at its default setting
        ref = None
        for d in [_baseline_dir(base)] + sorted(base.glob("exp*")):
            if d.is_dir():
                r = load_runs(d, "irl_recursive_rls", n_seeds)
                if r:
                    ref = agg(r, "eval_leader_return", how="tail")
                    break
        if ref:
            ax.axhline(ref[0], color=_C["RE-HG"], linestyle="--",
                       label="RE-HG (default settings)")
        ax.set_xscale("log"); ax.set_xlabel(f"Oracle {xname}")
        ax.set_ylabel("Leader return (eval)"); ax.grid(alpha=0.3); ax.legend(fontsize=10)
        ax.set_title(f"Exp 3 — Oracle {xname} sweep", fontsize=12)
        _save(fig, out / "exp3_oracle_lr.pdf")


def _plot_exp3_curves(base: Path, out: Path, n_seeds: int) -> None:
    """Learning curves behind the Exp 3 / 3b / 3c sweeps.

    Kept out of ``plot_exp3`` because that function returns early when no
    ``exp3_oracle_lr*`` directory exists, which would silently skip the curves
    whenever only 3b or 3c has been run.

    The tail means in the sweep plot collapse each arm to one number, which
    hides whether a slower Oracle is merely still climbing (a step-size effect)
    or has settled in a different basin (a direction effect). Only the latter
    supports the paper's claim, so the curves have to be shown alongside the
    endpoints.

    The three experiments answer different questions, so they get their own
    panels rather than one crowded axis:

        3   step size            -- can a slower Oracle catch up?
        3b  magnitude-only knobs -- normalization, and more episodes at low lr
        3c  influence scaling    -- the one control that does change direction
    """
    groups = [
        ("Exp 3 — Oracle step size",
         [(d, "softvi") for d in sorted(base.glob("exp3_oracle_lr*"))]),
        ("Exp 3b — normalization / episodes",
         [(d, "softvi") for d in sorted(base.glob("exp3b_oracle_*"))]
         + [(d, "irl_recursive_rls") for d in sorted(base.glob("exp3b_rehg_*"))]),
        ("Exp 3c — influence-term scale " + r"$\beta$",
         [(d, "softvi") for d in sorted(base.glob("exp3c_oracle_beta*"))]),
    ]
    groups = [(t, [(d, mo) for d, mo in items if d.is_dir()]) for t, items in groups]
    groups = [(t, items) for t, items in groups if items]
    if not groups:
        return

    # RE-HG at its default setting: the same reference line in every panel.
    ref_curve = None
    for d in [_baseline_dir(base)] + sorted(base.glob("exp*")):
        if not d.is_dir():
            continue
        arrs = [np.asarray(x["eval_leader_return"], dtype=np.float64)
                for x in load_runs(d, "irl_recursive_rls", n_seeds)
                if x.get("eval_leader_return")]
        if arrs:
            n = min(len(a) for a in arrs)
            ref_curve = np.stack([a[:n] for a in arrs]).mean(0)
            break

    fig, axes = plt.subplots(1, len(groups), figsize=(5.6 * len(groups), 4.6),
                             squeeze=False, sharey=True, layout="constrained")
    cmap = plt.get_cmap("viridis")
    for gi, (title, items) in enumerate(groups):
        ax = axes[0][gi]
        for i, (d, mode) in enumerate(items):
            arrs = [np.asarray(r["eval_leader_return"], dtype=np.float64)
                    for r in load_runs(d, mode, n_seeds)
                    if r.get("eval_leader_return")]
            if not arrs:
                continue
            n = min(len(a) for a in arrs)
            A = np.stack([a[:n] for a in arrs])
            m, s = A.mean(0), A.std(0)
            col = cmap(i / max(len(items) - 1, 1))
            lab = re.sub(r"^exp3[abc]?_(oracle|rehg)_?", "", d.name) or d.name
            ax.plot(m, label=lab, color=col, linewidth=1.8)
            ax.fill_between(np.arange(n), m - s, m + s, color=col, alpha=0.12)
        if ref_curve is not None:
            ax.plot(ref_curve, label="RE-HG (default)", color=_C["RE-HG"],
                    linestyle="--", linewidth=2.6)
        ax.set_xlabel("leader iteration")
        if gi == 0:
            ax.set_ylabel("Leader return (eval)")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Learning curves behind the Exp 3 sweeps", y=1.03)
    _save(fig, out / "exp3_curves.pdf")


# --------------------------------------------------- Exp 4: epsilon sweep ----

def plot_exp4(base: Path, out: Path, n_seeds: int) -> None:
    dirs = sorted(base.glob("exp4_eps*"))
    if not dirs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False,
                                 layout="constrained")
    xs, perf, perf_s, cos, cos_s, rank, rank_s = [], [], [], [], [], [], []
    for d in dirs:
        m = re.search(r"eps([0-9.eE+-]+)$", d.name)
        runs = load_runs(d, "irl_recursive_rls", n_seeds)
        if not m or not runs:
            continue
        p = agg(runs, "eval_leader_return", how="tail")
        c = agg(runs, "hg_cosine_similarity", how="tail")
        r = agg(runs, "Uk_rank", how="tail")
        if p is None:
            continue
        xs.append(float(m.group(1)))
        perf.append(p[0]); perf_s.append(p[1])
        cos.append(c[0] if c else np.nan); cos_s.append(c[1] if c else 0.0)
        rank.append(r[0] if r else np.nan); rank_s.append(r[1] if r else 0.0)
    if not xs:
        return
    for ax, y, s, lab in ((axes[0][0], perf, perf_s, "Leader return (eval)"),
                          (axes[0][1], cos, cos_s, "HG cosine vs Oracle"),
                          (axes[0][2], rank, rank_s, r"surviving rank of $U_k$")):
        _errbar(ax, xs, y, s, "RE-HG", _C["RE-HG"])
        if lab.startswith("Leader"):
            ax.set_title(r"Exp 4 — truncation threshold $\epsilon$", fontsize=12)
        ax.set_xscale("log"); ax.set_xlabel(r"truncation threshold $\epsilon$")
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    _save(fig, out / "exp4_eps.pdf")

    # rank(bar-Sigma) monotonicity: answers "does identification break once the
    # leader converges?"  (bar-Sigma is a cumulative PSD sum, so rank cannot fall)
    fig, ax = plt.subplots(figsize=(6, 4.2), layout="constrained")
    for d in dirs:
        runs = load_runs(d, "irl_recursive_rls", n_seeds)
        arrs = [np.asarray(r["barU_rank"], dtype=np.float64)
                for r in runs if r.get("barU_rank")]
        if not arrs:
            continue
        n = min(len(a) for a in arrs)
        ax.plot(np.stack([a[:n] for a in arrs]).mean(0), label=d.name.split("_")[-1])
    ax.set_xlabel("leader iteration"); ax.set_ylabel(r"rank($\bar\Sigma_K$)")
    ax.set_title("rank is met early, never lost", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    _save(fig, out / "exp4_rank.pdf")


# ------------------------------------------------- Exp 5: joint pooled IRL ---

def plot_exp5(base: Path, out: Path, n_seeds: int) -> None:
    d = base / "exp5_joint_irl"
    joint = load_runs(d, "irl_joint_pooled", n_seeds) if d.exists() else []
    rehg = load_runs(d, "irl_recursive_rls", n_seeds)
    if not rehg:
        for c in [_baseline_dir(base)] + sorted(base.glob("exp*")):
            if c.is_dir():
                rehg = load_runs(c, "irl_recursive_rls", n_seeds)
                if rehg:
                    break
    if not joint:
        return

    # Regularized arms (ridge / early stopping). Reporting the baseline at its
    # BEST setting is what separates "pooled vs sequential" from "regularized
    # vs not" -- an unregularized baseline alone differs in two ways at once.
    reg_arms = []
    for pat, tag in (("exp5_joint_ridge*", "ridge"),
                     ("exp5_joint_earlystop*", "early-stop")):
        for c in sorted(base.glob(pat)):
            runs = load_runs(c, "irl_joint_pooled", n_seeds)
            if runs:
                reg_arms.append((f"{tag} " + c.name.split("_")[-1].replace(tag.replace("-",""), "").replace("ridge","").replace("earlystop",""), runs))
    best_reg = None
    scored = [(agg(r, "eval_leader_return", how="tail"), lab, r)
              for lab, r in reg_arms]
    scored = [s for s in scored if s[0] is not None]
    if scored:
        b = max(scored, key=lambda t: t[0][0])
        best_reg = (f"Joint-IRL + {b[1]} (best)", b[2])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False,
                             layout="constrained")
    ax = axes[0][0]
    series = [(rehg, "RE-HG", _C["RE-HG"]),
              (joint, "Joint-IRL (no reg.)", _C["Joint-IRL"])]
    if best_reg:
        series.append((best_reg[1], best_reg[0], _C["contrast"]))
    for runs, lab, col in series:
        arrs = [np.asarray(r["eval_leader_return"], dtype=np.float64)
                for r in runs if r.get("eval_leader_return")]
        if not arrs:
            continue
        n = min(len(a) for a in arrs)
        A = np.stack([a[:n] for a in arrs])
        m, s = A.mean(0), A.std(0)
        ax.plot(m, label=lab, color=col,
                linestyle="--" if lab == "RE-HG" else "-",
                linewidth=2.6 if lab == "RE-HG" else 1.8)
        ax.fill_between(np.arange(n), m - s, m + s, alpha=0.15, color=col)
    ax.set_xlabel("leader iteration"); ax.set_ylabel("Leader return (eval)")
    ax.set_title("Exp 5 — same identification limit", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)

    ax = axes[0][1]
    for runs, key, lab, col in ((joint, "joint_irl_solve_time",
                                 "Joint-IRL re-solve", _C["Joint-IRL"]),
                                (rehg, "rls_update_time",
                                 "RE-HG aggregation", _C["RE-HG"])):
        arrs = [np.cumsum(np.asarray(r[key], dtype=np.float64))
                for r in runs if r.get(key)]
        if not arrs:
            continue
        n = min(len(a) for a in arrs)
        ax.plot(np.stack([a[:n] for a in arrs]).mean(0), label=lab, color=col,
                    linestyle="--" if lab == "RE-HG" else "-",
                    linewidth=2.6 if lab == "RE-HG" else 1.8)
    ax.set_yscale("log"); ax.set_xlabel("leader iteration")
    ax.set_ylabel("cumulative time [s]")
    ax.set_title("different cost", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)
    _save(fig, out / "exp5_joint_cost.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/internal/rebuttal",
                    help="rebuttal output directory to scan")
    ap.add_argument("--out", default=None, help="where to write figures "
                                                "(default: <dir>/figures)")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="maximum seed index to look for")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.exists():
        raise SystemExit(f"{base} does not exist")
    out = Path(args.out) if args.out else base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(_STYLE)

    print(f"scanning {base}")
    for fn in (plot_exp1, plot_exp2, plot_exp2c, plot_exp3, plot_exp4, plot_exp5):
        try:
            fn(base, out, args.n_seeds)
        except Exception as exc:      # one missing experiment must not stop the rest
            print(f"  [{fn.__name__}] skipped: {exc}")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()