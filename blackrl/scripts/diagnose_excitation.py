"""Which reward directions are identifiable, and in what order do they vanish?

Diagnostic for the under-specification / truncation discussion in the rebuttal.
For the actual Building Thermal Control system it reports, per state-cost
feature s_i^2:

  * the true cost Q_ii,
  * whether zone i has direct HVAC actuation (controllability),
  * the discounted second moment E[s_i^2] under the follower's closed loop,
  * the resulting feature excitation Var[s_i^2] = 2 (E[s_i^2])^2,

and then the rank correlations between cost / controllability and excitation.

Why this matters: the eigenvalue truncation removes the least-excited
directions first, so the excitation ordering *is* the order in which reward
components become unidentifiable. A natural guess is that expensive components
vanish first (tight control suppresses their variance). The measurement shows
the dominant factor is instead *controllability*: zones the follower can
actuate directly are regulated tightly and lose excitation, while an expensive
but unactuated zone stays well excited.

Usage:
    uv run python scripts/diagnose_excitation.py [--theta 0.75] [--json out.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from blackrl.algos.lqr_analytical import solve_confmdp_lqr
from scripts.run_conf_continuous import (
    _ENV_INIT_KWARGS,
    PARAMS,
    make_env,
)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(theta_val: float = 0.75) -> dict:
    env = make_env()
    n_s = env.num_zones
    gamma_F = PARAMS["discount_follower"]
    beta = PARAMS["beta"]

    theta = np.full(2 * n_s, float(theta_val))
    il, aa = theta[:n_s], theta[n_s:]
    A_theta = env.compute_parameterized_A(il, aa)

    sol = solve_confmdp_lqr(A_theta, env.B, env.Q, env.R, gamma_F)
    Ks = sol["Ks"]
    A_cl = A_theta - env.B @ Ks
    Cov_b = (beta / 2.0) * sol["S_inv"]

    Q_noise = env.B @ Cov_b @ env.B.T + (env.transition_std ** 2) * np.eye(n_s)
    Q_lyap = (env.init_temp_std ** 2) * np.eye(n_s) + (gamma_F / (1.0 - gamma_F)) * Q_noise
    M = solve_discrete_lyapunov(np.sqrt(gamma_F) * A_cl, Q_lyap)

    second_moment = np.diag(M).copy()
    excitation = 2.0 * second_moment ** 2  # Var[s_i^2] for zero-mean Gaussian
    cost = np.diag(env.Q).copy()
    controllable = np.array(
        [1 if np.any(env.B[i] != 0) else 0 for i in range(n_s)], dtype=int
    )

    # Open-loop comparison: excitation if the follower did NOT regulate at all.
    M_open = solve_discrete_lyapunov(np.sqrt(gamma_F) * A_theta, Q_lyap)
    excitation_open = 2.0 * np.diag(M_open) ** 2

    out = {
        "theta": float(theta_val),
        "cost": cost.tolist(),
        "controllable": controllable.tolist(),
        "second_moment": second_moment.tolist(),
        "excitation": excitation.tolist(),
        "excitation_open_loop": excitation_open.tolist(),
        "order_by_excitation_ascending": np.argsort(excitation).tolist(),
        "order_by_cost_descending": np.argsort(-cost).tolist(),
        "spearman_cost_vs_excitation": _spearman(cost, excitation),
        "spearman_controllable_vs_excitation": _spearman(controllable, excitation),
    }

    idx_c = np.where(controllable == 1)[0]
    idx_u = np.where(controllable == 0)[0]
    out["mean_excitation_controllable"] = float(excitation[idx_c].mean())
    out["mean_excitation_uncontrollable"] = float(excitation[idx_u].mean())
    if len(idx_c) > 1:
        out["spearman_cost_vs_excitation_controllable_only"] = _spearman(
            cost[idx_c], excitation[idx_c]
        )

    # Suppression ratio: how much the closed loop kills each feature.
    out["suppression_ratio"] = (excitation / np.maximum(excitation_open, 1e-12)).tolist()
    return out


def report(res: dict) -> None:
    cost = np.array(res["cost"])
    ctrl = np.array(res["controllable"])
    m2 = np.array(res["second_moment"])
    exc = np.array(res["excitation"])
    sup = np.array(res["suppression_ratio"])

    print(f"Building Thermal Control, theta = {res['theta']} (uniform)")
    print()
    print("zone  Q_ii   HVAC   E[s^2]      Var[s^2]      closed/open")
    for i in range(len(cost)):
        print(
            f" {i:3d}  {cost[i]:5.1f}    {ctrl[i]:d}    {m2[i]:9.3f}  {exc[i]:12.4e}   {sup[i]:9.4f}"
        )
    print()
    print(f"least-excited first (truncation order): {res['order_by_excitation_ascending']}")
    print(f"most-expensive first                  : {res['order_by_cost_descending']}")
    print()
    print(f"Spearman(cost, excitation)          = {res['spearman_cost_vs_excitation']:+.3f}")
    print(f"Spearman(controllable, excitation)  = {res['spearman_controllable_vs_excitation']:+.3f}")
    if "spearman_cost_vs_excitation_controllable_only" in res:
        print(
            "Spearman(cost, excitation | controllable only) = "
            f"{res['spearman_cost_vs_excitation_controllable_only']:+.3f}"
        )
    print()
    print(f"mean excitation, HVAC-actuated zones   = {res['mean_excitation_controllable']:.4e}")
    print(f"mean excitation, unactuated zones      = {res['mean_excitation_uncontrollable']:.4e}")
    print()
    print(
        "Reading: excitation is governed by controllability, not by cost magnitude.\n"
        "Directly actuated zones are regulated tightly and lose excitation, so their\n"
        "cost coefficients are the first to become unidentifiable under truncation;\n"
        "the expensive but unactuated zone stays well excited, which is why the\n"
        "hypergradient direction survives the filtering."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, default=0.75,
                    help="uniform leader configuration to evaluate at")
    ap.add_argument("--json", type=str, default=None,
                    help="also write the raw numbers to this JSON file")
    args = ap.parse_args()

    res = analyze(args.theta)
    report(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved raw numbers -> {args.json}")


if __name__ == "__main__":
    main()