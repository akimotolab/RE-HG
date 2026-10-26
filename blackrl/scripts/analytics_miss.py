import pickle, numpy as np
for d in ["exp2b_lowspec_cost_sev0.02","exp2b_lowspec_cost_sev0.25","exp2b_lowspec_excitation_full"]:
    s = pickle.load(open(f"data/internal/rebuttal/{d}/stats_irl_recursive_rls_seed0.pkl","rb"))
    print(d)
    print("  最終 Q_hat:", np.round(s["reward_Q_diag"][-1], 3))
    print("  barU_rank :", s["barU_rank"][-1], " Uk_rank平均:", np.mean(s["Uk_rank"]))