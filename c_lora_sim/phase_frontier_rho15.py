"""
The completing experiment: lambda-controller Pareto frontier at rho=1.5 (overload),
where the asymmetry-gap result shows free eviction is worth ~37%.

Question: how close does the placement-only lambda-controller get to the ASYMMETRIC
(passive-drain) oracle -- the best achievable WITHOUT eviction -- and how does it
compare to Fixed-N baselines? The fraction of the (best-Fixed-N -> asymmetric-oracle)
gap that the controller closes tells us how much of the no-eviction headroom it
captures; the residual to the UNCONSTRAINED oracle is what only free eviction (or a
policy that learns to use it) can buy.

Run:  .venv/bin/python -m c_lora_sim.phase_frontier_rho15
"""

from __future__ import annotations

import numpy as np

from c_lora_sim.workload import generate_workload
from c_lora_sim.runner import run_heuristic
from c_lora_sim.baselines import fixed_n, load_adaptive
from c_lora_sim.n_controller import lambda_controller
from c_lora_sim.oracles import oracle_adaptive_hindsight
from c_lora_sim.phase_asymmetry_gap import oracle_unconstrained_free

RHO = 1.5
NUM_JOBS = 70
SEEDS = [20000, 20001, 20002, 20003, 20004]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def mean_over_seeds(fn):
    return float(np.mean([fn(s) for s in SEEDS]))


def jct_heuristic(sched):
    return mean_over_seeds(lambda s: run_heuristic(
        generate_workload(num_jobs=NUM_JOBS, seed=s, target_rho=RHO, bursty=False),
        sched, num_gpus=8)["mean_jct"])


def main():
    print(f"lambda-controller frontier vs baselines + oracle bracket at rho={RHO} "
          f"({NUM_JOBS} jobs, {len(SEEDS)} seeds)\n")

    # --- baselines -------------------------------------------------------
    fixed = {f"Fixed-N{k}": jct_heuristic(fixed_n(k)) for k in [1, 2, 4, 8]}
    la = jct_heuristic(load_adaptive)

    # --- lambda sweep ----------------------------------------------------
    lam_jct = {}
    for lam in LAMBDAS:
        lam_jct[lam] = mean_over_seeds(lambda s, L=lam: run_heuristic(
            generate_workload(num_jobs=NUM_JOBS, seed=s, target_rho=RHO, bursty=False),
            lambda_controller(L), num_gpus=8)["mean_jct"])
    best_lam = min(lam_jct, key=lam_jct.get)
    ctrl = lam_jct[best_lam]

    # --- oracle bracket --------------------------------------------------
    oracle_asym = mean_over_seeds(lambda s: oracle_adaptive_hindsight(
        generate_workload(num_jobs=NUM_JOBS, seed=s, target_rho=RHO, bursty=False),
        num_gpus=8)["mean_jct"])
    oracle_unc = mean_over_seeds(lambda s: oracle_unconstrained_free(
        generate_workload(num_jobs=NUM_JOBS, seed=s, target_rho=RHO, bursty=False),
        num_gpus=8)["mean_jct"])

    # --- table -----------------------------------------------------------
    print(f"{'scheduler':<28}{'mean_jct':>11}{'gap to asym oracle':>22}")
    for name, v in fixed.items():
        print(f"{name:<28}{v:>11.0f}{(v-oracle_asym)/oracle_asym*100:>20.1f}%")
    print(f"{'LoadAdaptive':<28}{la:>11.0f}{(la-oracle_asym)/oracle_asym*100:>20.1f}%")
    for lam in LAMBDAS:
        tag = f"lambda={lam:.2f}" + ("  <-best" if lam == best_lam else "")
        v = lam_jct[lam]
        print(f"{tag:<28}{v:>11.0f}{(v-oracle_asym)/oracle_asym*100:>20.1f}%")
    print(f"{'ORACLE-asymmetric (drain)':<28}{oracle_asym:>11.0f}{0.0:>20.1f}%")
    print(f"{'ORACLE-unconstrained (evict)':<28}{oracle_unc:>11.0f}"
          f"{(oracle_unc-oracle_asym)/oracle_asym*100:>20.1f}%")

    # --- the story numbers ----------------------------------------------
    best_fixed_name = min(fixed, key=fixed.get)
    best_fixed = fixed[best_fixed_name]
    # how much of the (best Fixed-N -> asymmetric oracle) headroom the controller closes
    denom = best_fixed - oracle_asym
    closed = (best_fixed - ctrl) / denom * 100 if denom > 0 else 100.0
    ctrl_gap = (ctrl - oracle_asym) / oracle_asym * 100
    evict_gap = (oracle_asym - oracle_unc) / oracle_unc * 100

    print("\n=== THE STORY (rho=1.5, overload) ===")
    print(f"1. Free eviction is worth {evict_gap:.0f}% here "
          f"(asymmetric {oracle_asym:.0f} -> unconstrained {oracle_unc:.0f}).")
    print(f"2. Best Fixed-N is {best_fixed_name} ({best_fixed:.0f}); the lambda-controller "
          f"(lambda={best_lam}) reaches {ctrl:.0f}.")
    print(f"3. The controller closes {closed:.0f}% of the (best Fixed-N -> asymmetric "
          f"oracle) headroom, landing {ctrl_gap:+.1f}% from the asymmetric oracle.")
    print(f"4. The residual below the asymmetric oracle ({evict_gap:.0f}% of JCT) is "
          f"reachable ONLY with free eviction -- and PPO does not close it (flat, "
          f"unstable landscape; see findings).")
    print(f"5. RECOMMENDATION: use the lambda-controller everywhere (captures "
          f"{closed:.0f}% of no-eviction headroom, no preemption); build free eviction "
          f"specifically for OVERSUBSCRIBED clusters, where it buys the extra {evict_gap:.0f}%.")

    import json, os
    os.makedirs("c_lora_sim/results", exist_ok=True)
    with open("c_lora_sim/results/frontier_rho15.json", "w") as f:
        json.dump({"rho": RHO, "seeds": SEEDS, "fixed": fixed, "load_adaptive": la,
                   "lambda": lam_jct, "best_lambda": best_lam,
                   "oracle_asymmetric": oracle_asym, "oracle_unconstrained": oracle_unc,
                   "fraction_headroom_closed_pct": closed,
                   "controller_gap_to_asym_pct": ctrl_gap,
                   "free_eviction_value_pct": evict_gap}, f, indent=2)
    print("\nwrote c_lora_sim/results/frontier_rho15.json")


if __name__ == "__main__":
    main()
