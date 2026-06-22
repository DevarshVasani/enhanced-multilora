"""
20-seed 95% CI on the asymmetry (preemption) gap at rho=1.5.

Overload queue dynamics are chaotic, so the headline 36.6% needs a confidence
interval. We compute the gap PER SEED (paired: same workload, asym_SRPT vs
unconstrained, identical SRPT placement, only eviction differs) and report
mean +/- 95% CI over 20 seeds.
"""

from __future__ import annotations

import math
import numpy as np

from c_lora_sim.workload import generate_workload
from c_lora_sim.phase_gap_decompose import _best_over_grid, NO_PREEMPT

RHO, NUM_JOBS = 1.5, 70
SEEDS = list(range(20000, 20020))   # 20 seeds


def _boot_ci(fn, *arrays, n=10000):
    rng = np.random.RandomState(0)
    arrays = [np.asarray(a) for a in arrays]
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        vals.append(fn(*[a[idx] for a in arrays]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    asyms, uncs = [], []
    for s in SEEDS:
        jobs = generate_workload(num_jobs=NUM_JOBS, seed=s, target_rho=RHO, bursty=False)
        asym = _best_over_grid(jobs, NO_PREEMPT)        # SRPT placement, no eviction
        unc = min(_best_over_grid(jobs, 2.0), asym)     # + free preemption
        asyms.append(asym); uncs.append(unc)
        print(f"seed={s} asym={asym:.0f} unconstrained={unc:.0f} "
              f"gap={(asym-unc)/unc*100:+.1f}%")

    asyms = np.array(asyms); uncs = np.array(uncs)
    per_seed = (asyms - uncs) / uncs * 100.0

    # (1) aggregate ratio-of-means -- the operationally meaningful "fleet JCT saved"
    #     number, and the one the original 36.6% (5-seed) computed.
    agg = lambda a, u: (a.mean() - u.mean()) / u.mean() * 100
    agg_pt = agg(asyms, uncs)
    agg_lo, agg_hi = _boot_ci(agg, asyms, uncs)
    # (2) median per-seed gap -- robust to the heavy tail (typical workload).
    med_pt = float(np.median(per_seed))
    med_lo, med_hi = _boot_ci(lambda x: np.median(x), per_seed)
    # (3) mean per-seed gap -- pulled up by the tail.
    mean_pt = float(per_seed.mean())
    mean_lo, mean_hi = _boot_ci(lambda x: x.mean(), per_seed)

    print(f"\n=== preemption gap at rho={RHO}, {len(SEEDS)} seeds (95% bootstrap CI) ===")
    print(f"aggregate (ratio of mean JCT, = the old 36.6% statistic): "
          f"{agg_pt:.1f}%  CI [{agg_lo:.1f}%, {agg_hi:.1f}%]")
    print(f"median per-seed gap (typical workload):                 "
          f"{med_pt:.1f}%  CI [{med_lo:.1f}%, {med_hi:.1f}%]")
    print(f"mean per-seed gap (tail-weighted):                      "
          f"{mean_pt:.1f}%  CI [{mean_lo:.1f}%, {mean_hi:.1f}%]")
    print(f"distribution: min {per_seed.min():.0f}%  max {per_seed.max():.0f}%  "
          f"<5%: {(per_seed<5).sum()}/{len(SEEDS)} seeds  >150%: {(per_seed>=150).sum()}/{len(SEEDS)}")
    print("VERDICT: bimodal and workload-determined -- the value of free eviction is "
          "NOT a fixed fraction. Report the distribution, not a point estimate.")

    import json, os
    os.makedirs("c_lora_sim/results", exist_ok=True)
    with open("c_lora_sim/results/asymmetry_gap_ci.json", "w") as f:
        json.dump({"rho": RHO, "n_seeds": len(SEEDS),
                   "aggregate_ratio_of_means": {"pt": agg_pt, "ci": [agg_lo, agg_hi]},
                   "median_per_seed": {"pt": med_pt, "ci": [med_lo, med_hi]},
                   "mean_per_seed": {"pt": mean_pt, "ci": [mean_lo, mean_hi]},
                   "per_seed_gap": per_seed.tolist(),
                   "asym_jct": asyms.tolist(), "unc_jct": uncs.tolist()}, f, indent=2)
    print("wrote c_lora_sim/results/asymmetry_gap_ci.json")


if __name__ == "__main__":
    main()
