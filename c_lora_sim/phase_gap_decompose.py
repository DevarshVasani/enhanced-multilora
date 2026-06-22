"""
Decompose the asymmetry gap into (a) placement-ordering suboptimality and
(b) the TRUE cost of the passive-drain constraint (= preemption value).

Reviewer's objection: oracle_adaptive_hindsight (the asymmetric oracle in the
memo) places jobs in ARRIVAL order, so the gap to the free-eviction oracle
(which places shortest-remaining first) mixes 'SRPT placement' -- achievable
under passive drain, no eviction -- with actual preemption. We separate them.

  asym_fifo   : adaptive-depth, FIFO placement, NO eviction  (the old oracle)
  asym_srpt   : adaptive-depth, SRPT placement, NO eviction  (strong constrained
                oracle -- best passive-drain policy we can construct)
  unconstr    : adaptive-depth, SRPT placement, FREE eviction

  ordering_suboptimality = (asym_fifo - asym_srpt) / asym_srpt
  TRUE_constraint_cost   = (asym_srpt - unconstr) / unconstr   <- the honest number

Both asym_srpt and unconstr share identical placement, so their only difference
is eviction -> their gap is the pure preemption value, free of oracle-ordering
artefacts. asym_srpt is still only a STRONG constructed policy, not a proven
optimum, so TRUE_constraint_cost remains an upper bound -- but a much tighter one,
and we report how much the ordering fix moved it.

Run:  .venv/bin/python -m c_lora_sim.phase_gap_decompose
"""

from __future__ import annotations

import numpy as np

from c_lora_sim.workload import generate_workload
from c_lora_sim.oracles import oracle_adaptive_hindsight
from c_lora_sim.phase_asymmetry_gap import (
    _run_free_preempt, _DIVISORS, _LO, _HI,
)

SEEDS = [20000, 20001, 20002, 20003, 20004]
LOADS = [
    ("underloaded", 50, 0.50, False),
    ("critical", 60, 0.90, False),
    ("critical_bursty", 60, 0.90, True),
    ("overloaded", 70, 1.50, False),
]
NO_PREEMPT = 1e9   # ratio so high that _run_free_preempt never evicts


def _best_over_grid(jobs, ratio):
    best = None
    for d in _DIVISORS:
        for lo in _LO:
            for hi in _HI:
                if hi < lo:
                    continue
                m = _run_free_preempt(jobs, d, lo, hi, num_gpus=8, ratio=ratio)
                if best is None or m["mean_jct"] < best:
                    best = m["mean_jct"]
    return best


def main():
    print("Decomposing the asymmetry gap: placement ordering vs true preemption cost\n")
    print(f"{'regime':<17}{'asym_FIFO':>11}{'asym_SRPT':>11}{'unconstr':>10}"
          f"{'ordering%':>11}{'TRUE_cost%':>12}")
    for label, nj, rho, b in LOADS:
        ff, ss, uu = [], [], []
        for s in SEEDS:
            jobs = generate_workload(num_jobs=nj, seed=s, target_rho=rho, bursty=b)
            fifo = oracle_adaptive_hindsight(jobs, num_gpus=8)["mean_jct"]
            srpt = _best_over_grid(jobs, NO_PREEMPT)
            srpt = min(srpt, fifo)                 # strong asym >= both options
            unc = _best_over_grid(jobs, 2.0)
            unc = min(unc, srpt)                   # free eviction can fall back
            ff.append(fifo); ss.append(srpt); uu.append(unc)
        f, sp, u = np.mean(ff), np.mean(ss), np.mean(uu)
        ordering = (f - sp) / sp * 100 if sp > 0 else 0.0
        true_cost = (sp - u) / u * 100 if u > 0 else 0.0
        print(f"{label:<17}{f:>11.0f}{sp:>11.0f}{u:>10.0f}{ordering:>10.1f}%{true_cost:>11.1f}%")

    print("\nordering% = headroom the old FIFO asymmetric oracle left on the table "
          "(achievable WITHOUT eviction).")
    print("TRUE_cost% = pure cost of the passive-drain constraint (same SRPT placement "
          "both sides; only eviction differs).")


if __name__ == "__main__":
    main()
