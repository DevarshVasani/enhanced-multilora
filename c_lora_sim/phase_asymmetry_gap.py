"""
THE asymmetry-gap measurement: what is faster (free) eviction worth?

Two clairvoyant oracles on identical workloads:
  * oracle_asymmetric    -- clairvoyant + PASSIVE DRAIN only (no eviction; N falls
                            only as adapters finish). Best achievable under the real
                            physical actuator. == oracle_adaptive_hindsight.
  * oracle_unconstrained -- clairvoyant + FREE EVICTION: may preempt any running
                            job instantly for free (progress preserved) and re-place
                            it later. Optimal use = preemptive SRPT under adaptive
                            co-location depth (rush the shortest-remaining jobs,
                            park longer ones). Best achievable if eviction were free.

    asymmetry_gap_pct = (mean_jct(asymmetric) - mean_jct(unconstrained))
                        / mean_jct(unconstrained)

This is the number Trajectory has not measured: if it is ~5%, faster eviction is
not worth engineering; if it is ~30%, it is their next infrastructure project.

Run:  .venv/bin/python -m c_lora_sim.phase_asymmetry_gap
"""

from __future__ import annotations

import copy
import math
from typing import List

import numpy as np

from c_lora_sim.data_plane import CLoraDataPlane, LoraJob
from c_lora_sim.control_plane import CandidateGenerator
from c_lora_sim.workload import generate_workload
from c_lora_sim.oracles import oracle_adaptive_hindsight

# Adaptive-depth grid, tuned per-episode with hindsight (same family as the
# passive-drain oracle, so the only difference measured is the free eviction).
_DIVISORS = [1.0, 2.0, 4.0]
_LO = [1, 2]
_HI = [4, 6, 8]


def _place_for_depth(sim: CLoraDataPlane, job: LoraJob, target: int):
    """Place `job` honouring base-model locality and the depth target: an
    under-target same-base GPU, else a free GPU, else (forced) the least-loaded
    same-base GPU."""
    cand = CandidateGenerator._candidate_gpus(sim, job)
    under = [g for g in cand["warm_same"] if sim.gpus[g].n < target]
    if under:
        return min(under, key=lambda g: sim.gpus[g].n)
    if cand["empty"]:
        return cand["empty"][0]
    if cand["warm_same"]:
        return min(cand["warm_same"], key=lambda g: sim.gpus[g].n)
    if cand["repurpose"]:
        return cand["repurpose"][0]
    return None


def _adaptive_place(sim, job, target):
    """Clairvoyant adaptive placement: under-target same-base GPU (shortest job to
    least-loaded), else free GPU, else (forced) least-loaded same-base/repurpose."""
    cand = CandidateGenerator._candidate_gpus(sim, job)
    under = [g for g in cand["warm_same"] if sim.gpus[g].n < target]
    if under:
        return min(under, key=lambda g: sim.gpus[g].n)
    if cand["empty"]:
        return cand["empty"][0]
    if cand["warm_same"]:
        return min(cand["warm_same"], key=lambda g: sim.gpus[g].n)
    if cand["repurpose"]:
        return cand["repurpose"][0]
    return None


def _run_free_preempt(jobs, divisor, lo, hi, num_gpus=8, max_steps=200_000,
                      ratio=2.0):
    """Clairvoyant adaptive scheduling WITH conservative free SRPT preemption.

    Placement is adaptive-depth, shortest-remaining first. Additionally, when no
    under-target slot is free, a pending job that is much shorter (>= `ratio`x) than
    the longest-remaining running job may FREE-evict it (progress preserved, no
    partial-step loss) and take its place -- i.e. a short experiment jumps the queue
    ahead of a long one at no eviction cost. Re-placement still pays the normal
    (locality) cold start, which is the only honest remaining cost of free eviction.
    Conservative + one-at-a-time to avoid thrash."""
    sim = CLoraDataPlane(num_gpus=num_gpus, clairvoyant=True)
    sim.reset(copy.deepcopy(jobs))
    steps = 0
    while not sim.done() and steps < max_steps:
        n_pending = len(sim.pending)
        target = max(lo, min(hi, math.ceil(n_pending / divisor)))

        # --- conservative free SRPT preemption -----------------------------
        if sim.pending and sim.running:
            n_free = sum(1 for g in sim.gpus.values() if g.is_empty())
            shortest_pending = min(sim.pending, key=lambda j: sim.est_remaining_steps(j))
            longest_running = max(sim.running, key=lambda j: sim.est_remaining_steps(j))
            sp = sim.est_remaining_steps(shortest_pending)
            lr = sim.est_remaining_steps(longest_running)
            # only when no free capacity AND the waiting job is much shorter
            if n_free == 0 and lr > ratio * max(sp, 1.0):
                sim.evict(longest_running, free=True)   # FREE: no partial-step loss

        # --- adaptive placement, shortest-remaining first ------------------
        placed = True
        while placed and sim.pending:
            placed = False
            for j in sorted(sim.pending, key=lambda j: sim.est_remaining_steps(j)):
                gid = _adaptive_place(sim, j, target)
                if gid is not None:
                    sim.place(j, gid)
                    placed = True
        kind, _ = sim.advance()
        steps += 1
        if kind == "idle":
            break
    return sim.metrics()


def oracle_unconstrained_free(jobs, num_gpus=8):
    """Best free-eviction policy over the grid, OR the passive-drain oracle if
    preemption never helps (so this is >= as good as asymmetric by construction;
    the gap measures the MARGINAL value of free eviction)."""
    best = oracle_adaptive_hindsight(jobs, num_gpus=num_gpus)  # fall-back: no eviction
    for d in _DIVISORS:
        for lo in _LO:
            for hi in _HI:
                if hi < lo:
                    continue
                m = _run_free_preempt(jobs, d, lo, hi, num_gpus=num_gpus)
                if m["mean_jct"] < best["mean_jct"]:
                    best = m
    return best


LOADS = [
    ("underloaded", 50, 0.50, False),
    ("critical",    60, 0.90, False),
    ("critical_bursty", 60, 0.90, True),
    ("overloaded",  70, 1.50, False),
]
SEEDS = [20000, 20001, 20002, 20003, 20004]


def main():
    print("Asymmetry gap = value of FREE eviction (clairvoyant): "
          "passive-drain oracle vs free-preemptive oracle\n")
    print(f"{'load':<18}{'asymmetric':>12}{'unconstrained':>15}{'gap':>10}{'gap%':>9}")
    all_gaps = []
    rows = []
    for label, nj, rho, bursty in LOADS:
        asy, unc = [], []
        for s in SEEDS:
            jobs = generate_workload(num_jobs=nj, seed=s, target_rho=rho, bursty=bursty)
            asy.append(oracle_adaptive_hindsight(jobs, num_gpus=8)["mean_jct"])
            unc.append(oracle_unconstrained_free(jobs, num_gpus=8)["mean_jct"])
        a, u = float(np.mean(asy)), float(np.mean(unc))
        gap = a - u
        gap_pct = gap / u * 100 if u > 0 else 0.0
        all_gaps.append(gap_pct)
        rows.append((label, a, u, gap, gap_pct))
        print(f"{label:<18}{a:>12.0f}{u:>15.0f}{gap:>10.0f}{gap_pct:>8.1f}%")

    overall = float(np.mean(all_gaps))
    print(f"\nMEAN asymmetry gap across loads = {overall:+.1f}%")
    print("Interpretation: this is what instant/free eviction buys over passive drain.")
    if overall < 8:
        print("  -> SMALL: faster eviction is not worth the engineering.")
    elif overall < 20:
        print("  -> MODERATE: worth considering for latency-sensitive tenants.")
    else:
        print("  -> LARGE: faster eviction is a high-value infrastructure project.")

    import json, os
    os.makedirs("c_lora_sim/results", exist_ok=True)
    with open("c_lora_sim/results/asymmetry_gap.json", "w") as f:
        json.dump({"mean_gap_pct": overall,
                   "by_load": [{"load": l, "asymmetric": a, "unconstrained": u,
                                "gap": g, "gap_pct": p} for l, a, u, g, p in rows],
                   "seeds": SEEDS}, f, indent=2)
    print("\nwrote c_lora_sim/results/asymmetry_gap.json")


if __name__ == "__main__":
    main()
