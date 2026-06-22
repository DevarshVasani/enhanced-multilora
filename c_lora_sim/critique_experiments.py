"""
Critique-response experiments for RL-BSBF.

Runs the three eval-only studies that answer reviewer critiques 1, 2 and 4
(critique 3, multi-training-seed variance, is handled by seed_study/ +
seed_aggregate.py because it needs retraining). Everything here reuses the SAME
trained checkpoint and the SAME data-plane physics as the headline evaluation.

    python -m c_lora_sim.critique_experiments oracle        # Critique 1
    python -m c_lora_sim.critique_experiments jitter        # Critique 2
    python -m c_lora_sim.critique_experiments extrapolation # Critique 4
    python -m c_lora_sim.critique_experiments all

Results are written as JSON under c_lora_sim/results/critiques/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from c_lora_sim import calibration as cal
from c_lora_sim.baselines import fixed_n
from c_lora_sim.control_plane import CandidateGenerator, PendingJobQueue
from c_lora_sim.data_plane import CLoraDataPlane
from c_lora_sim.evaluate_clora import load_policy
from c_lora_sim.oracles import ORACLES, oracle_fixed_n, oracle_adaptive_hindsight
from c_lora_sim.runner import run_heuristic
from c_lora_sim.workload import generate_workload

DEVICE = torch.device("cpu")
RESULT_DIR = Path("c_lora_sim/results/critiques")

# Load sweep shared across experiments (matches evaluate_clora.py).
LOADS = [
    ("underloaded",       40, 0.50, False),
    ("light",             50, 0.75, False),
    ("critical",          60, 0.90, False),
    ("critical_bursty",   60, 0.90, True),
    ("overloaded",        70, 1.50, False),
    ("overloaded_bursty", 70, 1.50, True),
]


def ci95(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(len(xs))


def stat(xs: List[float]) -> Dict[str, float]:
    return {"mean": float(np.mean(xs)), "ci": ci95(xs), "std": float(np.std(xs)) if xs else 0.0}


# =========================================================================
# RL rollout that also returns the realized co-location depth distribution.
# =========================================================================
def run_rl_instrumented(policy, jobs, num_gpus=8, **noise):
    """Run the deterministic RL policy and record the resulting co-location depth
    (n_after) of every placement, so we can measure how often it packs past N=8."""
    import copy
    dp = CLoraDataPlane(num_gpus=num_gpus, **noise)
    dp.reset(copy.deepcopy(jobs))
    queue = PendingJobQueue(dp)
    depths: List[int] = []
    steps = 0
    while not dp.done() and steps < 50_000:
        cand, idx, _, _, cands, _ = policy.act(dp, queue, deterministic=True)
        if not cand:
            kind, _ = dp.advance()
            steps += 1
            if kind == "idle" and not dp.pending:
                break
            continue
        if cand.mode == "wait":
            kind, _ = dp.advance()
            if kind == "idle" and not dp.pending:
                break
        elif cand.mode == "migrate":
            job = next(j for j in dp.running if j.job_id == cand.job_id)
            dp.migrate(job, cand.gpu_id)
        else:
            job = next(j for j in dp.pending if j.job_id == cand.job_id)
            dp.place(job, cand.gpu_id)
            depths.append(cand.n_after)
        steps += 1
    return dp.metrics(), depths


# =========================================================================
# Critique 1 — oracle strength ladder.
# =========================================================================
def exp_oracle(seeds: List[int], num_gpus: int, policy) -> Dict:
    """Compare RL against the oracle LADDER (static fixed-N -> best-of-library ->
    clairvoyant adaptive). Honest claim: RL should BEAT the static fixed-N oracle
    and land within a few % of the adaptive oracle."""
    out: Dict[str, Dict] = {}
    for label, nj, rho, bursty in LOADS:
        rl_jcts: List[float] = []
        oracle_jcts: Dict[str, List[float]] = {k: [] for k in ORACLES}
        for seed in seeds:
            jobs = generate_workload(num_jobs=nj, seed=seed, target_rho=rho, bursty=bursty)
            m, _ = run_rl_instrumented(policy, jobs, num_gpus=num_gpus)
            rl_jcts.append(m["mean_jct"])
            for k, fn in ORACLES.items():
                oracle_jcts[k].append(fn(jobs, num_gpus=num_gpus)["mean_jct"])
        row = {"rho": rho, "bursty": bursty, "RL": stat(rl_jcts)}
        for k in ORACLES:
            o = stat(oracle_jcts[k])
            # signed margin: + => RL better (lower JCT) than this oracle
            margins = [100.0 * (oj - rj) / oj for oj, rj in zip(oracle_jcts[k], rl_jcts)]
            row[k] = {**o, "rl_margin_pct": stat(margins)}
        out[label] = row
        print(f"[oracle] {label:<18} RL={row['RL']['mean']:.0f}  "
              + "  ".join(f"{k.split('-')[1]}={row[k]['mean']:.0f}"
                          f"({row[k]['rl_margin_pct']['mean']:+.1f}%)" for k in ORACLES))
    return out


# =========================================================================
# Critique 2 — real-world jitter sensitivity.
# =========================================================================
# The policy PLANS against nominal physics (features use cal.* unchanged) but the
# REALIZED dynamics are perturbed by step-time and cold-start coefficients of
# variation. A policy that only works on noiseless physics will lose its edge.
JITTER_LEVELS = [
    ("noiseless", 0.00, 0.00),
    ("mild",      0.10, 0.10),
    ("realistic", 0.15, 0.20),   # the critique's "+-15%" step + heavier NAS/PCIe cold jitter
    ("harsh",     0.25, 0.35),
]


def exp_jitter(seeds: List[int], noise_reps: int, num_gpus: int, policy) -> Dict:
    competitors = {"Fixed-N2": fixed_n(2), "Fixed-N8": fixed_n(8)}
    out: Dict[str, Dict] = {}
    for label, nj, rho, bursty in LOADS:
        out[label] = {"rho": rho, "bursty": bursty, "levels": {}}
        for lvl, scv, ccv in JITTER_LEVELS:
            rl_j, comp_j = [], {c: [] for c in competitors}
            orac_j = []
            for seed in seeds:
                jobs = generate_workload(num_jobs=nj, seed=seed, target_rho=rho, bursty=bursty)
                for r in range(noise_reps):
                    nz = dict(step_time_cv=scv, cold_start_cv=ccv,
                              noise_seed=10_000 * r + seed)
                    m, _ = run_rl_instrumented(policy, jobs, num_gpus=num_gpus, **nz)
                    rl_j.append(m["mean_jct"])
                    for cname, csched in competitors.items():
                        comp_j[cname].append(
                            run_heuristic(jobs, csched, num_gpus=num_gpus, **nz)["mean_jct"])
                    orac_j.append(oracle_fixed_n(jobs, num_gpus=num_gpus, **nz)["mean_jct"])
                    if scv == 0.0 and ccv == 0.0:
                        break  # noiseless is deterministic; one rep suffices
            entry = {"RL": stat(rl_j), "ORACLE-fixedN": stat(orac_j)}
            for cname in competitors:
                margins = [100.0 * (cj - rj) / cj for cj, rj in zip(comp_j[cname], rl_j)]
                entry[cname] = {**stat(comp_j[cname]), "rl_margin_pct": stat(margins)}
            omarg = [100.0 * (oj - rj) / oj for oj, rj in zip(orac_j, rl_j)]
            entry["ORACLE-fixedN"]["rl_margin_pct"] = stat(omarg)
            out[label]["levels"][lvl] = entry
            print(f"[jitter] {label:<18} {lvl:<10} "
                  f"RL={entry['RL']['mean']:.0f}  "
                  f"vsN2={entry['Fixed-N2']['rl_margin_pct']['mean']:+.1f}%  "
                  f"vsN8={entry['Fixed-N8']['rl_margin_pct']['mean']:+.1f}%  "
                  f"vsOracle={entry['ORACLE-fixedN']['rl_margin_pct']['mean']:+.1f}%")
    return out


# =========================================================================
# Critique 4 — N>8 extrapolation robustness + reward-hacking check.
# =========================================================================
EXTRAP_VARIANTS = [
    ("linear",   dict(slope_mult=1.0)),   # flattest credible continuation
    ("default",  dict(slope_mult=1.6)),   # the paper's super-linear default
    ("harsh",    dict(slope_mult=3.0)),   # steep bend-up
    ("hardcap8", dict(hard_cap=8)),       # N>8 removed from the game entirely
]


def exp_extrapolation(seeds: List[int], num_gpus: int, policy) -> Dict:
    """Re-evaluate the SAME trained policy under different N>8 penalty shapes. If
    the RL-vs-oracle margin is stable AND the policy rarely packs past 8, the
    headline result does not depend on the invented extrapolation curve, refuting
    the reward-hacking concern."""
    out: Dict[str, Dict] = {}
    for label, nj, rho, bursty in LOADS:
        out[label] = {"rho": rho, "bursty": bursty, "variants": {}}
        for vname, kw in EXTRAP_VARIANTS:
            rl_j, orac_j = [], []
            frac_gt8: List[float] = []   # fraction of placements landing at N>8
            max_depths: List[int] = []
            with cal.extrapolation(**kw):
                for seed in seeds:
                    jobs = generate_workload(num_jobs=nj, seed=seed, target_rho=rho, bursty=bursty)
                    m, depths = run_rl_instrumented(policy, jobs, num_gpus=num_gpus)
                    rl_j.append(m["mean_jct"])
                    orac_j.append(oracle_adaptive_hindsight(jobs, num_gpus=num_gpus)["mean_jct"])
                    if depths:
                        frac_gt8.append(sum(1 for d in depths if d > 8) / len(depths))
                        max_depths.append(max(depths))
            margins = [100.0 * (oj - rj) / oj for oj, rj in zip(orac_j, rl_j)]
            out[label]["variants"][vname] = {
                "RL": stat(rl_j),
                "ORACLE-adaptive": stat(orac_j),
                "rl_margin_pct_vs_adaptive": stat(margins),
                "frac_placements_N_gt_8": stat(frac_gt8),
                "max_depth": stat([float(d) for d in max_depths]),
            }
            e = out[label]["variants"][vname]
            print(f"[extrap] {label:<18} {vname:<9} "
                  f"RL={e['RL']['mean']:.0f}  "
                  f"vsAdaptive={e['rl_margin_pct_vs_adaptive']['mean']:+.1f}%  "
                  f"N>8={100*e['frac_placements_N_gt_8']['mean']:.1f}%  "
                  f"maxN={e['max_depth']['mean']:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["oracle", "jitter", "extrapolation", "all"])
    ap.add_argument("--model", default="c_lora_sim/models/best.pt")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--noise-reps", type=int, default=4)
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.model, DEVICE)
    seeds = [20_000 + i for i in range(args.seeds)]

    todo = ["oracle", "jitter", "extrapolation"] if args.experiment == "all" else [args.experiment]
    for exp in todo:
        if exp == "oracle":
            res = exp_oracle(seeds, args.num_gpus, policy)
        elif exp == "jitter":
            res = exp_jitter(seeds, args.noise_reps, args.num_gpus, policy)
        else:
            res = exp_extrapolation(seeds, args.num_gpus, policy)
        path = RESULT_DIR / f"{exp}.json"
        with path.open("w") as f:
            json.dump(res, f, indent=2)
        print(f"  -> wrote {path}")


if __name__ == "__main__":
    main()
