"""
Multi-training-seed aggregation (Critique 3: single-seed fragility).

Loads every per-seed checkpoint produced by seed_study/run_seeds.sh, evaluates
each one across the full load sweep, and reports the RL-vs-oracle margin as a
distribution OVER TRAINING SEEDS (mean +- CI, min, max, and the fraction of seeds
that stay margin-positive). The reviewer's question -- "did this one checkpoint
get lucky, or does the pipeline reliably converge to a winning policy?" -- is
answered by whether the margin is positive for ALL seeds, not just the headline one.

    python -m c_lora_sim.seed_aggregate --seeds 42 1 2 3 4 --eval-seeds 8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from c_lora_sim.baselines import fixed_n
from c_lora_sim.evaluate_clora import load_policy, run_rl
from c_lora_sim.oracles import oracle_fixed_n, oracle_adaptive_hindsight
from c_lora_sim.runner import run_heuristic
from c_lora_sim.workload import generate_workload

DEVICE = torch.device("cpu")
RESULT_DIR = Path("c_lora_sim/results/critiques")

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


def eval_one_seed(policy, eval_seeds, num_gpus) -> Dict[str, Dict[str, float]]:
    """Return per-regime mean JCT for RL and the comparators, for ONE checkpoint."""
    per: Dict[str, Dict[str, float]] = {}
    for label, nj, rho, bursty in LOADS:
        rl, n2, ofx, oad = [], [], [], []
        for s in eval_seeds:
            jobs = generate_workload(num_jobs=nj, seed=s, target_rho=rho, bursty=bursty)
            rl.append(run_rl(policy, jobs, num_gpus, DEVICE)["mean_jct"])
            n2.append(run_heuristic(jobs, fixed_n(2), num_gpus=num_gpus)["mean_jct"])
            ofx.append(oracle_fixed_n(jobs, num_gpus=num_gpus)["mean_jct"])
            oad.append(oracle_adaptive_hindsight(jobs, num_gpus=num_gpus)["mean_jct"])
        rl_m, n2_m = float(np.mean(rl)), float(np.mean(n2))
        ofx_m, oad_m = float(np.mean(ofx)), float(np.mean(oad))
        per[label] = {
            "RL": rl_m,
            "vs_FixedN2_pct": 100.0 * (n2_m - rl_m) / n2_m,
            "vs_oracle_fixedN_pct": 100.0 * (ofx_m - rl_m) / ofx_m,
            "vs_oracle_adaptive_pct": 100.0 * (oad_m - rl_m) / oad_m,
        }
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--model-root", default="c_lora_sim/seed_study")
    args = ap.parse_args()

    eval_seeds = [20_000 + i for i in range(args.eval_seeds)]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed: Dict[int, Dict] = {}
    for seed in args.seeds:
        ckpt = Path(args.model_root) / f"models_s{seed}" / "best.pt"
        if not ckpt.exists():
            print(f"[skip] seed {seed}: no checkpoint at {ckpt}")
            continue
        policy = load_policy(str(ckpt), DEVICE)
        per_seed[seed] = eval_one_seed(policy, eval_seeds, args.num_gpus)
        # quick line: margin vs adaptive oracle averaged over regimes
        avg = np.mean([per_seed[seed][l[0]]["vs_oracle_adaptive_pct"] for l in LOADS])
        print(f"seed {seed:>3}: mean margin vs adaptive-oracle = {avg:+.2f}%  "
              + " ".join(f"{l[0][:4]}={per_seed[seed][l[0]]['vs_oracle_fixedN_pct']:+.1f}" for l in LOADS))

    # ---- aggregate across training seeds ----
    agg: Dict[str, Dict] = {}
    keys = ["vs_FixedN2_pct", "vs_oracle_fixedN_pct", "vs_oracle_adaptive_pct"]
    for label, *_ in LOADS:
        agg[label] = {}
        for k in keys:
            vals = [per_seed[s][label][k] for s in per_seed]
            agg[label][k] = {
                "mean": float(np.mean(vals)), "ci": ci95(vals),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "frac_positive": float(np.mean([v > 0 for v in vals])),
                "n_seeds": len(vals),
            }

    print("\n===== cross-training-seed margin (mean +/- CI [min,max], % positive) =====")
    for label, *_ in LOADS:
        a = agg[label]["vs_oracle_fixedN_pct"]
        b = agg[label]["vs_oracle_adaptive_pct"]
        print(f"{label:<18} vs fixedN-oracle {a['mean']:+.2f}+-{a['ci']:.2f} "
              f"[{a['min']:+.1f},{a['max']:+.1f}] {100*a['frac_positive']:.0f}%pos   "
              f"| vs adaptive {b['mean']:+.2f}+-{b['ci']:.2f} {100*b['frac_positive']:.0f}%pos")

    out = {"per_seed": per_seed, "aggregate": agg,
           "seeds": list(per_seed.keys()), "eval_seeds": eval_seeds}
    with (RESULT_DIR / "seed_study.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> wrote {RESULT_DIR / 'seed_study.json'}")


if __name__ == "__main__":
    main()
