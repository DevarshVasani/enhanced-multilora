"""
Generalization + literature baselines (reviewer Reasons 4 & 5).

Protocol: TUNE every controller's parameters on rho=0.9, then EVALUATE the frozen
config on UNSEEN rho=1.5. A small held-out gap => generalization, not overfitting.
Baselines include AIMD (TCP) and threshold-hysteresis (HPA), the standard feedback
solutions, each tuned by the same protocol so the comparison is fair.
"""

from __future__ import annotations

import itertools
import numpy as np

from c_lora_sim.workload import generate_workload
from c_lora_sim.runner import run_heuristic
from c_lora_sim.baselines import fixed_n, load_adaptive
from c_lora_sim.n_controller import lambda_controller
from c_lora_sim.n_baselines import aimd_controller, threshold_hysteresis
from c_lora_sim.oracles import oracle_adaptive_hindsight

TUNE = dict(num_jobs=60, rho=0.90)      # tuning load
TEST = dict(num_jobs=70, rho=1.50)      # unseen eval load
SEEDS = [20000, 20001, 20002, 20003, 20004]


def _jct(make_ctrl, spec):
    return float(np.mean([
        run_heuristic(generate_workload(num_jobs=spec["num_jobs"], seed=s,
                                        target_rho=spec["rho"], bursty=False),
                      make_ctrl(), num_gpus=8)["mean_jct"]
        for s in SEEDS]))


def _tune(name, factories):
    """Return (best_label, best_factory, jct_on_tune) minimising JCT on TUNE."""
    best = None
    for label, fac in factories:
        j = _jct(fac, TUNE)
        if best is None or j < best[2]:
            best = (label, fac, j)
    print(f"  tuned {name} on rho={TUNE['rho']}: {best[0]} (jct={best[2]:.0f})")
    return best


def main():
    print("Tuning on rho=0.9, evaluating on UNSEEN rho=1.5\n")

    lam_fac = [(f"lambda={l}", (lambda l=l: lambda_controller(l)))
               for l in [0.0, 0.25, 0.5, 0.75, 1.0]]
    aimd_fac = [(f"AIMD(ai={ai},md={md},hi={hi},lo={lo})",
                 (lambda ai=ai, md=md, hi=hi, lo=lo: aimd_controller(ai=ai, md=md, hi=hi, lo=lo)))
                for ai, md, hi, lo in itertools.product([1.0, 2.0], [0.5, 0.7],
                                                        [0.5, 1.0], [0.25, 0.5])]
    thr_fac = [(f"THR(up={up},down={dn},step={st})",
                (lambda up=up, dn=dn, st=st: threshold_hysteresis(up=up, down=dn, step=st)))
               for up, dn, st in itertools.product([0.75, 1.0], [0.25, 0.5], [1, 2, 4])]

    lam = _tune("lambda-controller", lam_fac)
    aimd = _tune("AIMD", aimd_fac)
    thr = _tune("threshold-hysteresis", thr_fac)

    # oracle + static baselines on the UNSEEN test load
    oracle = float(np.mean([oracle_adaptive_hindsight(
        generate_workload(num_jobs=TEST["num_jobs"], seed=s, target_rho=TEST["rho"],
                          bursty=False), num_gpus=8)["mean_jct"] for s in SEEDS]))
    fixedN = {f"Fixed-N{k}": _jct((lambda k=k: fixed_n(k)), TEST) for k in [2, 4, 8]}
    la = _jct((lambda: load_adaptive), TEST)

    print(f"\nEVALUATION on unseen rho={TEST['rho']} ({len(SEEDS)} seeds):")
    print(f"{'scheduler (tuned on 0.9)':<32}{'jct@1.5':>10}{'gap to oracle':>16}")
    rows = [(f"Fixed-N{k}", fixedN[f'Fixed-N{k}']) for k in [2, 4, 8]]
    rows.append(("LoadAdaptive", la))
    rows.append((f"AIMD [{aimd[0]}]", _jct(aimd[1], TEST)))
    rows.append((f"ThresholdHyst [{thr[0]}]", _jct(thr[1], TEST)))
    rows.append((f"lambda-controller [{lam[0]}]", _jct(lam[1], TEST)))
    for name, j in rows:
        print(f"{name:<32}{j:>10.0f}{(j-oracle)/oracle*100:>15.1f}%")
    print(f"{'ORACLE-asymmetric':<32}{oracle:>10.0f}{0.0:>15.1f}%")

    lam_gap = (_jct(lam[1], TEST) - oracle) / oracle * 100
    aimd_gap = (_jct(aimd[1], TEST) - oracle) / oracle * 100
    print(f"\nGENERALIZATION: lambda-controller tuned on 0.9 -> {lam_gap:.1f}% from oracle "
          f"on unseen 1.5  ({'GENERALIZES (<10%)' if lam_gap < 10 else 'TUNING RESULT (>=10%)'}).")
    print(f"vs AIMD (standard solution): {aimd_gap:.1f}% from oracle -> lambda-controller "
          f"{'BEATS' if lam_gap < aimd_gap else 'does NOT beat'} AIMD by {aimd_gap-lam_gap:.1f} pts.")

    import json, os
    os.makedirs("c_lora_sim/results", exist_ok=True)
    with open("c_lora_sim/results/baselines_generalization.json", "w") as f:
        json.dump({"tune": TUNE, "test": TEST, "seeds": SEEDS, "oracle": oracle,
                   "lambda": {"config": lam[0], "gap_pct": lam_gap},
                   "aimd": {"config": aimd[0], "gap_pct": aimd_gap},
                   "threshold": {"config": thr[0], "gap_pct": (_jct(thr[1], TEST)-oracle)/oracle*100},
                   "fixedN": fixedN, "load_adaptive": la}, f, indent=2)
    print("wrote c_lora_sim/results/baselines_generalization.json")


if __name__ == "__main__":
    main()
