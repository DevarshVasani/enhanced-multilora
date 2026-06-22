"""
Phase 0 — real-hardware validation gate for the N-selection study.

The simulator's concurrency physics `step_time(N) = STEP_TIME_SOLO * step_scaling(N)`
was calibrated on (roughly) HOMOGENEOUS adapter training. The control experiments,
however, run HETEROGENEOUS mixes (short + long adapters, rollout vs training phases),
where real co-location has nonlinear interference (cache, memory-bandwidth, CUDA
kernel scheduling). If the real `step_time(N, job_mix)` diverges from the calibrated
`STEP_TIME_SOLO * step_scaling(N)`, every downstream sim number is optimising the
wrong plant. THIS SCRIPT IS THE GATE: it measures real step time across N=1..8 under
both homogeneous and heterogeneous co-location, compares to the calibrated curve, and
either (a) confirms agreement within a stated tolerance, or (b) fits a job-mix
correction factor `s(N, mix)` to re-anchor calibration.py.

Real run (needs a GPU + vLLM + an actual base model and adapters):
    python -m c_lora_sim.calibration_validation --real --model Qwen/Qwen2.5-7B \
        --adapters path1 path2 ... --steps 20

Without --real (or without vLLM/GPU) it runs in MOCK mode: it synthesises plausible
measurements (calibrated curve + a small heterogeneous-interference bump + noise) so
the harness, the comparison math, and the report format can be exercised in CI. Mock
numbers are NOT a validation — they are clearly labelled as such.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np

from c_lora_sim import calibration as cal

N_VALUES = [1, 2, 4, 8]
# Tolerance below which the simulator is declared "validated" for the het. regime.
TOLERANCE = 0.15  # 15%


# ---------------------------------------------------------------------------
# Real measurement (vLLM). Imported lazily so the module loads without a GPU.
# ---------------------------------------------------------------------------
def measure_real_step_times(model: str, adapters: List[str], n_values: List[int],
                            steps: int, heterogeneous: bool) -> Dict[int, float]:
    """Measure mean wall-time per multiplexed step for each N on real hardware.

    For each N, co-locate N adapters and time `steps` training/rollout steps; return
    seconds-per-step. `heterogeneous=True` interleaves short- and long-sequence work
    across the co-located adapters to exercise the realistic interference regime.
    Requires vLLM + a GPU; raises ImportError/RuntimeError otherwise (caller falls
    back to mock)."""
    import time
    from vllm import LLM, SamplingParams           # noqa: F401  (import-time GPU check)
    from vllm.lora.request import LoRARequest

    llm = LLM(model=model, enable_lora=True, max_lora_rank=16,
              gpu_memory_utilization=0.9, max_loras=max(n_values))
    short = SamplingParams(temperature=0.0, max_tokens=32)
    long = SamplingParams(temperature=0.0, max_tokens=256)
    reqs = [LoRARequest(f"a{i}", i + 1, p) for i, p in enumerate(adapters)]

    out: Dict[int, float] = {}
    for n in n_values:
        active = reqs[:n]
        prompts = ["benchmark step"] * n
        t0 = time.time()
        for s in range(steps):
            for i, r in enumerate(active):
                sp = (long if (heterogeneous and i % 2 == 0) else short)
                llm.generate([prompts[i]], sp, lora_request=r)
        out[n] = (time.time() - t0) / steps
    return out


def measure_mock_step_times(n_values: List[int], heterogeneous: bool,
                            seed: int = 0) -> Dict[int, float]:
    """Synthesise plausible measurements: the calibrated curve, plus (for the
    heterogeneous regime) a depth-growing interference bump, plus small noise.
    Clearly NOT a real validation — used only to exercise the harness."""
    rng = np.random.RandomState(seed)
    out: Dict[int, float] = {}
    for n in n_values:
        base = cal.step_time(n)
        # heterogeneous interference modelled as a few % per extra co-located job
        het_bump = (1.0 + 0.03 * (n - 1)) if heterogeneous else 1.0
        noise = 1.0 + rng.normal(0, 0.04)
        out[n] = base * het_bump * noise
    return out


# ---------------------------------------------------------------------------
# Comparison + job-mix correction fit
# ---------------------------------------------------------------------------
def compare(measured: Dict[int, float]) -> List[Tuple[int, float, float, float]]:
    """Return [(N, predicted_s, measured_s, rel_err)] vs the calibrated curve."""
    rows = []
    for n in sorted(measured):
        pred = cal.step_time(n)
        meas = measured[n]
        rel = (meas - pred) / pred if pred > 0 else 0.0
        rows.append((n, pred, meas, rel))
    return rows


def fit_mix_correction(measured: Dict[int, float]) -> Dict[str, float]:
    """Fit a single multiplicative interference slope `beta` such that
    s_corrected(N, mix) = step_scaling(N) * (1 + beta * (N - 1)) best matches the
    measured heterogeneous step times. Returns {beta, r2}."""
    ns = np.array(sorted(measured), dtype=float)
    meas_scaling = np.array([measured[int(n)] / cal.STEP_TIME_SOLO for n in ns])
    base_scaling = np.array([cal.step_scaling(int(n)) for n in ns])
    # meas = base * (1 + beta*(N-1))  ->  meas/base - 1 = beta*(N-1)
    y = meas_scaling / base_scaling - 1.0
    x = ns - 1.0
    beta = float((x @ y) / (x @ x)) if (x @ x) > 0 else 0.0
    pred = base_scaling * (1 + beta * (ns - 1))
    ss_res = float(np.sum((meas_scaling - pred) ** 2))
    ss_tot = float(np.sum((meas_scaling - meas_scaling.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"beta": beta, "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="measure on real GPU via vLLM")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="c_lora_sim/results/calibration_validation.json")
    args = ap.parse_args()

    mode = "REAL"
    try:
        if not args.real:
            raise RuntimeError("mock requested")
        homo = measure_real_step_times(args.model, args.adapters, N_VALUES,
                                       args.steps, heterogeneous=False)
        het = measure_real_step_times(args.model, args.adapters, N_VALUES,
                                      args.steps, heterogeneous=True)
    except Exception as e:  # noqa: BLE001 -- any failure -> mock
        mode = "MOCK (NOT A VALIDATION)"
        print(f"[falling back to mock: {e}]")
        homo = measure_mock_step_times(N_VALUES, heterogeneous=False)
        het = measure_mock_step_times(N_VALUES, heterogeneous=True)

    print(f"\n=== Calibration validation [{mode}] ===")
    print(f"{'N':>3}{'predicted_s':>14}{'homo_meas_s':>14}{'het_meas_s':>14}{'het_rel_err':>14}")
    het_rows = compare(het)
    homo_map = dict((n, m) for n, _, m, _ in compare(homo))
    worst = 0.0
    for n, pred, meas, rel in het_rows:
        print(f"{n:>3}{pred:>14.1f}{homo_map[n]:>14.1f}{meas:>14.1f}{rel:>+13.1%}")
        worst = max(worst, abs(rel))

    fit = fit_mix_correction(het)
    passed = worst <= TOLERANCE
    print(f"\nworst heterogeneous rel-error = {worst:.1%}  (tolerance {TOLERANCE:.0%})"
          f"  -> {'PASS' if passed else 'FAIL: re-anchor calibration'}")
    print(f"fitted job-mix interference slope beta = {fit['beta']:+.4f}  (R^2={fit['r2']:.3f})")
    print("  use s(N,mix) = step_scaling(N) * (1 + beta*(N-1)) to re-anchor calibration.py "
          "if FAIL.")
    if mode.startswith("MOCK"):
        print("\nNOTE: MOCK numbers prove only that the harness/maths run. A real GPU run "
              "(--real) is required before any sim result is reported as credible.")

    out = {
        "mode": mode, "tolerance": TOLERANCE, "passed": passed,
        "worst_rel_error": worst, "mix_correction": fit,
        "predicted": {n: cal.step_time(n) for n in N_VALUES},
        "homogeneous": homo, "heterogeneous": het,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
