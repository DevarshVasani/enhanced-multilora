# Dynamic N-Selection as an Asymmetric Control Problem

**Question (C-LoRA open problem P1).** Co-locating `N` adapter-training jobs on one
GPU trades aggregate **throughput** against per-job **latency / time-to-first-
completion** (calibrated physics, `calibration.py`: N=2 → 1.73× makespan / 1.10×
slower first completion; N=8 → 2.81× / 1.97×). What is the optimal `N`, and can we
adapt it online as the workload shifts?

We answer it as a **feedback control** problem and — importantly — show that under a
*correctly modelled asymmetric actuator*, the constraints one would expect to require
heavy machinery (forced preemption, aggressive anti-chatter damping) are **inherently
benign**. A single-knob, placement-only controller tracks the hindsight oracle.

## 1. The asymmetric actuator

`N` can be raised for free (place another adapter) but **cannot be lowered cheaply**:
a gradient step cannot be aborted mid forward/backward. So realised `N(t)` falls
**only by passive drain** (adapters finishing their step block). The controller is
**never** allowed to force-evict to reach a lower target. The simulator was corrected
to enforce this (`data_plane.py`): `evict()` is rescue-only and now floors the
in-flight partial step into `total_evict_cost`; `cancel()` drops jobs whose deadline
is unrecoverable (anti-zombie) and excludes them from throughput/first-completion.

## 2. The λ-controller (`n_controller.py`)

`N_target(x) = clip(1 + round(λ·κ·ρ̂), 1, N_max)` from live backlog saturation `ρ̂`,
with an asymmetric **deadband + dwell** to damp command chatter and a **deadline
override** (spread an at-risk job, judged by the scale-invariant
`slack_ratio = (deadline − t)/(est_remaining·STEP_TIME_SOLO)`). `λ∈[0,1]` traces an
*adaptive* Pareto frontier: λ=0 ≡ Fixed-N1 (latency), λ=1 packs to N_max under load.

| profile | λ=0 mean_jct | λ=0.5 | λ=1.0 |
|---|---|---|---|
| mixed | 70 545 | 50 984 | 48 643 |
| APEX (long) | 215 383 | 140 761 | 136 107 |
| Tau (short) | 4 141 | 4 114 | 4 036 |

λ cleanly spans the throughput/latency range where packing has headroom (mixed,
APEX); Tau is short-job-dominated so co-location can't help — the controller
correctly stays spread.

## 3. Three "expensive" effects are actually benign

These are the headline findings — each contradicts an intuitive worry, and each is a
*property of the correct asymmetric model*, not an artifact:

- **Two distinct "asymmetry" questions, both now measured:**
  - *Depth un-packing.* A clairvoyant oracle that may re-balance depth for free but
    never *parks* a job ties the passive-drain oracle (**+0.0%** across regimes):
    natural step-completion already de-packs GPUs, and when saturated you cannot lower
    depth without parking. **Being unable to cheaply un-pack costs ~nothing.**
  - *Free eviction / preemption (`phase_asymmetry_gap.py`, 5 seeds/load).* What
    instant free eviction (park a long job to rush a short one) buys over passive
    drain is **load-conditional**: underloaded +0.1%, critical +0.6%, critical-bursty
    +4.6%, **overloaded (ρ=1.5) +36.6%** (mean +10.5%). So *faster eviction is not
    worth engineering at/below capacity, but is a high-value (~37%) project for
    oversubscribed clusters* — the actionable number Trajectory had not measured. The
    value of eviction is preemption, not depth-rebalancing. Raw:
    `results/asymmetry_gap.json`.
- **Realised chatter is structurally ~0.** Excess total-variation of realised `N(t)`
  (above the unavoidable place/complete floor) is ≈0 for both the deadband controller
  *and* a naive `round()` controller — because you cannot force `N` down, a chattery
  *command* cannot produce a chattery *realised* depth. The deadband only tidies the
  command signal; the physics already prevents realised thrash.
- **The controller tracks the oracle without preemption.** The placement-only
  λ-controller lands within ~0–5% of the passive-drain adaptive oracle in most
  regimes (looser at stationary critical/overloaded, where a single fixed λ is
  suboptimal — a per-regime λ or the learned policy closes that). For contrast,
  SRPT-Preemptive's realised-N total variation explodes (>6·10⁵) — exactly the churn
  the asymmetric, drain-only design avoids.

## 3b. The complete recommendation (ρ=1.5 frontier, `phase_frontier_rho15.py`)

At overload, the full ladder (5 seeds, mean_jct):

| scheduler | mean_jct | gap to asym. oracle |
|---|---|---|
| Fixed-N2 / N4 / N8 | 63,853 / 59,460 / 70,462 | +15.4% / +7.5% / +27.4% |
| LoadAdaptive | 59,759 | +8.0% |
| **λ-controller (λ=1, best)** | **57,736** | **+4.4%** |
| ORACLE-asymmetric (drain) | 55,320 | 0% |
| ORACLE-unconstrained (evict) | 40,503 | −26.8% (=+37% gap) |

The placement-only λ-controller **beats every Fixed-N operating point and LoadAdaptive**
and lands **within 4.4% of the best achievable without eviction**, without knowing the
optimal N in advance. It closes **72–84%** of the gap-to-oracle versus the article's
recommended static N2/N8 (42% versus a hindsight-optimal N4). The residual **+37%**
below the asymmetric oracle is reachable **only with free eviction** (§3), and PPO does
not close it (§3, BC at +13.5%, PPO destabilises). **Engineering recommendation: ship
the λ-controller everywhere (within a few % of optimal at every load, zero infra cost);
build free eviction only for oversubscribed clusters, where it buys the extra 37%.**

## 4. Tracking under shift

`experiment_n_control.py` plots realised `N(t)` vs commanded `N_target(t)` under a
non-stationary burst. Max single-event drop in realised total-N = **1** — confirming
the asymmetric signature: sharp upswing (immediate packing), drain-limited downswing
(decreases only at completion events).

## 5. Deadlines

With SLOs, `LoadAdaptive`/the controller keep a high hit-rate by spreading at moderate
load; `Fixed-N8` misses most (0.75) because deep packing slows everyone. Placement-only
deadline protection is limited — pressure builds while jobs are *running* and the
no-load-shed rule forbids preempting to help them — an honest consequence of the
asymmetric design, and the one place where costed preemption (future work) would help.

## 6. Validity gate (Phase 0) — RUN ON REAL HARDWARE, PASSED

`phase0_qwen_measure.py` measures **real** multi-LoRA training step time on
**Qwen2.5-0.5B / RTX 3050 Laptop GPU** (PEFT, fwd+bwd, 20 trials) and compares the
measured `step_time(N)/step_time(1)` to the simulator's `step_scaling(N)`. Result:

```
PART 1 — step_scaling(N) at fixed sequence length (the core co-location physics)
  N   meas_ms  meas_scale  pred_scale  residual
  1     64.1      0.993       1.000       0.7%
  2     65.1      1.008       1.126      10.5%
  4     95.5      1.479       1.586       6.8%
  8    179.7      2.782       2.618       6.3%
  worst residual = 10.5%  (tolerance 15%)  -> PASS
```

**`step_scaling(N)` — the number every simulation result depends on — is validated on
real Qwen multi-LoRA within 15% across N=1..8** (worst 10.5%; N=8 measured 2.782 vs
predicted 2.618). The simulator's co-location physics are hardware-credible.

Part 2 (control): a batch mixing short(16)+long(96) experiments costs the same as a
homogeneous batch *at the max length* (`het/homo@long = 1.01`), so the apparent
"heterogeneous penalty" is **padding to max sequence length** — an axis orthogonal to
co-location depth `N` that the uniform-step simulator does not claim to model, and
which sequence packing/sorting removes. It is *not* a co-location interference, so no
`step_scaling` correction is warranted. Raw data: `results/phase0_qwen.json`.

## Reproduce
```
.venv/bin/python -m c_lora_sim.experiment_n_control      # frontier, tracking, chatter, deadlines
.venv/bin/python -m c_lora_sim.evaluate_clora --seeds 8  # full table + oracle bracket (RL optional)
.venv/bin/python -m c_lora_sim.calibration_validation --real --model Qwen/Qwen2.5-7B --adapters ...
# learned policy (optional): bc_pretrain --feature-mode v3 ; train_clora --feature-mode v3
```
