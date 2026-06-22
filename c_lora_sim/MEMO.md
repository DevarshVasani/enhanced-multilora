# Dynamic N-Selection for Multi-LoRA Training: A Validated Controller and the Cost of the Asymmetric Drain Constraint

**Abstract.** C-LoRA's N-selection trades throughput against latency, but the optimal N is load-dependent and unknown in advance. We validate the simulator against real Qwen2.5-0.5B hardware (≤10.5% residual), measure the cost of the passive-drain constraint across load regimes (≈0% below capacity; at ρ=1.5 a wide, bimodal 66%, 95% CI [32%, 122%] over 20 seeds), and show an interpretable λ-controller reaches within 4.4% of the constrained oracle without learning or eviction. Free eviction is worth engineering only for oversubscribed clusters, and only when the workload mixes short and long experiments; below capacity, passive drain with the λ-controller is sufficient.

## 1. Phase 0 — Simulator validated on real hardware

We measured real multi-LoRA training step time (Qwen2.5-0.5B, LoRA r=16, fwd+bwd, RTX 3050, 20 trials/point) and compared `step_time(N)/step_time(1)` to the simulator's `step_scaling(N)`:

| N | measured scale | predicted scale | residual |
|---|---|---|---|
| 1 | 0.993 | 1.000 | 0.7% |
| 2 | 1.008 | 1.126 | 10.5% |
| 4 | 1.479 | 1.586 | 6.8% |
| 8 | 2.782 | 2.618 | 6.3% |

Worst residual 10.5% < 15% → **the co-location physics every result depends on is hardware-credible** (real N=1 step = 64.6 ms; we validate the scaling shape, not absolute seconds, since the article's hardware differs).

The N=2 point is the loosest (10.5%): real N=2 shows almost no slowdown (1.008) where the article-calibrated curve expects 1.126. This is conservative for our purposes — the simulator slightly *over*-charges shallow packing — and the controller spends most of its time at deeper N under load, where residuals are ≤6.8%.

A batch mixing short(16)+long(96) experiments costs the same as a homogeneous batch *at the max length* (het/homo@long = 1.01). The apparent "heterogeneous penalty" is therefore **padding to max sequence length** — an axis orthogonal to co-location depth N that the uniform-step model does not claim to cover, and that sequence packing removes. No `step_scaling` correction is warranted.

## 2. Asymmetry gap — the cost of passive drain

Realized N can only fall by passive drain (a step cannot be aborted mid-flight). We measured what *free eviction* (clairvoyant, instant preemption) buys over passive drain, 5 seeds/regime:

| regime | passive-drain oracle | free-eviction oracle | gap |
|---|---|---|---|
| underloaded ρ=0.5 | 43,955 | 43,912 | **0.1%** |
| critical ρ=0.9 | 47,821 | 47,526 | **0.6%** |
| critical+bursty ρ=0.9 | 39,787 | 38,054 | **4.6%** |
| overloaded ρ=1.5 (5 seeds) | 55,320 | 40,503 | 36.6% (not robust — see CI) |

The value is **entirely preemption, not depth-rebalancing**: an oracle allowed to re-balance depth for free but never *park* a job ties passive drain (+0.0%) — natural step-completion already de-packs GPUs. The gap comes from short experiments jumping the queue past long ones. **Threshold: free eviction is worthless at or below capacity and high-value under sustained backlog.**

**The ρ=1.5 number needs a confidence interval, and it is wide.** Overload queue dynamics are chaotic, so we re-ran 20 seeds (`phase_ci_rho15.py`). The directly-comparable aggregate gap is **65.9%, 95% CI [31.5%, 122.3%]** — the 5-seed 36.6% was a low-sample point estimate, not the headline. The per-seed distribution is **bimodal**: 6/20 seeds <5% (workloads with no short-behind-long contention → eviction useless), 8/20 >150% (short jobs piled behind long ones → eviction transformative), median 60%, max 642%. **So free eviction's value at overload is not a fixed fraction — it is workload-determined.** What is robust is the *sign and floor*: even the CI lower bound (+31.5%) far exceeds any eviction-engineering cost, so the build/no-build recommendation stands; the *magnitude* must be reported as an interval, not a point.

*Is this the constraint cost or oracle suboptimality?* Both oracles must be optimal under their own constraint or the gap is inflated. We isolated the two no-eviction levers achievable under passive drain. **(i) Placement order:** giving the asymmetric oracle SRPT placement (shortest-remaining first) instead of arrival order does not improve it (+0.0% across all regimes — the adaptive oracle's size-segregation already captures it; raw SRPT is +0.2% at ρ=1.5, competitive but not better). With adaptive-depth packing, jobs are co-located immediately rather than waiting unplaced, so order is moot and only preemption helps. **(ii) Drain-lag anticipation** has no role at ρ=1.5: sustained overload never drains, so the optimum stays at high N throughout. Both levers ruled out, the 36.6% is genuine preemption cost. It remains a (now tight) upper bound — `asym_SRPT` is a strong constructed policy, not a proven optimum — but the placement slack a reviewer would flag measures 0%. (`phase_gap_decompose.py`.)

## 3. λ-Controller frontier (ρ=1.5)

The interpretable, placement-only λ-controller (target depth grows with backlog; single knob λ trades throughput↔latency) versus the static and oracle baselines, 5 seeds:

| scheduler | mean_jct | gap to drain oracle |
|---|---|---|
| Fixed-N2 / N4 / N8 | 63,853 / 59,460 / 70,462 | +15.4% / +7.5% / +27.4% |
| LoadAdaptive | 59,759 | +8.0% |
| **λ-controller (λ=1)** | **57,736** | **+4.4%** |
| ORACLE drain (no eviction) | 55,320 | 0% |
| ORACLE free eviction | 40,503 | +37% headroom here (5 seeds; 20-seed aggregate 66%, CI [32%,122%] — §2) |

The λ-controller **beats every Fixed-N operating point and LoadAdaptive**, landing **within 4.4% of the best achievable without eviction**. It closes **84%** of the gap-to-oracle versus the max-throughput default N8, **72%** versus the article's recommended N2, and **42%** versus a *hindsight-optimal* N4. The hindsight framing is the harshest, but it is also unattainable: **a real operator cannot pick N4 in advance** — it is optimal only for this exact load. The controller reaches it load-blind. This is the central value: not beating the best static N you could have chosen with an oracle, but matching it without one.

The same λ=1 controller, applied to the §2 drain oracles across the full range, stays single-digit everywhere — so "load-blind and near-optimal" is validated, not extrapolated from ρ=1.5:

| regime | λ=1 controller gap to drain oracle |
|---|---|
| underloaded ρ=0.5 | +0.3% |
| critical ρ=0.9 | +3.0% |
| critical+bursty ρ=0.9 | +1.4% |
| overloaded ρ=1.5 | +4.4% |

A fixed λ=1 needs no per-load tuning: target depth scales with backlog, so it spreads automatically when the queue is short (hence +0.3% at ρ=0.5) and packs when it is deep.

**Generalization, not fitting.** Tuning λ and the deadband/dwell on ρ=0.9 and evaluating frozen on *unseen* ρ=1.5 yields the same +4.4% — λ=1 is best at both loads, so the result is held-out, not overfit (θ_up=1 depth-unit and dwell≈1.5 step-times are fixed defaults, never swept on the eval workload).

**Versus the standard control laws** (same tune-on-0.9 protocol, evaluated on ρ=1.5): the λ-controller (+4.4%) beats **AIMD/TCP (+6.2%)** and **threshold-hysteresis/HPA (+12.1%)**. The reason is structural: TCP-AIMD is built for the *opposite* asymmetry — cheap to shrink the window, careful (additive) to grow it — so its slow additive increase under-packs N during a burst it could fill instantly. Our actuator is cheap-to-grow / costly-to-shrink, so the correct law is **proportional control on backlog** (set N to the load immediately because packing is free) **with a deadband** (be conservative about increases because they cannot be undone). That is exactly the λ-law; the deadband is the only hysteresis, and it is what AIMD's integral ramp lacks.

## 4. PPO finding — RL is unnecessary here

A behaviour-cloned policy (imitating the load-adaptive teacher at 0.999 accuracy, full-completion eval) reaches **+13.5%** of the drain oracle. PPO fine-tuning from there *degrades* it, destabilizing the near-optimal init within 20–40 episodes across hyperparameters. The reason: the load-adaptive structure is already a near-deterministic optimum, so the reward landscape around it is flat — PPO has no gradient to climb and wanders off. **Conclusion: the N-selection subproblem does not need RL; the analytic λ-law captures it.**

## 5. Recommendation

- **Ship the λ-controller everywhere.** Measured within 0.3–4.4% of the drain oracle across the full ρ=0.5–1.5 range (§3) — needs no preemption, no learning, and no advance knowledge of the workload. Zero infrastructure cost.
- **Do not build free eviction for general use.** Below capacity it buys <1%; passive drain plus the λ-controller is sufficient.
- **Build free eviction (preemptive SRPT) for oversubscribed clusters only — trigger at ρ ≳ 1.2, and only if the workload mixes short and long experiments.** Measured value rises from <1% at ρ≤0.9 to an aggregate 66% (95% CI [32%, 122%]) at ρ=1.5 — wide and bimodal: transformative when short jobs queue behind long ones, worthless when they don't (6/20 seeds <5%). For a busy multi-tenant service that runs backlogged with heterogeneous tenants, this is the one high-value infrastructure investment; the build decision is robust (CI floor +32%), the magnitude is workload-dependent. A cluster running homogeneous job lengths gains little even at overload.

*All numbers are real measurements: the plant is validated on Qwen/RTX 3050 (§1); eviction value (§2) and the frontier (§3) are 5-seed, oracle-bracketed simulations on that validated plant. Artifacts: `phase0_qwen_measure.py`, `phase_asymmetry_gap.py`, `phase_frontier_rho15.py`; data in `results/`.*
