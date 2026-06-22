"""
Workload generation for the C-LoRA scheduler experiments.

A workload is a stream of LoRA adapter-training jobs. The knobs that make
scheduling hard, all grounded in the article's setting, are:

  * base-model diversity: the continual-learning service maps each experiment
    to its own adapter, but adapters cluster onto a handful of shared base
    models (Llama / Qwen / Mistral families). Locality-aware packing matters.
  * arrival pressure: jobs arrive over time. The critical fix vs. the first
    attempt is that arrivals are scaled to the *measured service time* and
    parameterised by a target utilisation `target_rho`, so the cluster actually
    spans the underloaded -> critically-loaded -> overloaded regimes. Under low
    rho the scheduler should spread (low N, low latency); under high rho it must
    pack (high N, high throughput). See ARCHITECTURE.md Section 1-2.
  * within-episode bursts: continual-learning submission is bursty. A two-state
    Markov-modulated Poisson process (burst/lull) makes the optimal co-location
    depth a function of *time within the episode*, which is what forces a
    genuinely adaptive (learned) scheduler -- C-LoRA's open problem of "dynamic
    load balancing across jobs" (ARCHITECTURE.md Theorem 2).
  * size heterogeneity: experiments differ in length (short vs long). Because
    the per-step penalty is shared by all co-located adapters, packing a short
    job behind long ones inflates its latency, so size-aware co-location beats
    size-blind packing for mean flow time (ARCHITECTURE.md Proposition 3).
"""

from __future__ import annotations

import random
from typing import List

from c_lora_sim import calibration as cal
from c_lora_sim.data_plane import LoraJob

# A few base-model families with realistic relative popularity. The skew is
# what makes locality packing pay off: most jobs share a handful of bases.
BASE_MODELS = ["Llama-3-8B", "Qwen-2.5-7B", "Mistral-7B", "Llama-3-70B", "Gemma-2-9B"]
BASE_WEIGHTS = [0.34, 0.26, 0.18, 0.12, 0.10]

# --- job-size distribution (steps). Two modes: short and long experiments. ---
P_SHORT = 0.30
SHORT_RANGE = (5, 30)
LONG_RANGE = (100, 500)

# Reference hardware for the rho->arrival-rate conversion. Must match the sim's
# default num_gpus; overridable via generate_workload(num_gpus=...).
NUM_GPUS_REF = 8

# --- bimodal task-structure profiles (Flaw 5) ------------------------------
# The synthetic 30/70 split varies LOAD but not task STRUCTURE. Trajectory's
# real workloads are strongly bimodal: a "Tau"-style stream of short, high-rate
# prompt-tuning runs, and an "APEX"-style stream of long, low-rate agentic
# trajectories. A controller tuned on the synthetic mix may behave completely
# differently on these, so we expose them as named presets and a combined
# stream, used by the generalization experiment. Each preset overrides the size
# distribution; arrival rate is still set by target_rho unless given explicitly.
PROFILES = {
    # short tasks dominate, packed tightly together (high arrival pressure)
    "tau":  {"p_short": 0.90, "short_range": (5, 20),  "long_range": (60, 120)},
    # long agentic trajectories dominate, arriving sparsely
    "apex": {"p_short": 0.05, "short_range": (40, 80), "long_range": (300, 900)},
}


def mean_steps(p_short: float = P_SHORT,
               short_range=SHORT_RANGE, long_range=LONG_RANGE) -> float:
    """Analytic mean of the job-size distribution (uniform within each mode)."""
    short_mean = 0.5 * (short_range[0] + short_range[1])
    long_mean = 0.5 * (long_range[0] + long_range[1])
    return p_short * short_mean + (1.0 - p_short) * long_mean


def reference_capacity_n1(num_gpus: int = NUM_GPUS_REF,
                          p_short: float = P_SHORT) -> float:
    """Max sustainable job-completion rate [jobs/s] with NO multiplexing (N=1).

    mu = num_gpus * (1 adapter) / (mean_steps * step_time(1)). Utilisation is
    defined against this N=1 capacity so that `target_rho >= 1` means the
    no-multiplexing reference is overloaded (the scheduler MUST multiplex to
    stay stable) and `target_rho < 1` means there is genuine idle capacity to
    spread into. See ARCHITECTURE.md Section 2.1.
    """
    return num_gpus / (mean_steps(p_short) * cal.step_time(1))


def mean_interarrival_for_rho(target_rho: float, num_gpus: int = NUM_GPUS_REF,
                              p_short: float = P_SHORT) -> float:
    """Inter-arrival mean [s] that yields time-average utilisation `target_rho`
    against the N=1 reference capacity."""
    return 1.0 / (target_rho * reference_capacity_n1(num_gpus, p_short))


def _sample_steps(rng: random.Random, p_short: float,
                  short_range=SHORT_RANGE, long_range=LONG_RANGE) -> int:
    if rng.random() < p_short:
        return rng.randint(*short_range)        # short experiment
    return rng.randint(*long_range)             # long experiment


def generate_workload(
    num_jobs: int = 60,
    seed: int = 0,
    *,
    target_rho: float | None = None,
    mean_interarrival: float | None = None,
    bursty: bool = False,
    burst_rate_mult: float = 1.75,   # arrival-rate multiplier during a burst
    lull_rate_mult: float = 0.25,    # ... and during a lull
    mean_phase_jobs: float = 10.0,   # expected jobs per burst/lull phase
    p_short: float = P_SHORT,
    num_gpus: int = NUM_GPUS_REF,
    immediate: bool = False,
    profile: str | None = None,      # "tau" | "apex" -> bimodal size preset
    p_deadline: float = 0.0,         # fraction of jobs given an SLO deadline
    deadline_slack_mult: float = 2.0,  # mean window = mult x best-case (N=1) runtime
) -> List[LoraJob]:
    """Generate a reproducible stream of LoRA training jobs.

    Load is set by exactly one of:
      * `immediate=True`      -- all jobs at t=0 (backlog burst).
      * `mean_interarrival=`  -- explicit Poisson mean (backward compatible).
      * `target_rho=`         -- utilisation against N=1 capacity (preferred).
    If none is given, defaults to `target_rho=0.9`.

    With `bursty=True`, arrivals follow a two-state Markov-modulated Poisson
    process whose burst/lull rate multipliers average to 1 (so the time-average
    utilisation stays `target_rho`) but whose instantaneous load crosses 1 in
    both directions, exercising the dynamic-load-balancing regime.
    """
    rng = random.Random(seed)

    # Bimodal task-structure preset overrides the size distribution.
    short_range, long_range = SHORT_RANGE, LONG_RANGE
    if profile is not None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {list(PROFILES)}")
        cfg = PROFILES[profile]
        p_short = cfg["p_short"]
        short_range, long_range = cfg["short_range"], cfg["long_range"]

    if immediate:
        base_ia = 0.0
    elif mean_interarrival is not None:
        base_ia = mean_interarrival
    else:
        base_ia = mean_interarrival_for_rho(target_rho if target_rho is not None else 0.9,
                                            num_gpus, p_short)

    # rate multipliers chosen so the time-average rate matches base (with equal
    # expected time in each phase the average of the two multipliers must be 1).
    switch_p = 1.0 / max(1.0, mean_phase_jobs)
    in_burst = rng.random() < 0.5

    jobs: List[LoraJob] = []
    t = 0.0
    for i in range(num_jobs):
        if not immediate:
            if bursty:
                mult = burst_rate_mult if in_burst else lull_rate_mult
                rate = mult / base_ia if base_ia > 0 else 0.0
                t += rng.expovariate(rate) if rate > 0 else 0.0
                if rng.random() < switch_p:
                    in_burst = not in_burst
            else:
                # plain Poisson with a small chance of a simultaneous arrival
                if rng.random() >= 0.2:
                    t += rng.expovariate(1.0 / base_ia) if base_ia > 0 else 0.0

        base = rng.choices(BASE_MODELS, weights=BASE_WEIGHTS, k=1)[0]
        steps = _sample_steps(rng, p_short, short_range, long_range)
        arrival = 0.0 if immediate else t
        # Optional SLO: window = slack_mult x the best-case (N=1) solo runtime,
        # drawn around the mean so tenants are a mix of tight and loose.
        deadline = -1.0
        if p_deadline > 0.0 and rng.random() < p_deadline:
            solo_runtime = steps * cal.step_time(1)
            mult = rng.uniform(0.6 * deadline_slack_mult, 1.4 * deadline_slack_mult)
            deadline = arrival + mult * solo_runtime
        jobs.append(
            LoraJob(
                job_id=i,
                base_model_id=base,
                total_steps=steps,
                arrival_time=arrival,
                deadline=deadline,
            )
        )
    return jobs
