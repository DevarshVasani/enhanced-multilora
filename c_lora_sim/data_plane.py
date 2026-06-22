"""
Calibrated discrete-event simulator for continuous multi-LoRA training.

Physics (see calibration.py for the article-derived constants):

  * A GPU is "warm" for exactly one base model. Adapters (LoRA training jobs)
    of that base model can be hot-loaded and co-located on it.
  * When N adapters share a GPU, true SGMV multiplexing advances all of them,
    but each individual adapter's per-step wall-time is `step_time(N)` which
    grows with N. So:
        - per-job latency  GROWS with N        (article: 1.97x at N=8)
        - cluster throughput = N / step_time(N) PEAKS near N=8 then declines.
    This is the throughput<->latency tension the scheduler must navigate.
  * Placing a job whose base model differs from the GPU's warm base model costs
    a full base-model load (cold start). Same base model => ~free adapter swap.

The simulator is driven one placement at a time so that the SAME physics back
both the RL policy (candidate selection) and the heuristic baselines, making
the comparison apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from c_lora_sim import calibration as cal
from spark_env.timeline import Timeline

# --- online job-length estimation (non-clairvoyant scheduling) -------------
# Real continual-learning jobs run until convergence, so their length is NOT
# known at submission time. Unless the data plane is created with
# `clairvoyant=True`, every online scheduler (RL policy, BC teacher, heuristic
# baselines) must plan against an ESTIMATE rather than `job.total_steps`: a
# running EMA of the lengths of jobs that have ALREADY completed, seeded with a
# nominal prior. This removes the look-ahead the policy/teacher previously got
# by reading the ground-truth length straight off the job. Hindsight oracles
# are constructed with clairvoyant=True, so they remain a true upper bound.
EMA_PRIOR_STEPS = 150.0   # nominal "typical job length" before anything completes
EMA_ALPHA = 0.3           # weight on the most recent completed job

# --- online per-job length re-estimation (Flaw 3) --------------------------
# The global EMA above is frozen between completions, so a job that runs LONGER
# than the EMA would otherwise show ~0 estimated-remaining (false "it's safe").
# We therefore grow a job's estimated TOTAL length once its observed progress
# approaches/exceeds the prior, guaranteeing a strictly positive remaining
# estimate for over-running jobs. ESTIMATE_OVERRUN_MULT sets how far above the
# observed progress the running estimate floats (1.2 == assume ~20% more left).
ESTIMATE_OVERRUN_MULT = 1.2

# A very large slack ratio standing in for "no deadline" / "infinitely safe".
SLACK_INF = 1e6


@dataclass
class LoraJob:
    job_id: int
    base_model_id: str
    total_steps: int
    arrival_time: float

    # SLO: absolute wall-time by which the job must finish. -1 == no deadline.
    deadline: float = -1.0

    # dynamic state
    steps_done: float = 0.0
    gpu_id: int = -1
    start_time: float = -1.0          # when it first began progressing (post cold start)
    completion_time: float = -1.0
    cold_start_paid: float = 0.0      # seconds of cold start this job caused
    # True iff this job was force-preempted (rescue/SRPT path). Used by the
    # controller assert guard to prove no LOAD-SHED eviction ever happened on a
    # downswing (the asymmetric-actuator invariant).
    force_evicted: bool = False
    # True iff the job was cancelled because its deadline became unrecoverable.
    dropped: bool = False

    def remaining_steps(self) -> float:
        return max(0.0, self.total_steps - self.steps_done)

    def is_done(self) -> bool:
        return self.remaining_steps() <= 1e-9


class GPUPool:
    """One GPU, warm for a single base model, hosting a set of co-located adapters."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.base_model_id: Optional[str] = None
        self.active: List[LoraJob] = []
        self.last_update: float = 0.0
        # Realized per-step jitter multiplier for the GPU's *current* occupancy
        # regime. 1.0 == nominal physics. Re-drawn whenever N changes (i.e. each
        # time the co-location depth — and thus the operating point — shifts), so
        # the actual wall-time the scheduler gets for a given packing depth is
        # uncertain. See CLoraDataPlane noise model.
        self.step_noise: float = 1.0

    @property
    def n(self) -> int:
        return len(self.active)

    def is_empty(self) -> bool:
        return len(self.active) == 0


@dataclass
class _Completion:
    job_id: int
    gpu_id: int
    version: int


@dataclass
class _Arrival:
    job_id: int


@dataclass
class _Ready:
    """A job finished its cold start and begins progressing on its GPU."""
    job_id: int
    gpu_id: int


class CLoraDataPlane:
    """Discrete-event continuous multi-LoRA training simulator (Data Plane)."""

    def __init__(self, num_gpus: int = 8, max_adapters: int = cal.MAX_ADAPTERS_PER_GPU,
                 step_time_cv: float = 0.0, cold_start_cv: float = 0.0,
                 noise_seed: Optional[int] = None, clairvoyant: bool = False):
        self.num_gpus = num_gpus
        self.max_adapters = max_adapters
        # When True, schedulers may read the true job length (job.total_steps /
        # remaining_steps) -- reserved for hindsight oracles. When False (the
        # default for every online scheduler), length is hidden behind the EMA
        # estimator below. The simulator's own physics ALWAYS uses the true
        # length; only what the scheduler is allowed to *see* is gated here.
        self.clairvoyant = clairvoyant
        self._ema_steps = EMA_PRIOR_STEPS
        # ---- real-world-jitter model (off by default => exact deterministic) --
        # Hardware does not honour the calibrated constants exactly: NVMe/NAS
        # contention perturbs base-model reloads; shared PCIe bandwidth and CUDA
        # context switching perturb per-step compute. These CVs inject that
        # jitter into the *realized* dynamics, WITHOUT changing the nominal
        # values the scheduler plans against (features still use cal.*). A policy
        # that only works on noiseless physics will degrade here.
        self.step_time_cv = step_time_cv      # coeff. of variation on step_time(N)
        self.cold_start_cv = cold_start_cv    # coeff. of variation on cold-start cost
        self.noise_seed = noise_seed
        self._rng = np.random.RandomState(noise_seed)
        self.gpus: Dict[int, GPUPool] = {i: GPUPool(i) for i in range(num_gpus)}
        self.timeline = Timeline()
        self.wall_time = 0.0

        self.jobs: List[LoraJob] = []
        self.pending: List[LoraJob] = []      # arrived, awaiting placement
        self.running: List[LoraJob] = []      # placed (cold-starting or progressing)
        self.finished: List[LoraJob] = []
        self.dropped: List[LoraJob] = []      # cancelled (deadline unrecoverable)
        self._version: Dict[int, int] = {}    # job_id -> live completion-event version
        self.total_cold_start = 0.0
        self.total_evict_cost = 0.0           # seconds of in-flight work lost to preemption
        # --- realized-N volatility (the honest chatter metric, Flaw 4) --------
        # Total variation of the cluster-wide co-location depth Σ_g n_g over the
        # episode. Captures genuine packing churn (place / drain / re-place),
        # unlike a count of *commanded* N_target changes which can move while the
        # physically-locked realized N does not.
        self._realized_n_tv = 0.0
        self._last_total_n = 0

    # -- setup -------------------------------------------------------------
    def reset(self, jobs: List[LoraJob]) -> None:
        self.gpus = {i: GPUPool(i) for i in range(self.num_gpus)}
        self._rng = np.random.RandomState(self.noise_seed)
        self.timeline.reset()
        self.wall_time = 0.0
        self.jobs = jobs
        self.pending = []
        self.running = []
        self.finished = []
        self.dropped = []
        self._version = {}
        self.total_cold_start = 0.0
        self.total_evict_cost = 0.0
        self._ema_steps = EMA_PRIOR_STEPS
        self._realized_n_tv = 0.0
        self._last_total_n = 0
        for job in jobs:
            self.timeline.push(job.arrival_time, _Arrival(job.job_id))

    # -- online length estimation -----------------------------------------
    def est_total_steps(self, job: LoraJob) -> float:
        """Total length of `job` as seen by an *online* scheduler.

        Clairvoyant mode returns the ground truth. Otherwise we use the running
        EMA of already-completed lengths, but ONLINE-CORRECTED (Flaw 3): once a
        job's observed progress approaches/exceeds the EMA prior, the estimate
        grows with progress (`steps_done * ESTIMATE_OVERRUN_MULT`) so an
        over-running job is never reported as "done", which would give a false
        sense of deadline safety."""
        if self.clairvoyant:
            return float(job.total_steps)
        return max(self._ema_steps, job.steps_done * ESTIMATE_OVERRUN_MULT)

    def est_remaining_steps(self, job: LoraJob) -> float:
        """Remaining steps as seen by an online scheduler. `steps_done` is
        observable (we have watched the job train); the total is not. The online
        estimate (above) guarantees a strictly positive remaining for jobs that
        have already over-run the EMA prior."""
        if self.clairvoyant:
            return job.remaining_steps()
        return max(0.0, self.est_total_steps(job) - job.steps_done)

    def progress_ratio(self, job: LoraJob) -> float:
        """Observed progress as a fraction of the (online-corrected) estimate,
        in [0, 1]. A value pinned near 1.0 while the job keeps running is the
        signal that the length estimate is stale / the job is over-running."""
        est = self.est_total_steps(job)
        return min(1.0, job.steps_done / est) if est > 0 else 0.0

    def overrun(self, job: LoraJob) -> float:
        """How far observed progress has exceeded the EMA prior, normalised.
        0.0 while the job is within its expected length; grows once it over-runs.
        Lets a policy learn to distrust the deadline-safety signal for long
        agentic tasks whose length was under-estimated."""
        if self.clairvoyant or self._ema_steps <= 0:
            return 0.0
        return max(0.0, job.steps_done - self._ema_steps) / self._ema_steps

    def slack_ratio(self, job: LoraJob) -> float:
        """Scale-invariant deadline pressure (Flaw 2/3): available time divided
        by the *best-case* remaining wall-time (at N=1, no co-location penalty).
            slack_ratio = (deadline - now) / (est_remaining_steps * STEP_TIME_SOLO)
        >1 comfortable, ~1 on the edge, <1 at risk, <0 already missed. Returns a
        large sentinel for deadline-free jobs (infinitely safe)."""
        if job.deadline < 0:
            return SLACK_INF
        best_case_work = self.est_remaining_steps(job) * cal.STEP_TIME_SOLO
        if best_case_work <= 0:
            return SLACK_INF
        return (job.deadline - self.wall_time) / best_case_work

    # -- noise model -------------------------------------------------------
    def _draw_step_noise(self) -> float:
        """Multiplier on step_time(N) for a GPU's current occupancy regime."""
        if self.step_time_cv <= 0.0:
            return 1.0
        # truncated-normal-ish: never let compute get implausibly fast.
        return max(0.5, 1.0 + self._rng.normal(0.0, self.step_time_cv))

    def _draw_cold_noise(self) -> float:
        """Multiplier on a cold-start cost (NVMe/NAS/PCIe contention)."""
        if self.cold_start_cv <= 0.0:
            return 1.0
        return max(0.05, 1.0 + self._rng.normal(0.0, self.cold_start_cv))

    def _eff_step_time(self, gpu: GPUPool) -> float:
        """Realized (jittered) seconds-per-step for this GPU's current depth."""
        return cal.step_time(gpu.n) * gpu.step_noise

    # -- progress bookkeeping ---------------------------------------------
    def _accrue(self, gpu: GPUPool, now: float) -> None:
        """Advance every progressing adapter on `gpu` up to `now`."""
        delta = now - gpu.last_update
        gpu.last_update = now
        if delta <= 0 or gpu.n == 0:
            return
        rate = 1.0 / self._eff_step_time(gpu)   # steps per second, per adapter
        gained = rate * delta
        for job in gpu.active:
            if job.start_time >= 0 and job.start_time <= now:
                job.steps_done += gained

    def _bump_version(self, job: LoraJob) -> int:
        v = self._version.get(job.job_id, 0) + 1
        self._version[job.job_id] = v
        return v

    def _record_total_n(self) -> None:
        """Accumulate the total variation of cluster-wide co-location depth.
        Call after every event that changes any GPU's occupancy."""
        total_n = sum(g.n for g in self.gpus.values())
        self._realized_n_tv += abs(total_n - self._last_total_n)
        self._last_total_n = total_n

    def _floor_partial_step(self, job: LoraJob, gpu: GPUPool) -> float:
        """Discard the in-flight (fractional) training step of `job`: a real
        gradient step cannot be checkpointed mid forward/backward, so preempting
        wastes the partial step. Floors `steps_done` to the last completed step
        and returns the wasted wall-time (~fraction * step_time at this depth)."""
        frac = job.steps_done - float(int(job.steps_done))
        job.steps_done = float(int(job.steps_done))
        wasted = frac * self._eff_step_time(gpu)
        self.total_evict_cost += wasted
        return wasted

    def _reschedule_gpu_completions(self, gpu: GPUPool) -> None:
        """Recompute completion events for all progressing adapters on a GPU
        (their rate changed because N changed). Stale events self-invalidate
        via the version counter."""
        if gpu.n == 0:
            return
        # The occupancy regime just changed: re-draw the realized per-step jitter
        # for this GPU, then schedule completions against the jittered step time.
        gpu.step_noise = self._draw_step_noise()
        step = self._eff_step_time(gpu)
        for job in gpu.active:
            if job.start_time < 0 or job.start_time > self.wall_time:
                continue  # still cold-starting; its _Ready event will schedule it
            v = self._bump_version(job)
            eta = self.wall_time + job.remaining_steps() * step
            self.timeline.push(eta, _Completion(job.job_id, gpu.gpu_id, v))

    # -- placement (the scheduler's action) -------------------------------
    def place(self, job: LoraJob, gpu_id: int) -> float:
        """Place a pending job on a GPU. Returns the cold-start delay incurred."""
        gpu = self.gpus[gpu_id]
        self._accrue(gpu, self.wall_time)
        cold = cal.placement_cold_start(gpu.base_model_id, job.base_model_id) * self._draw_cold_noise()
        if gpu.base_model_id != job.base_model_id:
            gpu.base_model_id = job.base_model_id  # (only legal when gpu idle)
        gpu.active.append(job)
        job.gpu_id = gpu_id
        job.cold_start_paid = cold
        self.total_cold_start += cold
        self.pending.remove(job)
        self.running.append(job)
        # Job becomes ready (starts progressing) after the cold start.
        self.timeline.push(self.wall_time + cold, _Ready(job.job_id, gpu_id))
        # Existing adapters on this GPU keep their rate until the new job is
        # actually ready; recompute their completions when it goes ready.
        self._record_total_n()
        return cold

    def migrate(self, job: LoraJob, target_gpu_id: int) -> float:
        """Migrates a progressing job to a new GPU. Returns swap delay."""
        source_gpu = self.gpus[job.gpu_id]
        target_gpu = self.gpus[target_gpu_id]
        
        self._accrue(source_gpu, self.wall_time)
        self._accrue(target_gpu, self.wall_time)
        
        # Remove from source
        source_gpu.active.remove(job)
        if source_gpu.is_empty():
            self.unload_base_model(source_gpu)
        else:
            self._reschedule_gpu_completions(source_gpu)
            
        # Add to target
        cold = cal.placement_cold_start(target_gpu.base_model_id, job.base_model_id) * self._draw_cold_noise()
        if target_gpu.base_model_id != job.base_model_id:
            target_gpu.base_model_id = job.base_model_id
            
        target_gpu.active.append(job)
        job.gpu_id = target_gpu_id
        job.start_time = -1.0  # Paused during swap
        
        # Schedule ready event for target GPU
        self.timeline.push(self.wall_time + cold, _Ready(job.job_id, target_gpu_id))
        return cold

    def unload_adapter(self, gpu: GPUPool, job: LoraJob) -> None:
        """Remove a single adapter's execution state from a GPU. Does NOT touch
        the shared base-model weights (no CUDA-context re-init), so re-placing a
        same-base adapter later costs only the 0.5 s swap. This is the cheap half
        of teardown used on the rescue/preempt path."""
        gpu.active.remove(job)
        self._bump_version(job)          # invalidate any stale completion event
        if not gpu.is_empty():
            self._reschedule_gpu_completions(gpu)

    def unload_base_model(self, gpu: GPUPool) -> None:
        """Drop the GPU's resident base-model weights (the expensive teardown:
        a fresh mount later pays the full base-model reload). Only legal once the
        GPU has no remaining adapters."""
        if gpu.is_empty():
            gpu.base_model_id = None

    def evict(self, job: LoraJob, *, free: bool = False) -> float:
        """Preempt a running job back to the pending queue (rescue/SRPT path only
        — NEVER for load-shedding to a lower N_target).

        Physical model: the in-flight gradient step is lost (`steps_done` floored
        to the last completed step) and the wasted partial-step wall-time is
        charged to `total_evict_cost`. The base model stays warm (only
        `unload_adapter` is called), so re-placement costs an adapter-swap, not a
        base reload. Returns the wasted seconds.

        `free=True` restores the OLD unphysical behaviour (no partial-step loss):
        reserved exclusively for the unconstrained lower-bound oracle, which is
        allowed to scale N down instantaneously for free."""
        gpu = self.gpus[job.gpu_id]
        self._accrue(gpu, self.wall_time)
        wasted = 0.0 if free else self._floor_partial_step(job, gpu)
        # base_model_id is intentionally kept warm in VRAM (cheap teardown).
        self.unload_adapter(gpu, job)
        self.running.remove(job)
        job.gpu_id = -1
        job.start_time = -1.0
        if not free:
            job.force_evicted = True     # invariant marker for the assert guard
        self.pending.append(job)
        self._record_total_n()
        return wasted

    def cancel(self, job: LoraJob) -> None:
        """Drop a job whose deadline is unrecoverable (anti-zombie, Flaw 3). It
        frees its GPU slot immediately and is EXCLUDED from throughput and
        first-completion accounting — a missed SLO must not flatter the metrics
        by still being 'processed'."""
        if job in self.running:
            gpu = self.gpus[job.gpu_id]
            self._accrue(gpu, self.wall_time)
            self.unload_adapter(gpu, job)
            self.running.remove(job)
            self.unload_base_model(gpu)
        elif job in self.pending:
            self.pending.remove(job)
        job.dropped = True
        job.gpu_id = -1
        self.dropped.append(job)
        self._record_total_n()

    # -- event loop --------------------------------------------------------
    def advance(self) -> Tuple[str, Optional[int]]:
        """Pop and apply the next event. Returns (event_kind, job_id)."""
        if len(self.timeline) == 0:
            return ("idle", None)
        when_raw, event = self.timeline.pop()
        assert when_raw is not None  # guarded by the length check above
        when = float(when_raw)
        # accrue progress on all GPUs up to this event time
        for gpu in self.gpus.values():
            self._accrue(gpu, when)
        self.wall_time = when

        # Anti-zombie: drop any job whose deadline has now irrecoverably passed,
        # freeing its GPU slot so it cannot bottleneck the pool or flatter the
        # throughput metric by lingering (Flaw 3). No-op for deadline-free jobs.
        self._process_deadline_cancellations()

        if isinstance(event, _Arrival):
            job = next((j for j in self.jobs if j.job_id == event.job_id), None)
            if job is not None:
                self.pending.append(job)
            return ("arrival", event.job_id)

        if isinstance(event, _Ready):
            job = next((j for j in self.running if j.job_id == event.job_id), None)
            if job is not None and job.start_time < 0:
                job.start_time = self.wall_time
                gpu = self.gpus[event.gpu_id]
                # now that N effectively includes this job, reschedule the pool
                self._reschedule_gpu_completions(gpu)
            return ("ready", event.job_id)

        if isinstance(event, _Completion):
            if event.version != self._version.get(event.job_id, -1):
                return ("stale", event.job_id)        # superseded; ignore
            job = next((j for j in self.running if j.job_id == event.job_id), None)
            if job is None:
                return ("stale", event.job_id)
            job.steps_done = job.total_steps
            job.completion_time = self.wall_time
            self.running.remove(job)
            self.finished.append(job)
            # The true length is observable only now (at convergence); fold it
            # into the running estimate future placements will plan against.
            self._ema_steps += EMA_ALPHA * (job.total_steps - self._ema_steps)
            gpu = self.gpus[job.gpu_id]
            gpu.active.remove(job)
            if gpu.is_empty():
                self.unload_base_model(gpu)
            else:
                self._reschedule_gpu_completions(gpu)
            self._record_total_n()
            return ("complete", event.job_id)

        return ("unknown", None)

    def _process_deadline_cancellations(self) -> None:
        """Cancel jobs whose deadline has already passed without completion.
        Conservative (hard-miss only): a job is dropped once `now > deadline`,
        which removes confirmed zombies while never pre-empting a job that might
        still finish in time."""
        for job in list(self.running) + list(self.pending):
            if job.deadline >= 0 and self.wall_time > job.deadline:
                self.cancel(job)

    def done(self) -> bool:
        return len(self.finished) + len(self.dropped) == len(self.jobs)

    # -- metrics -----------------------------------------------------------
    def metrics(self) -> Dict[str, float]:
        jcts = [j.completion_time - j.arrival_time for j in self.finished]
        jcts_sorted = sorted(jcts)

        def pct(p: float) -> float:
            if not jcts_sorted:
                return 0.0
            k = min(len(jcts_sorted) - 1, int(round(p * (len(jcts_sorted) - 1))))
            return jcts_sorted[k]

        makespan = max((j.completion_time for j in self.finished), default=0.0)
        total_steps = sum(j.total_steps for j in self.finished)
        # GPU-seconds of useful training delivered, normalised by a single
        # adapter's per-step cost. `serial_1gpu / makespan` measures total
        # cluster throughput (parallelism * multiplexing); the article-style
        # *multiplexing* speedup is computed in evaluation against a
        # no-multiplexing (N=1) reference on the same hardware.
        serial_1gpu = total_steps * cal.STEP_TIME_SOLO

        # First-completion latency (time-to-first-result), over genuinely
        # completed (non-dropped) jobs only.
        first_completion = min((j.completion_time for j in self.finished),
                               default=0.0)

        # Deadline accounting. A job with a deadline is a HIT iff it actually
        # finished on time; cancelled jobs and finished-late jobs are misses.
        deadline_jobs = [j for j in self.jobs if j.deadline >= 0]
        hits = sum(1 for j in self.finished
                   if j.deadline >= 0 and j.completion_time <= j.deadline)
        n_dl = len(deadline_jobs)

        # Completion-robust flow time: charge every NOT-yet-finished job (still
        # pending or running because the episode was truncated, or the policy
        # stalled) its elapsed time-in-system. Unlike `mean_jct` (finished only),
        # this CANNOT be gamed by a policy that completes a few short jobs and
        # stalls on the rest — use it as the honest eval/learning-curve metric.
        unfinished = self.pending + self.running
        flows = jcts + [self.wall_time - j.arrival_time for j in unfinished]
        mean_flow_all = (sum(flows) / len(flows)) if flows else 0.0
        frac_unfinished = len(unfinished) / len(self.jobs) if self.jobs else 0.0

        return {
            "jobs_finished": len(self.finished),
            "jobs_dropped": len(self.dropped),
            "mean_jct": sum(jcts) / len(jcts) if jcts else 0.0,
            "makespan": makespan,
            "time_to_first_completion": first_completion,
            "p50_jct": pct(0.50),
            "p95_jct": pct(0.95),
            "p99_jct": pct(0.99),
            "max_jct": jcts_sorted[-1] if jcts_sorted else 0.0,
            "total_cold_start": self.total_cold_start,
            "total_evict_cost": self.total_evict_cost,
            "cluster_throughput_x": (serial_1gpu / makespan) if makespan > 0 else 0.0,
            "deadline_hit_rate": (hits / n_dl) if n_dl > 0 else 1.0,
            "deadline_miss": (n_dl - hits),
            "realized_N_tv": self._realized_n_tv,
            "mean_flow_all": mean_flow_all,
            "frac_unfinished": frac_unfinished,
        }
