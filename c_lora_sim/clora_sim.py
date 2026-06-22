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

from c_lora_sim import calibration as cal
from spark_env.timeline import Timeline


@dataclass
class LoraJob:
    job_id: int
    base_model_id: str
    total_steps: int
    arrival_time: float

    # dynamic state
    steps_done: float = 0.0
    gpu_id: int = -1
    start_time: float = -1.0          # when it first began progressing (post cold start)
    completion_time: float = -1.0
    cold_start_paid: float = 0.0      # seconds of cold start this job caused

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


class CLoraSim:
    """Discrete-event continuous multi-LoRA training simulator."""

    def __init__(self, num_gpus: int = 8, max_adapters: int = cal.MAX_ADAPTERS_PER_GPU):
        self.num_gpus = num_gpus
        self.max_adapters = max_adapters
        self.gpus: Dict[int, GPUPool] = {i: GPUPool(i) for i in range(num_gpus)}
        self.timeline = Timeline()
        self.wall_time = 0.0

        self.jobs: List[LoraJob] = []
        self.pending: List[LoraJob] = []      # arrived, awaiting placement
        self.running: List[LoraJob] = []      # placed (cold-starting or progressing)
        self.finished: List[LoraJob] = []
        self._version: Dict[int, int] = {}    # job_id -> live completion-event version
        self.total_cold_start = 0.0

    # -- setup -------------------------------------------------------------
    def reset(self, jobs: List[LoraJob]) -> None:
        self.gpus = {i: GPUPool(i) for i in range(self.num_gpus)}
        self.timeline.reset()
        self.wall_time = 0.0
        self.jobs = jobs
        self.pending = []
        self.running = []
        self.finished = []
        self._version = {}
        self.total_cold_start = 0.0
        for job in jobs:
            self.timeline.push(job.arrival_time, _Arrival(job.job_id))

    # -- progress bookkeeping ---------------------------------------------
    def _accrue(self, gpu: GPUPool, now: float) -> None:
        """Advance every progressing adapter on `gpu` up to `now`."""
        delta = now - gpu.last_update
        gpu.last_update = now
        if delta <= 0 or gpu.n == 0:
            return
        rate = 1.0 / cal.step_time(gpu.n)   # steps per second, per adapter
        gained = rate * delta
        for job in gpu.active:
            if job.start_time >= 0 and job.start_time <= now:
                job.steps_done += gained

    def _bump_version(self, job: LoraJob) -> int:
        v = self._version.get(job.job_id, 0) + 1
        self._version[job.job_id] = v
        return v

    def _reschedule_gpu_completions(self, gpu: GPUPool) -> None:
        """Recompute completion events for all progressing adapters on a GPU
        (their rate changed because N changed). Stale events self-invalidate
        via the version counter."""
        if gpu.n == 0:
            return
        step = cal.step_time(gpu.n)
        for job in gpu.active:
            if job.start_time < 0 or job.start_time > self.wall_time:
                continue  # still cold-starting; its _Ready event will schedule it
            v = self._bump_version(job)
            eta = self.wall_time + job.remaining_steps() * step
            self.timeline.push(eta, _Completion(job.job_id, gpu.gpu_id, v))

    # -- placement (the scheduler's action) -------------------------------
    def candidate_gpus(self, job: LoraJob) -> Dict[str, List[int]]:
        """Legal placements for `job`, grouped by kind."""
        warm_same: List[int] = []      # warm for this base model, has room
        empty: List[int] = []          # completely free GPU
        repurpose: List[int] = []      # idle but warm for a DIFFERENT base model
        for gpu in self.gpus.values():
            if gpu.base_model_id == job.base_model_id and gpu.n < self.max_adapters:
                warm_same.append(gpu.gpu_id)
            elif gpu.is_empty():
                empty.append(gpu.gpu_id)
            elif gpu.n == 0:
                repurpose.append(gpu.gpu_id)
        return {"warm_same": warm_same, "empty": empty, "repurpose": repurpose}

    def place(self, job: LoraJob, gpu_id: int) -> float:
        """Place a pending job on a GPU. Returns the cold-start delay incurred."""
        gpu = self.gpus[gpu_id]
        self._accrue(gpu, self.wall_time)
        cold = cal.placement_cold_start(gpu.base_model_id, job.base_model_id)
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
        return cold

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
            gpu = self.gpus[job.gpu_id]
            gpu.active.remove(job)
            if gpu.is_empty():
                gpu.base_model_id = None
            else:
                self._reschedule_gpu_completions(gpu)
            return ("complete", event.job_id)

        return ("unknown", None)

    def done(self) -> bool:
        return len(self.finished) == len(self.jobs)

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
        return {
            "jobs_finished": len(self.finished),
            "mean_jct": sum(jcts) / len(jcts) if jcts else 0.0,
            "makespan": makespan,
            "p50_jct": pct(0.50),
            "p95_jct": pct(0.95),
            "p99_jct": pct(0.99),
            "max_jct": jcts_sorted[-1] if jcts_sorted else 0.0,
            "total_cold_start": self.total_cold_start,
            "cluster_throughput_x": (serial_1gpu / makespan) if makespan > 0 else 0.0,
        }
