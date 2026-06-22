"""
Candidate action construction + featurisation for the C-LoRA RL scheduler.

At each decision point the policy chooses among a bounded set of candidate
actions. Each candidate is a concrete (job, gpu) placement annotated with a
"mode" that captures the scheduler's two coupled levers:

    pack      -> tightest same-base GPU (max co-location N => max throughput)
    spread    -> least-loaded same-base GPU (low N => low latency)
    fresh     -> a brand-new GPU pool for this base model
    repurpose -> take over an idle GPU warm for a different base (pays cold start)

plus a single global WAIT action (advance time without placing). This lets the
learned policy reproduce, mix, or improve on every heuristic baseline, and in
particular *adapt the co-location depth to the current load* -- the operating
point the article fixes by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from c_lora_sim import calibration as cal
from c_lora_sim.clora_sim import CLoraSim

# Number of pending jobs (by arrival order) considered each decision. Bounds the
# candidate set so the policy sees a stable, tractable action space.
QUEUE_WINDOW = 10
CLORA_FEATURE_DIM = 20

# normalisation scales (rough, just to keep features O(1))
_STEPS_SCALE = 120.0
_TIME_SCALE = 20_000.0
_COLD_SCALE = cal.BASE_MODEL_LOAD_S


@dataclass
class Candidate:
    job_id: int            # -1 for WAIT
    gpu_id: int            # -1 for WAIT
    mode: str              # pack|spread|fresh|repurpose|wait
    n_after: int           # resulting co-location count on the target GPU
    cold_start: float


def _best_same_base(sim: CLoraSim, gpu_ids: List[int], most_loaded: bool) -> Optional[int]:
    if not gpu_ids:
        return None
    key = (lambda g: sim.gpus[g].n)
    return max(gpu_ids, key=key) if most_loaded else min(gpu_ids, key=key)


def build_candidates(sim: CLoraSim) -> List[Candidate]:
    """Construct the candidate action list for the current sim state."""
    cands: List[Candidate] = []
    pending = sorted(sim.pending, key=lambda j: j.arrival_time)[:QUEUE_WINDOW]

    for job in pending:
        groups = sim.candidate_gpus(job)
        warm = groups["warm_same"]
        empty = groups["empty"]
        repurpose = groups["repurpose"]

        if warm:
            g_pack = _best_same_base(sim, warm, most_loaded=True)
            g_spread = _best_same_base(sim, warm, most_loaded=False)
            assert g_pack is not None and g_spread is not None
            cands.append(Candidate(job.job_id, g_pack, "pack",
                                   sim.gpus[g_pack].n + 1,
                                   cal.placement_cold_start(sim.gpus[g_pack].base_model_id, job.base_model_id)))
            if g_spread != g_pack:
                cands.append(Candidate(job.job_id, g_spread, "spread",
                                       sim.gpus[g_spread].n + 1,
                                       cal.placement_cold_start(sim.gpus[g_spread].base_model_id, job.base_model_id)))
        if empty:
            g = empty[0]
            cands.append(Candidate(job.job_id, g, "fresh", 1,
                                   cal.placement_cold_start(None, job.base_model_id)))
        if not warm and not empty and repurpose:
            g = repurpose[0]
            cands.append(Candidate(job.job_id, g, "repurpose", 1,
                                   cal.placement_cold_start(sim.gpus[g].base_model_id, job.base_model_id)))

    # Global WAIT, only meaningful if something is running to free capacity.
    if sim.running:
        cands.append(Candidate(-1, -1, "wait", 0, 0.0))

    # Safety: never return an empty action set while jobs are pending.
    if not cands and sim.pending:
        job = pending[0]
        groups = sim.candidate_gpus(job)
        for key in ("warm_same", "empty", "repurpose"):
            if groups[key]:
                g = groups[key][0]
                cands.append(Candidate(job.job_id, g, key,
                                       sim.gpus[g].n + 1,
                                       cal.placement_cold_start(sim.gpus[g].base_model_id, job.base_model_id)))
                break
    return cands


_MODE_INDEX = {"pack": 0, "spread": 1, "fresh": 2, "repurpose": 3, "wait": 4}


def feature_matrix(sim: CLoraSim, cands: List[Candidate]) -> np.ndarray:
    """[num_candidates, CLORA_FEATURE_DIM] feature matrix."""
    n_gpus = sim.num_gpus
    total_pending = len(sim.pending)
    total_running = len(sim.running)
    free_gpus = sum(1 for g in sim.gpus.values() if g.is_empty())
    active_ns = [g.n for g in sim.gpus.values() if g.n > 0]
    mean_n = (sum(active_ns) / len(active_ns)) if active_ns else 0.0

    job_lookup = {j.job_id: j for j in sim.pending}
    rows = []
    for c in cands:
        mode_onehot = [0.0] * 5
        mode_onehot[_MODE_INDEX[c.mode]] = 1.0

        if c.job_id == -1:  # WAIT
            job_feats = [0.0, 0.0, 0.0]
            base_pending = 0
            base_warm_gpus = 0
            base_frac_warm = 0.0
            n_after = mean_n
        else:
            job = job_lookup[c.job_id]
            job_feats = [
                job.remaining_steps() / _STEPS_SCALE,
                job.total_steps / _STEPS_SCALE,
                (sim.wall_time - job.arrival_time) / _TIME_SCALE,
            ]
            base_pending = sum(1 for j in sim.pending if j.base_model_id == job.base_model_id)
            base_warm_gpus = sum(1 for g in sim.gpus.values() if g.base_model_id == job.base_model_id)
            base_frac_warm = base_warm_gpus / n_gpus
            n_after = c.n_after

        row = [
            *job_feats,                                   # 0-2
            n_after / cal.MAX_ADAPTERS_PER_GPU,           # 3 resulting co-location depth
            cal.step_scaling(max(1, int(round(n_after)))),# 4 per-job latency multiplier incurred
            cal.aggregate_speedup(max(1, int(round(n_after)))),  # 5 throughput at that depth
            c.cold_start / _COLD_SCALE,                   # 6 cold-start cost
            *mode_onehot,                                 # 7-11
            base_pending / max(1, QUEUE_WINDOW),          # 12 queue pressure for this base
            base_warm_gpus / n_gpus,                      # 13
            base_frac_warm,                               # 14
            total_pending / max(1, len(sim.jobs)),        # 15 global queue pressure
            total_running / max(1, n_gpus),               # 16
            free_gpus / n_gpus,                           # 17
            mean_n / cal.MAX_ADAPTERS_PER_GPU,            # 18 cluster co-location load
            1.0 if c.mode == "wait" else 0.0,             # 19
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)
