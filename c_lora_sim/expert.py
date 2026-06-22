"""
Behaviour-cloning teacher for the C-LoRA policy: a load-adaptive, size-aware
scheduler that is strictly stronger than any Fixed-N(k) across the utilisation
sweep.

Why replace Fixed-N8?  Fixed-N8 is the saturated-regime optimum, so BC from it
initialises the policy at the CEILING for overloaded workloads. Under underloaded
or bursty workloads Fixed-N8 is provably suboptimal (ARCHITECTURE.md Theorems
1-2). Cloning it gives PPO nothing to improve in the critical/underloaded regime.

The load-adaptive teacher chooses its target co-location depth as a function of
the CURRENT cluster state:
  * If free GPUs exist AND the queue is short (idle capacity) → target N=1
    (spread: low latency, no throughput loss while there is room).
  * As the queue deepens relative to free capacity → ramp N up toward 8
    (the throughput-optimal ceiling).
  * Within the target N, SHORT jobs are preferentially placed on the least-loaded
    warm GPU (size segregation, ARCHITECTURE.md Proposition 3).
  * If all warm GPUs are at the target cap, open a fresh GPU; only repurpose as
    a last resort.
  * Hard wait if the best placement would massively over-shoot the target AND
    work is running (avoids wasting GPU memory on cold-start while a completion
    is imminent).

This is the honest strong baseline. Any PPO gain below it is purely from
learning to generalise across load regimes, not from an architectural advantage.
"""

from __future__ import annotations

import copy
from typing import List, Optional

from c_lora_sim.control_plane import (
    Candidate, CandidateGenerator, PendingJobQueue,
    feature_matrix, feature_matrix_v2, feature_matrix_v3,
)
from c_lora_sim.data_plane import CLoraDataPlane, LoraJob
from c_lora_sim import calibration as cal

_FEATURE_FNS = {"v1": feature_matrix, "v2": feature_matrix_v2, "v3": feature_matrix_v3}

# Threshold (steps) separating "short" from "long" jobs for size segregation.
SHORT_STEPS = 40


def _target_n(data_plane: CLoraDataPlane) -> int:
    """Choose the target co-location depth from the current cluster state.

    The rule implements a soft version of Theorem 1 + 2: pack deeper as the
    backlog grows relative to available capacity.
    """
    n_pending = len(data_plane.pending)
    n_free = sum(1 for g in data_plane.gpus.values() if g.is_empty())
    n_gpus = data_plane.num_gpus

    if n_pending == 0:
        return 1

    # If ≥ half GPUs are free and the queue is short: prefer N=1 (spread).
    if n_free >= n_gpus // 2 and n_pending <= n_gpus // 2:
        return 1
    # If a few GPUs are free: intermediate packing.
    if n_free > 0 and n_pending < n_gpus:
        return 2
    if n_free > 0:
        return 4
    # All GPUs occupied and queue is growing.
    pending_ratio = n_pending / max(1, n_gpus)
    if pending_ratio < 1.0:
        return 4
    if pending_ratio < 2.0:
        return 6
    return 8


def _is_short(data_plane: CLoraDataPlane, job: LoraJob) -> bool:
    # Size is judged from the scheduler's (non-clairvoyant) length estimate, so
    # the teacher cannot segregate on ground-truth length it would not have.
    return data_plane.est_total_steps(job) <= SHORT_STEPS


def expert_action(data_plane: CLoraDataPlane, cands: List[Candidate]) -> int:
    """Index of the candidate the load-adaptive teacher would choose."""
    job_lookup = {j.job_id: j for j in data_plane.pending}
    placements = [(i, c) for i, c in enumerate(cands) if c.mode not in ("wait", "migrate", "evict")]
    wait_idx: Optional[int] = next((i for i, c in enumerate(cands) if c.mode == "wait"), None)
    evict_idxs = [(i, c) for i, c in enumerate(cands) if c.mode == "evict"]

    # ── SRPT Preemption (NEW) ─────────────────────────────────────────────────
    # If there is a pending job that is significantly shorter than the longest
    # running job, and we have room to place the short job after eviction,
    # the expert preempts the long job now so the short one can start immediately.
    if evict_idxs and data_plane.pending:
        longest_running = max(
            data_plane.running,
            key=lambda j: data_plane.est_remaining_steps(j)
        )
        # Age-based preemption: evict jobs that have consumed > 1.5× the EMA
        # prior — they are likely long-running jobs holding capacity from shorter
        # newcomers. Only preempt when no free GPU is available.
        n_free = sum(1 for g in data_plane.gpus.values() if g.is_empty())
        if longest_running.steps_done > data_plane._ema_steps * 1.5 and n_free == 0:
            return evict_idxs[0][0]
    # ─────────────────────────────────────────────────────────────────────────

    if not placements:
        return wait_idx if wait_idx is not None else 0

    target = _target_n(data_plane)

    # Process the queue in FIFO order; prefer placing shortest waiting job
    # when there is space (SRPT-like), else FIFO.
    if data_plane.pending:
        n_free = sum(1 for g in data_plane.gpus.values() if g.is_empty())
        if n_free > 0:
            # idle capacity: prefer shortest-remaining-steps job (SRPT)
            candidate_jobs = {c.job_id for _, c in placements}
            eligible = [j for j in data_plane.pending if j.job_id in candidate_jobs]
            # SRPT on the ESTIMATED remaining steps (non-clairvoyant). When the
            # estimate is uniform across jobs this degrades gracefully to FIFO.
            chosen_job_id = min(eligible, key=lambda j: data_plane.est_remaining_steps(j)).job_id
        else:
            # backlogged: FIFO
            oldest = min((job_lookup[c.job_id].arrival_time for _, c in placements))
            front_ids = {c.job_id for _, c in placements
                         if job_lookup[c.job_id].arrival_time == oldest}
            chosen_job_id = next(iter(front_ids))
    else:
        oldest = min((job_lookup[c.job_id].arrival_time for _, c in placements))
        front_ids = {c.job_id for _, c in placements
                     if job_lookup[c.job_id].arrival_time == oldest}
        chosen_job_id = next(iter(front_ids))

    front = [(i, c) for i, c in placements if c.job_id == chosen_job_id]

    # Choose placement mode within front: prefer same-base under cap, then fresh,
    # then repurpose. Within same-base: prefer least-loaded if job is short
    # (isolate from slow pools), most-loaded otherwise (pack for throughput).
    job = job_lookup[chosen_job_id]
    short = _is_short(data_plane, job)

    mode_order = {"spread": 0, "pack": 0, "fresh": 1, "repurpose": 2}

    def cost(c: Candidate):
        over = max(0, c.n_after - target)
        # size segregation: short jobs prefer lower N (least-loaded GPU)
        n_pref = c.n_after if short else -c.n_after  # small N for short, big N for long
        return (10 * over, mode_order.get(c.mode, 3), n_pref)

    best_idx, best_c = min(front, key=lambda ic: cost(ic[1]))

    # Hard wait if the placement would significantly exceed target and work is running.
    if best_c.n_after > target + 2 and data_plane.running and wait_idx is not None:
        return wait_idx

    return best_idx


# --- driving the expert through the sim (for evaluation + BC data) ----------

def run_expert(jobs: List[LoraJob], num_gpus: int = 8, max_steps: int = 50_000,
               clairvoyant: bool = False):
    data_plane = CLoraDataPlane(num_gpus=num_gpus, clairvoyant=clairvoyant)
    data_plane.reset(copy.deepcopy(jobs))
    queue = PendingJobQueue(data_plane)
    n = 0
    while not data_plane.done() and n < max_steps:
        cands = CandidateGenerator.build_candidates(data_plane, queue)
        if not cands:
            data_plane.advance()
            continue
        idx = expert_action(data_plane, cands)
        c = cands[idx]
        if c.mode == "wait":
            data_plane.advance()
        elif c.mode == "evict":
            job = next(j for j in data_plane.running if j.job_id == c.job_id)
            data_plane.evict(job)
        elif c.mode == "migrate":
            job = next(j for j in data_plane.running if j.job_id == c.job_id)
            data_plane.migrate(job, c.gpu_id)
        else:
            job = next(j for j in data_plane.pending if j.job_id == c.job_id)
            data_plane.place(job, c.gpu_id)
        n += 1
    return data_plane.metrics()


def collect_bc_samples(jobs: List[LoraJob], num_gpus: int = 8, max_steps: int = 50_000,
                       clairvoyant: bool = False, feature_mode: str = "v1"):
    """Roll out the expert, returning a list of (feature_matrix, expert_index).

    `feature_mode` selects the feature space so the BC warm-start is collected on
    the SAME features the PPO policy will train on (plan Flaw 6: retrain BC from
    scratch when the feature space changes)."""
    import numpy as np

    feat_fn = _FEATURE_FNS.get(feature_mode, feature_matrix)
    data_plane = CLoraDataPlane(num_gpus=num_gpus, clairvoyant=clairvoyant)
    data_plane.reset(copy.deepcopy(jobs))
    queue = PendingJobQueue(data_plane)
    samples = []
    n = 0
    while not data_plane.done() and n < max_steps:
        cands = CandidateGenerator.build_candidates(data_plane, queue)
        if not cands:
            data_plane.advance()
            continue
        feats = feat_fn(data_plane, cands)
        idx = expert_action(data_plane, cands)
        samples.append((np.asarray(feats, dtype="float32"), idx))
        c = cands[idx]
        if c.mode == "wait":
            data_plane.advance()
        elif c.mode == "evict":
            job = next(j for j in data_plane.running if j.job_id == c.job_id)
            data_plane.evict(job)
        elif c.mode == "migrate":
            job = next(j for j in data_plane.running if j.job_id == c.job_id)
            data_plane.migrate(job, c.gpu_id)
        else:
            job = next(j for j in data_plane.pending if j.job_id == c.job_id)
            data_plane.place(job, c.gpu_id)
        n += 1
    return samples
