"""
Control Plane for the C-LoRA RL Scheduler.

This module encapsulates the scheduling logic (RL-BSBF) and Candidate Generation,
bridging the job stream and the Data Plane (C-LoRA Runtime Cluster).

Components:
- PendingJobQueue: Buffers arriving jobs and exposes an optimization window (top 10 oldest).
- Candidate Generator: Permissible structural actions (Pack/Spread/Fresh/Repurpose/Wait/Migrate).
- RLBSBFControlPlane: Wraps the Transformer Actor-Critic network to select actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from c_lora_sim import calibration as cal
from c_lora_sim.data_plane import CLoraDataPlane, LoraJob

# Number of pending jobs (by arrival order) considered each decision. Bounds the
# candidate set so the policy sees a stable, tractable action space.
QUEUE_WINDOW = 10
CLORA_FEATURE_DIM = 24    # 24-dim feature vector (v1 + evict mode)
CLORA_FEATURE_DIM_V2 = 29 # extended with 5 SRPT-aware features (v2)
CLORA_FEATURE_DIM_V3 = 34 # v2 + 5 asymmetric-actuator / deadline features (v3)

# normalisation scales (rough, just to keep features O(1))
_STEPS_SCALE = 120.0
_TIME_SCALE = 20_000.0
_COLD_SCALE = cal.BASE_MODEL_LOAD_S


def _target_n_from_state(n_pending: int, n_free: int, n_gpus: int) -> int:
    """Compute the load-adaptive target co-location depth from queue state.
    Mirrors expert._target_n but takes raw counts so it can be called without
    a data_plane reference (used for feature engineering).
    """
    if n_pending == 0:
        return 1
    if n_free >= n_gpus // 2 and n_pending <= n_gpus // 2:
        return 1
    if n_free > 0 and n_pending < n_gpus:
        return 2
    if n_free > 0:
        return 4
    ratio = n_pending / max(1, n_gpus)
    if ratio < 1.0:
        return 4
    if ratio < 2.0:
        return 6
    return 8


@dataclass
class Candidate:
    job_id: int            # -1 for WAIT
    gpu_id: int            # -1 for WAIT / EVICT uses source gpu_id
    mode: str              # pack|spread|fresh|repurpose|wait|migrate|evict
    n_after: int           # resulting co-location count on the target GPU
    cold_start: float


class PendingJobQueue:
    """Pending Job Queue Buffer (Control Plane)"""
    def __init__(self, data_plane: CLoraDataPlane):
        self.data_plane = data_plane

    def get_optimisation_window(self) -> List[LoraJob]:
        """Enforces FIFO-bypass for SJF behavior by exposing the top 10 oldest pending requests."""
        return sorted(self.data_plane.pending, key=lambda j: j.arrival_time)[:QUEUE_WINDOW]


class CandidateGenerator:
    """Generates valid candidate actions (Pack/Spread/Wait/Repurpose/Fresh/Migrate)."""

    @staticmethod
    def _candidate_gpus(data_plane: CLoraDataPlane, job: LoraJob) -> Dict[str, List[int]]:
        """Legal placements for `job`, grouped by kind."""
        warm_same: List[int] = []      # warm for this base model, has room
        empty: List[int] = []          # completely free GPU
        repurpose: List[int] = []      # idle but warm for a DIFFERENT base model
        for gpu in data_plane.gpus.values():
            if gpu.base_model_id == job.base_model_id and gpu.n < data_plane.max_adapters:
                warm_same.append(gpu.gpu_id)
            elif gpu.is_empty():
                empty.append(gpu.gpu_id)
            elif gpu.n == 0:
                repurpose.append(gpu.gpu_id)
        return {"warm_same": warm_same, "empty": empty, "repurpose": repurpose}

    @staticmethod
    def _best_same_base(data_plane: CLoraDataPlane, gpu_ids: List[int], most_loaded: bool) -> Optional[int]:
        if not gpu_ids:
            return None
        key = (lambda g: data_plane.gpus[g].n)
        return max(gpu_ids, key=key) if most_loaded else min(gpu_ids, key=key)

    @classmethod
    def build_candidates(cls, data_plane: CLoraDataPlane, queue: PendingJobQueue) -> List[Candidate]:
        """Construct the candidate action list for the current sim state."""
        cands: List[Candidate] = []
        pending = queue.get_optimisation_window()

        for job in pending:
            groups = cls._candidate_gpus(data_plane, job)
            warm = groups["warm_same"]
            empty = groups["empty"]
            repurpose = groups["repurpose"]

            if warm:
                g_pack = cls._best_same_base(data_plane, warm, most_loaded=True)
                g_spread = cls._best_same_base(data_plane, warm, most_loaded=False)
                assert g_pack is not None and g_spread is not None
                cands.append(Candidate(job.job_id, g_pack, "pack",
                                       data_plane.gpus[g_pack].n + 1,
                                       cal.placement_cold_start(data_plane.gpus[g_pack].base_model_id, job.base_model_id)))
                if g_spread != g_pack:
                    cands.append(Candidate(job.job_id, g_spread, "spread",
                                           data_plane.gpus[g_spread].n + 1,
                                           cal.placement_cold_start(data_plane.gpus[g_spread].base_model_id, job.base_model_id)))
            if empty:
                g = empty[0]
                cands.append(Candidate(job.job_id, g, "fresh", 1,
                                       cal.placement_cold_start(None, job.base_model_id)))
            if not warm and not empty and repurpose:
                g = repurpose[0]
                cands.append(Candidate(job.job_id, g, "repurpose", 1,
                                       cal.placement_cold_start(data_plane.gpus[g].base_model_id, job.base_model_id)))

        # Add migration candidates
        base_to_gpus = {}
        empty_gpus = []
        for gpu in data_plane.gpus.values():
            if gpu.is_empty():
                empty_gpus.append(gpu.gpu_id)
            else:
                base_to_gpus.setdefault(gpu.base_model_id, []).append(gpu.gpu_id)

        for b, g_list in base_to_gpus.items():
            most = max(g_list, key=lambda g: data_plane.gpus[g].n)
            least = min(g_list, key=lambda g: data_plane.gpus[g].n)
            
            target = None
            if empty_gpus:
                target = empty_gpus[0]
            else:
                target = least
                
            n_most = data_plane.gpus[most].n
            n_target = 0 if target in empty_gpus else data_plane.gpus[target].n
            
            if n_most >= n_target + 2 and data_plane.gpus[target].n < data_plane.max_adapters:
                job_to_migrate = min(data_plane.gpus[most].active, key=lambda j: j.arrival_time)
                cands.append(Candidate(
                    job_id=job_to_migrate.job_id,
                    gpu_id=target,
                    mode="migrate",
                    n_after=n_target + 1,
                    cold_start=cal.placement_cold_start(data_plane.gpus[target].base_model_id, b)
                ))

        # Global WAIT, only meaningful if something is running to free capacity.
        if data_plane.running:
            cands.append(Candidate(-1, -1, "wait", 0, 0.0))

        # Eviction candidates: allow preempting the running job with the most
        # remaining steps whenever there are pending jobs that are shorter.
        # This is the key action that enables SRPT-style preemption.
        if data_plane.running and data_plane.pending:
            longest_running = max(
                data_plane.running,
                key=lambda j: data_plane.est_remaining_steps(j)
            )
            # Only offer eviction if the best pending job is meaningfully shorter
            # than the longest running job (avoid thrashing).
                # Age-based preemption (non-clairvoyant-safe): a running job that has
            # already consumed more than 1.5× the EMA prior is very likely long.
            # Offer eviction when such a job exists AND pending jobs haven't yet
            # started (i.e., they might be short relative to the stalled job).
            # This avoids the EMA-deadlock of length-based preemption where all
            # pending jobs look identical (EMA=150) and the 4× guard never fires.
            if (longest_running.steps_done > data_plane._ema_steps * 1.5):
                cands.append(Candidate(
                    job_id=longest_running.job_id,
                    gpu_id=longest_running.gpu_id,
                    mode="evict",
                    n_after=data_plane.gpus[longest_running.gpu_id].n - 1,
                    cold_start=0.0,   # adapter swap is nearly free
                ))

        # Safety: never return an empty action set while jobs are pending.
        if not cands and data_plane.pending:
            job = pending[0]
            groups = cls._candidate_gpus(data_plane, job)
            for key in ("warm_same", "empty", "repurpose"):
                if groups[key]:
                    g = groups[key][0]
                    cands.append(Candidate(job.job_id, g, key,
                                           data_plane.gpus[g].n + 1,
                                           cal.placement_cold_start(data_plane.gpus[g].base_model_id, job.base_model_id)))
                    break
        return cands


_MODE_INDEX = {"pack": 0, "spread": 1, "fresh": 2, "repurpose": 3, "wait": 4, "migrate": 5, "evict": 6}


def feature_matrix(data_plane: CLoraDataPlane, cands: List[Candidate]) -> np.ndarray:
    """[num_candidates, CLORA_FEATURE_DIM] feature matrix."""
    n_gpus = data_plane.num_gpus
    total_pending = len(data_plane.pending)
    total_running = len(data_plane.running)
    free_gpus = sum(1 for g in data_plane.gpus.values() if g.is_empty())
    active_ns = [g.n for g in data_plane.gpus.values() if g.n > 0]
    mean_n = (sum(active_ns) / len(active_ns)) if active_ns else 0.0

    job_lookup = {j.job_id: j for j in data_plane.pending + data_plane.running}
    rows = []
    for c in cands:
        mode_onehot = [0.0] * 7   # pack|spread|fresh|repurpose|wait|migrate|evict
        mode_onehot[_MODE_INDEX[c.mode]] = 1.0
        
        gpu_rem_steps = 0.0
        if c.gpu_id != -1:
            gpu_rem_steps = sum(j.remaining_steps() for j in data_plane.gpus[c.gpu_id].active) / _STEPS_SCALE

        if c.job_id == -1:  # WAIT
            job_feats = [0.0, 0.0, 0.0]
            base_pending = 0
            base_warm_gpus = 0
            base_frac_warm = 0.0
            n_after = mean_n
            queue_affinity = 0.0
        else:
            job = job_lookup[c.job_id]
            # Use est_remaining_steps so the policy sees the same EMA-based
            # estimate that the baselines plan against (non-clairvoyant parity).
            job_feats = [
                data_plane.est_remaining_steps(job) / _STEPS_SCALE,
                data_plane.est_total_steps(job) / _STEPS_SCALE,
                (data_plane.wall_time - job.arrival_time) / _TIME_SCALE,
            ]
            base_pending = sum(1 for j in data_plane.pending if j.base_model_id == job.base_model_id)
            base_warm_gpus = sum(1 for g in data_plane.gpus.values() if g.base_model_id == job.base_model_id)
            base_frac_warm = base_warm_gpus / n_gpus
            n_after = c.n_after
            queue_affinity = base_pending / max(1, QUEUE_WINDOW)

        row = [
            *job_feats,                                   # 0-2
            n_after / cal.MAX_ADAPTERS_PER_GPU,           # 3 resulting co-location depth
            cal.step_scaling(max(1, int(round(n_after)))),# 4 per-job latency multiplier incurred
            cal.aggregate_speedup(max(1, int(round(n_after)))),  # 5 throughput at that depth
            c.cold_start / _COLD_SCALE,                   # 6 cold-start cost
            *mode_onehot,                                 # 7-13
            base_pending / max(1, QUEUE_WINDOW),          # 14 queue pressure for this base
            base_warm_gpus / n_gpus,                      # 15
            base_frac_warm,                               # 16
            total_pending / max(1, len(data_plane.jobs)), # 17 global queue pressure
            total_running / max(1, n_gpus),               # 18
            free_gpus / n_gpus,                           # 19
            mean_n / cal.MAX_ADAPTERS_PER_GPU,            # 20 cluster co-location load
            1.0 if c.mode == "wait" else 0.0,             # 21
            gpu_rem_steps,                                # 22 GPU remaining steps
            queue_affinity,                               # 23 Queue base model affinity
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def feature_matrix_v2(data_plane: CLoraDataPlane, cands: List[Candidate]) -> np.ndarray:
    """[num_candidates, CLORA_FEATURE_DIM_V2] feature matrix.

    Extends the v1 23-dim matrix with 5 SRPT-aware features that give the
    policy an explicit ordering signal without requiring it to infer rank
    comparisons from raw absolute values:

        23  srpt_rank       normalised rank by est_remaining_steps (0=shortest, 1=longest)
        24  srpt_gap        est_remaining(j) / max(min_est_remaining_in_pending, 1) - 1
                            => 0 for the shortest job, >0 for longer jobs
        25  fifo_rank       normalised rank by arrival_time (0=oldest, 1=newest)
        26  regime_srpt     1.0 when free GPUs exist (SRPT regime), 0.0 when backlogged
        27  pack_alignment  (n_after - target_n) / MAX_ADAPTERS_PER_GPU
                            signed deviation from load-adaptive target depth
    """
    base = feature_matrix(data_plane, cands)  # [N, 23]

    n_gpus = data_plane.num_gpus
    total_pending = len(data_plane.pending)
    free_gpus = sum(1 for g in data_plane.gpus.values() if g.is_empty())

    # Pre-compute SRPT ordering of pending jobs (by ESTIMATED remaining steps,
    # consistent with what the online policy can observe).
    pending_sorted_srpt = sorted(
        data_plane.pending,
        key=lambda j: data_plane.est_remaining_steps(j),
    )
    srpt_rank_map = {j.job_id: i for i, j in enumerate(pending_sorted_srpt)}
    n_pending_total = max(len(data_plane.pending) - 1, 1)

    min_est_rem = (
        data_plane.est_remaining_steps(pending_sorted_srpt[0])
        if pending_sorted_srpt else 1.0
    )
    min_est_rem = max(min_est_rem, 1e-6)

    # Pre-compute FIFO ordering (oldest first).
    pending_sorted_fifo = sorted(data_plane.pending, key=lambda j: j.arrival_time)
    fifo_rank_map = {j.job_id: i for i, j in enumerate(pending_sorted_fifo)}

    regime_srpt = 1.0 if free_gpus > 0 else 0.0

    target_n = _target_n_from_state(total_pending, free_gpus, n_gpus)

    job_lookup = {j.job_id: j for j in data_plane.pending + data_plane.running}

    extra_rows = []
    for c in cands:
        if c.job_id == -1:  # WAIT
            srpt_rank = 0.5
            srpt_gap  = 0.0
            fifo_rank = 0.5
        else:
            job = job_lookup.get(c.job_id)
            if job is None:
                srpt_rank = 0.5
                srpt_gap  = 0.0
                fifo_rank = 0.5
            else:
                srpt_rank = srpt_rank_map.get(job.job_id, 0) / n_pending_total
                srpt_gap  = data_plane.est_remaining_steps(job) / min_est_rem - 1.0
                fifo_rank = fifo_rank_map.get(job.job_id, 0) / n_pending_total

        pack_align = (c.n_after - target_n) / max(cal.MAX_ADAPTERS_PER_GPU, 1)

        extra_rows.append([
            float(srpt_rank),         # 23
            float(min(srpt_gap, 5.0)),# 24  cap at 5 to bound outliers
            float(fifo_rank),         # 25
            regime_srpt,              # 26
            float(pack_align),        # 27
        ])

    extra = np.asarray(extra_rows, dtype=np.float32)
    return np.concatenate([base, extra], axis=1)  # [N, 28]


# v3 normalisation: large slack standing in for "no deadline".
_SLACK_CAP = 5.0
_N_MAX_REF = 8.0


def feature_matrix_v3(data_plane: CLoraDataPlane, cands: List[Candidate]) -> np.ndarray:
    """[num_candidates, CLORA_FEATURE_DIM_V3] = v2 (29) + 5 features that expose
    the asymmetric actuator and deadline state to the policy (plan Flaws 2/3):

        29  slack_ratio      scale-invariant deadline pressure, capped to [-2, 5];
                             a large value == deadline-free / safe.
        30  at_risk          1.0 if slack_ratio < 1.5 (continuous warning region)
        31  draining_state   max(0, mean_realized_depth - load_target) / N_MAX:
                             >0 means the cluster is packed ABOVE what current load
                             justifies, i.e. a down-command is physically locked and
                             the system is draining (no immediate authority).
        32  progress_ratio   steps_done / est_total of this candidate's job (stale-
                             estimate signal: pinned near 1 while still running).
        33  overrun          how far observed progress exceeds the EMA prior, so the
                             policy can distrust slack for under-estimated long jobs.
    """
    base = feature_matrix_v2(data_plane, cands)  # [N, 29]

    n_gpus = data_plane.num_gpus
    total_pending = len(data_plane.pending)
    free_gpus = sum(1 for g in data_plane.gpus.values() if g.is_empty())
    load_target = _target_n_from_state(total_pending, free_gpus, n_gpus)
    busy_depths = [g.n for g in data_plane.gpus.values() if g.n > 0]
    mean_realized = (sum(busy_depths) / len(busy_depths)) if busy_depths else 0.0
    draining_state = max(0.0, mean_realized - load_target) / _N_MAX_REF

    job_lookup = {j.job_id: j for j in data_plane.pending + data_plane.running}

    extra_rows = []
    for c in cands:
        job = job_lookup.get(c.job_id) if c.job_id != -1 else None
        if job is None:
            slack = _SLACK_CAP
            at_risk = 0.0
            progress = 0.0
            ovr = 0.0
        else:
            slack = max(-2.0, min(_SLACK_CAP, data_plane.slack_ratio(job)))
            at_risk = 1.0 if data_plane.slack_ratio(job) < 1.5 else 0.0
            progress = data_plane.progress_ratio(job)
            ovr = min(5.0, data_plane.overrun(job))
        extra_rows.append([
            float(slack),            # 29
            float(at_risk),          # 30
            float(draining_state),   # 31
            float(progress),         # 32
            float(ovr),              # 33
        ])

    extra = np.asarray(extra_rows, dtype=np.float32)
    return np.concatenate([base, extra], axis=1)  # [N, 34]


class RLBSBFControlPlane:
    """
    Control Plane using the CandidateActorCritic network to issue scheduling
    decisions. Wraps the Transformer policy.

    feature_mode: "v1" uses the original 23-dim features (default, compatible
    with bc_init.pt).  "v2" uses 28-dim features with SRPT rank/gap/regime
    signals; requires a model trained on v2 features.
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3,
                 device: Optional[torch.device] = None, feature_mode: str = "v1",
                 architecture: str = "residual"):
        self.device = device or torch.device("cpu")
        self.feature_mode = feature_mode
        fdim = {
            "v1": CLORA_FEATURE_DIM,
            "v2": CLORA_FEATURE_DIM_V2,
            "v3": CLORA_FEATURE_DIM_V3,
        }.get(feature_mode, CLORA_FEATURE_DIM)
        from gpu_rl.policy import CandidateActorCritic  # lazy: only needed for RL path
        self.policy = CandidateActorCritic(
            feature_dim=fdim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            architecture=architecture,
        ).to(self.device)

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict)
        
    def state_dict(self):
        return self.policy.state_dict()
        
    def parameters(self):
        return self.policy.parameters()
        
    def eval(self):
        self.policy.eval()

    def forward(self, features: torch.Tensor):
        return self.policy.forward(features)

    def evaluate_action(self, features: torch.Tensor, actions: torch.Tensor):
        return self.policy.evaluate_action(features, actions)

    def _feature_fn(self):
        return {
            "v1": feature_matrix,
            "v2": feature_matrix_v2,
            "v3": feature_matrix_v3,
        }.get(self.feature_mode, feature_matrix)

    def act(self, data_plane: CLoraDataPlane, queue: PendingJobQueue, deterministic: bool = False) -> Tuple[Optional[Candidate], Optional[int], Optional[float], Optional[float], List[Candidate], Optional[np.ndarray]]:
        """Given the current state, selects a candidate action."""
        cands = CandidateGenerator.build_candidates(data_plane, queue)
        if not cands:
            return None, None, None, None, [], None

        feats = self._feature_fn()(data_plane, cands)
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action_t, logp_t, value_t, _ = self.policy.act(feats_t, deterministic=deterministic)
            
        idx = int(action_t.item())
        cand = cands[idx]
        return cand, idx, float(logp_t.item()), float(value_t.item()), cands, feats
