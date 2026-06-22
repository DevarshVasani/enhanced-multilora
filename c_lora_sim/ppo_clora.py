"""
PPO training wrapper for the C-LoRA scheduler.

This module orchestrates the episode rollout using the explicitly separated
Control Plane (RLBSBFControlPlane) and Data Plane (CLoraDataPlane).
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch

from c_lora_sim.data_plane import CLoraDataPlane, LoraJob
from c_lora_sim.control_plane import RLBSBFControlPlane, PendingJobQueue
try:
    from gpu_rl.ppo import RolloutStep
except ModuleNotFoundError:
    RolloutStep = None  # gpu_rl not bundled; RL path unavailable


def make_policy(hidden_dim: int = 128, num_layers: int = 3,
                device: Optional[torch.device] = None,
                architecture: str = "residual",
                feature_mode: str = "v1") -> RLBSBFControlPlane:
    return RLBSBFControlPlane(hidden_dim=hidden_dim, num_layers=num_layers,
                              device=device, architecture=architecture,
                              feature_mode=feature_mode)


def _advance_to_decision(data_plane: CLoraDataPlane, reward_scale: float,
                         tail_coef: float = 0.0,
                         deadline_shaping_coef: float = 0.0,
                         miss_penalty: float = 0.0) -> Tuple[float, bool]:
    """Pop events until something changes the decision or the sim ends.

    Reward = Little's-law surrogate: -(pending + running) * dt / Z.
    This is the exact dense proxy for mean JCT (∫ n(t) dt / |J|). It does NOT
    penalise useful running work artificially, and the gradient correctly points
    toward spreading when GPUs are idle and packing when the queue is deep.
    Optional tail_coef adds -beta * Σ_pending age^2 * dt for tail-latency mode.

    Deadline terms (plan Flaws 2/3):
      * deadline_shaping_coef (alpha): a SMOOTH, continuous warning potential
        -alpha * Σ_{running} max(0, 1 - slack_ratio_i)^2 * dt, which supplies a
        non-zero gradient pulling the policy off the deadline edge BEFORE a job is
        lost (a one-time miss penalty alone gives no gradient until it's too late).
      * miss_penalty (mu_miss): a fixed one-time cost charged when a job is
        cancelled (deadline became unrecoverable). Bounded, so the agent cannot be
        tempted to 'give up' once the penalty is paid.
    """
    reward = 0.0
    while True:
        n_in_system = len(data_plane.pending) + len(data_plane.running)
        n_dropped_before = len(data_plane.dropped)

        t_before = data_plane.wall_time
        kind, _ = data_plane.advance()
        dt = data_plane.wall_time - t_before

        reward -= n_in_system * dt / reward_scale
        if tail_coef > 0.0:
            age_sq = sum(
                ((data_plane.wall_time - j.arrival_time) / reward_scale) ** 2
                for j in data_plane.pending
            )
            reward -= tail_coef * age_sq * dt
        if deadline_shaping_coef > 0.0:
            # continuous deadline-proximity potential over jobs still in flight
            at_risk = 0.0
            for j in data_plane.running:
                if j.deadline >= 0:
                    sr = data_plane.slack_ratio(j)
                    at_risk += max(0.0, 1.0 - sr) ** 2
            reward -= deadline_shaping_coef * at_risk * dt
        if miss_penalty > 0.0:
            newly_dropped = len(data_plane.dropped) - n_dropped_before
            if newly_dropped > 0:
                reward -= miss_penalty * newly_dropped

        if data_plane.done():
            return reward, True
        if kind in ("arrival", "complete", "idle"):
            if kind == "idle" and not data_plane.pending:
                return reward, True
            return reward, False
        # 'ready' / 'stale' -> keep draining; nothing to decide on
        if len(data_plane.timeline) == 0:
            return reward, True


def run_episode(
    control_plane: RLBSBFControlPlane,
    jobs: List[LoraJob],
    num_gpus: int = 8,
    deterministic: bool = False,
    reward_scale: float = 50_000.0,
    cold_penalty_coef: float = 0.0,
    tail_coef: float = 0.0,
    deadline_shaping_coef: float = 0.0,
    miss_penalty: float = 0.0,
    churn_coef: float = 0.0,
    evict_coef: float = 0.0,
    max_steps: int = 20_000,
    device: Optional[torch.device] = None,
    collect: bool = True,
):
    data_plane = CLoraDataPlane(num_gpus=num_gpus)
    data_plane.reset(copy.deepcopy(jobs))
    queue = PendingJobQueue(data_plane)

    def _wait():
        return _advance_to_decision(data_plane, reward_scale, tail_coef,
                                    deadline_shaping_coef, miss_penalty)

    steps: List[RolloutStep] = []
    done = False
    n_steps = 0
    while not done and n_steps < max_steps:
        cand, idx, logp_t, value_t, _, feats = control_plane.act(data_plane, queue, deterministic=deterministic)

        if not cand:
            _, done = _wait()
            continue

        if cand.mode == "wait":
            reward, done = _wait()
        elif cand.mode == "migrate":
            job = next(j for j in data_plane.running if j.job_id == cand.job_id)
            cold = data_plane.migrate(job, cand.gpu_id)
            reward = -cold_penalty_coef * (cold / reward_scale)
            done = False
        elif cand.mode == "evict":
            # Rescue/SRPT preemption only. Eviction is NO LONGER free: the in-flight
            # partial step is lost, charged here via the realised evict cost.
            job = next((j for j in data_plane.running if j.job_id == cand.job_id), None)
            wasted = data_plane.evict(job) if job is not None else 0.0
            reward = -evict_coef * (wasted / reward_scale)
            done = False
        else:
            job = next(j for j in data_plane.pending if j.job_id == cand.job_id)
            # realised-N churn penalty (Flaw 4): packing onto an already-occupied
            # GPU is the only agent-controllable realised UP-move; penalise it so
            # the policy packs only when the flow-time gain justifies the (later
            # un-droppable) depth. Opening a fresh GPU (spreading) is not penalised.
            depth_after_pack = data_plane.gpus[cand.gpu_id].n + 1
            churn = 1.0 if depth_after_pack > 1 else 0.0
            cold = data_plane.place(job, cand.gpu_id)
            reward = -cold_penalty_coef * (cold / reward_scale) - churn_coef * churn
            done = False

        if collect and feats is not None and idx is not None:
            steps.append(RolloutStep(
                features=feats,
                action=idx,
                log_prob=logp_t,
                value=value_t,
                reward=reward,
                done=done,
            ))
            
        n_steps += 1

    return steps, data_plane.metrics(), data_plane


@torch.no_grad()
def evaluate(control_plane: RLBSBFControlPlane, jobs: List[LoraJob], num_gpus=8,
             device=None, max_steps: int = 20_000):
    _, metrics, _ = run_episode(
        control_plane, jobs, num_gpus=num_gpus, deterministic=True, collect=False,
        device=device, max_steps=max_steps,
    )
    return metrics
