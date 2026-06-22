"""
Drives a scheduler through the calibrated C-LoRA simulator.

`run_heuristic` is used by every baseline; the RL policy uses its own rollout
(see ppo_clora.py) but on the SAME `CLoraSim`, so the comparison is fair.
"""

from __future__ import annotations

import copy
from typing import Dict, List

from c_lora_sim.baselines import Scheduler
from c_lora_sim.data_plane import CLoraDataPlane, LoraJob


def _placement_phase(sim: CLoraDataPlane, scheduler: Scheduler) -> None:
    """Let the scheduler place as many pending jobs as it wants, right now."""
    progress = True
    while progress and sim.pending:
        progress = False
        for job in list(sim.pending):
            gpu_id = scheduler(sim, job)
            if gpu_id is not None:
                sim.place(job, gpu_id)
                progress = True


def run_heuristic(
    jobs: List[LoraJob],
    scheduler: Scheduler,
    num_gpus: int = 8,
    max_steps: int = 1_000_000,
    step_time_cv: float = 0.0,
    cold_start_cv: float = 0.0,
    noise_seed: int | None = None,
    clairvoyant: bool = False,
) -> Dict[str, float]:
    sim = CLoraDataPlane(num_gpus=num_gpus, step_time_cv=step_time_cv,
                         cold_start_cv=cold_start_cv, noise_seed=noise_seed,
                         clairvoyant=clairvoyant)
    sim.reset(copy.deepcopy(jobs))
    steps = 0
    while not sim.done() and steps < max_steps:
        _placement_phase(sim, scheduler)
        kind, _ = sim.advance()
        steps += 1
        if kind == "idle":           # timeline empty: nothing left to do
            break
    return sim.metrics()
