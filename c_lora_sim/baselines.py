"""
Heuristic baseline schedulers for continuous multi-LoRA training.

Each scheduler is a function `(sim, job) -> gpu_id | None` that decides where to
place ONE pending job, given the current cluster state. Returning None means
"don't place this job right now" (wait for capacity / a better moment).

The runner in `runner.py` drives every scheduler -- heuristic or RL -- through
the exact same discrete-event physics, so differences in the reported metrics
come only from placement quality.

Baselines, from naive to the article's implied practice:

  fifo          : first GPU with any room (locality-blind). The strawman.
  best_fit      : same-base GPU with the MOST adapters that still has room
                  (packs tight) else emptiest GPU. Throughput-greedy.
  locality_greedy: prefer same-base GPU under a soft cap N*, else a fresh GPU.
                  This is the article's hand-tuned static policy.
  fixed_n(k)    : locality-aware but hard-capped at k co-located adapters.
                  fixed_n(2) == the article's hand-picked latency operating
                  point; fixed_n(8) == its max-throughput point.
"""

from __future__ import annotations

from typing import Callable, Optional

from c_lora_sim.data_plane import CLoraDataPlane, LoraJob
from c_lora_sim.control_plane import CandidateGenerator

Scheduler = Callable[[CLoraDataPlane, LoraJob], Optional[int]]


def fifo(sim: CLoraDataPlane, job: LoraJob) -> Optional[int]:
    cand = CandidateGenerator._candidate_gpus(sim, job)
    # locality-blind: first warm-same, then first empty, then repurpose
    for key in ("warm_same", "empty", "repurpose"):
        if cand[key]:
            return cand[key][0]
    return None


def best_fit(sim: CLoraDataPlane, job: LoraJob) -> Optional[int]:
    cand = CandidateGenerator._candidate_gpus(sim, job)
    if cand["warm_same"]:
        # tightest pack: the warm GPU with the most adapters that still has room
        return max(cand["warm_same"], key=lambda g: sim.gpus[g].n)
    if cand["empty"]:
        return cand["empty"][0]
    if cand["repurpose"]:
        return cand["repurpose"][0]
    return None


def locality_greedy(sim: CLoraDataPlane, job: LoraJob, soft_cap: int = 8) -> Optional[int]:
    """Article's implied static policy: pack onto a same-base GPU while it's
    under the soft cap; otherwise spin up a fresh GPU; only as a last resort
    repurpose a GPU warm for another base (paying the cold start)."""
    cand = CandidateGenerator._candidate_gpus(sim, job)
    under_cap = [g for g in cand["warm_same"] if sim.gpus[g].n < soft_cap]
    if under_cap:
        # least-loaded among under-cap same-base GPUs => spreads load, keeps N low
        return min(under_cap, key=lambda g: sim.gpus[g].n)
    if cand["empty"]:
        return cand["empty"][0]
    if cand["warm_same"]:          # over soft cap but no empty GPU: pack anyway
        return min(cand["warm_same"], key=lambda g: sim.gpus[g].n)
    if cand["repurpose"]:
        return cand["repurpose"][0]
    return None


def fixed_n(k: int) -> Scheduler:
    """Locality-aware scheduler hard-capped at k co-located adapters per GPU."""

    def sched(sim: CLoraDataPlane, job: LoraJob) -> Optional[int]:
        cand = CandidateGenerator._candidate_gpus(sim, job)
        under = [g for g in cand["warm_same"] if sim.gpus[g].n < k]
        if under:
            return min(under, key=lambda g: sim.gpus[g].n)
        if cand["empty"]:
            return cand["empty"][0]
        if cand["repurpose"]:
            return cand["repurpose"][0]
        # all same-base GPUs at cap and no free GPU: must wait, unless nothing
        # is running at all (avoid deadlock) -> then relax the cap.
        if not sim.running and cand["warm_same"]:
            return min(cand["warm_same"], key=lambda g: sim.gpus[g].n)
        return None

    return sched


def load_adaptive(sim: CLoraDataPlane, job: LoraJob) -> Optional[int]:
    """State-dependent scheduler: chooses target N from current cluster state.

    Same logic as the expert teacher (expert.py), exposed as a heuristic baseline
    so the evaluation table contains a strong adaptive comparator that PPO must
    beat purely through learned generalisation.
    """
    n_pending = len(sim.pending)
    n_free = sum(1 for g in sim.gpus.values() if g.is_empty())
    n_gpus = sim.num_gpus

    # target N: same rule as expert._target_n
    if n_free >= n_gpus // 2 and n_pending <= n_gpus // 2:
        target = 1
    elif n_free > 0 and n_pending < n_gpus:
        target = 2
    elif n_free > 0:
        target = 4
    else:
        ratio = n_pending / max(1, n_gpus)
        if ratio < 1.0:
            target = 4
        elif ratio < 2.0:
            target = 6
        else:
            target = 8

    cand = CandidateGenerator._candidate_gpus(sim, job)
    short = sim.est_total_steps(job) <= 40   # non-clairvoyant length estimate

    under = [g for g in cand["warm_same"] if sim.gpus[g].n < target]
    if under:
        # size-segregate: short jobs to least-loaded, long jobs to most-loaded
        return min(under, key=lambda g: sim.gpus[g].n) if short else max(under, key=lambda g: sim.gpus[g].n)
    if cand["empty"]:
        return cand["empty"][0]
    if cand["warm_same"]:          # over target but no free GPU: pack anyway
        return min(cand["warm_same"], key=lambda g: sim.gpus[g].n)
    if cand["repurpose"]:
        return cand["repurpose"][0]
    if not sim.running and cand["warm_same"]:
        return cand["warm_same"][0]
    return None


BASELINES = {
    "FIFO": fifo,
    "BestFit": best_fit,
    "LocalityGreedy": locality_greedy,
    "Fixed-N1": fixed_n(1),
    "Fixed-N2": fixed_n(2),
    "Fixed-N8": fixed_n(8),
    "LoadAdaptive": load_adaptive,
}

# No-multiplexing reference (one adapter per GPU at a time). Used as the
# denominator for the article-style multiplexing speedup, NOT a competitor.
NO_MULTIPLEX_REF = fixed_n(1)


# ---------------------------------------------------------------------------
# SRPT-Preemptive: a heuristic that adds preemption on top of load_adaptive.
# This is the honest strong non-RL preemptive baseline: the RL policy must
# beat it to claim that *learning* (not just access to eviction) is the gain.
# ---------------------------------------------------------------------------

def run_srpt_preemptive(
    jobs,
    num_gpus: int = 8,
    max_steps: int = 1_000_000,
    clairvoyant: bool = False,
):
    """Run the SRPT-Preemptive heuristic through the calibrated simulator.

    At each decision point:
      1. Preempt the longest-running job if a pending job is ≥4× shorter AND
         no free GPU is available (mirrors the expert's guard).
      2. Otherwise place with load_adaptive.
    """
    import copy
    sim = CLoraDataPlane(num_gpus=num_gpus, clairvoyant=clairvoyant)
    sim.reset(copy.deepcopy(jobs))
    steps = 0

    while not sim.done() and steps < max_steps:
        # -- preemption check ------------------------------------------------
        if sim.running and sim.pending:
            n_free = sum(1 for g in sim.gpus.values() if g.is_empty())
            longest_r = max(sim.running, key=lambda j: sim.est_remaining_steps(j))
            if longest_r.steps_done > sim._ema_steps * 1.5 and n_free == 0:
                sim.evict(longest_r)
                steps += 1
                continue

        # -- placement phase -------------------------------------------------
        placed_any = True
        while placed_any and sim.pending:
            placed_any = False
            for job in list(sim.pending):
                gpu_id = load_adaptive(sim, job)
                if gpu_id is not None:
                    sim.place(job, gpu_id)
                    placed_any = True

        kind, _ = sim.advance()
        steps += 1
        if kind == "idle":
            break

    return sim.metrics()
