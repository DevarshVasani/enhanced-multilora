"""
Interpretable feedback controller for the N-selection (co-location depth) problem.

This recasts the article's "what is the optimal N, and can we adapt it online"
question (problem P1) as a feedback control law. The control input is the target
co-location depth N_target(t); the plant is the calibrated concurrency physics
(calibration.py). The defining constraint is ASYMMETRY:

  * Up-moves are free and instantaneous: just place another adapter on the GPU.
  * Down-moves are NOT: a running gradient step cannot be aborted mid
    forward/backward. Realized N therefore falls ONLY by passive drain (an adapter
    naturally finishing its step block). This controller is PLACEMENT-ONLY and is
    forbidden from force-evicting to reach a lower N_target (the assert guard at
    the bottom encodes the invariant).

The single knob `lam` in [0, 1] traces an adaptive Pareto frontier:
  * lam = 0  -> N_target == 1 always  (spread; minimum first-completion latency,
               equivalent to Fixed-N1).
  * lam = 1  -> packs toward N_MAX under load (maximum throughput, ~Fixed-N8).
Intermediate lam interpolates, but ADAPTIVELY: it only packs deep when the live
backlog justifies it, so it dominates any static Fixed-N(k) across load regimes.

Chatter is damped with an asymmetric deadband + minimum dwell time on commanded
changes (see ARCHITECTURE / plan Flaw 4): N_target only moves when the load signal
clears a threshold by a margin and a minimum wall-time has elapsed since the last
change.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from c_lora_sim.data_plane import CLoraDataPlane, LoraJob
from c_lora_sim.control_plane import CandidateGenerator

Scheduler = Callable[[CLoraDataPlane, LoraJob], Optional[int]]

# Measured physics is anchored up to N=8; treat that as the packing ceiling.
N_MAX = 8
# Gain mapping backlog saturation -> depth. Kept gentle so intermediate lam
# yields intermediate operating points (rather than saturating to N_MAX at the
# first burst); lam=0 stays at 1, lam=1 reaches N_MAX under a deep backlog.
KAPPA = 1.5
# A job whose slack_ratio drops below this is "at risk": spread it to protect the
# SLO rather than packing it behind long jobs.
SLACK_DANGER = 1.5


def n_target(n_pending: int, n_gpus: int, lam: float) -> int:
    """Pure feedback law: target co-location depth from backlog saturation.

        N_target = clip( 1 + round(lam * KAPPA * rho_eff), 1, N_MAX )

    where rho_eff = n_pending / n_gpus is the live backlog pressure. Exposed
    separately so the report/plots can draw the commanded law alongside the
    realized trajectory.
    """
    rho_eff = n_pending / max(1, n_gpus)
    return int(max(1, min(N_MAX, round(1 + lam * KAPPA * rho_eff))))


class _LambdaController:
    """Stateful controller: holds the deadband/dwell state for one episode and
    resets automatically when it sees a fresh simulator (so a single instance can
    be reused across evaluation seeds without state leaking)."""

    def __init__(self, lam: float, theta_up: float, dwell: float,
                 deadline_aware: bool):
        self.lam = lam
        self.theta_up = theta_up      # margin (in depth units) required to change
        self.dwell = dwell            # min wall-seconds between commanded changes
        self.deadline_aware = deadline_aware
        self._sim_id: Optional[int] = None
        self.target = 1
        self._last_change_t = 0.0
        # trajectory log (wall_time, N_target, realized_total_N) for plotting
        self.trace: list = []

    def _maybe_reset(self, sim: CLoraDataPlane) -> None:
        if id(sim) != self._sim_id:
            self._sim_id = id(sim)
            self.target = 1
            self._last_change_t = sim.wall_time
            self.trace = []

    def _update_target(self, sim: CLoraDataPlane) -> int:
        raw = n_target(len(sim.pending), sim.num_gpus, self.lam)
        now = sim.wall_time
        # Deadband + dwell: only accept a change once it clears the margin AND the
        # minimum dwell has elapsed. Prevents high-frequency toggling on the
        # round() boundary in bursty regimes.
        if abs(raw - self.target) >= self.theta_up and (now - self._last_change_t) >= self.dwell:
            self.target = raw
            self._last_change_t = now
        realized = sum(g.n for g in sim.gpus.values())
        self.trace.append((now, self.target, realized))
        return self.target

    def __call__(self, sim: CLoraDataPlane, job: LoraJob) -> Optional[int]:
        self._maybe_reset(sim)
        target = self._update_target(sim)
        return place_to_target(sim, job, target, self.deadline_aware)


def place_to_target(sim: CLoraDataPlane, job: LoraJob, target: int,
                    deadline_aware: bool = True):
    """Shared placement rule for any depth-target N-controller (lambda, AIMD,
    threshold-hysteresis): pack to `target`, size-segregate, and in the latency
    regime WAIT rather than exceed target. This is the placement-only, never-evict
    half; only the TARGET law differs between controllers."""
    cand = CandidateGenerator._candidate_gpus(sim, job)
    # Deadline override: an at-risk job is spread to the lowest available depth.
    at_risk = deadline_aware and sim.slack_ratio(job) < SLACK_DANGER
    eff_target = 1 if at_risk else target

    short = sim.est_total_steps(job) <= 40
    under = [g for g in cand["warm_same"] if sim.gpus[g].n < eff_target]
    if under:
        if short or at_risk:
            return min(under, key=lambda g: sim.gpus[g].n)
        return max(under, key=lambda g: sim.gpus[g].n)
    if cand["empty"]:
        return cand["empty"][0]

    # Over target with no free GPU: pack (packing regime) vs wait (latency regime).
    packing_regime = eff_target >= 2
    if not sim.running:
        if cand["warm_same"]:
            return min(cand["warm_same"], key=lambda g: sim.gpus[g].n)
        if cand["repurpose"]:
            return cand["repurpose"][0]
        return None
    if at_risk:
        return None
    if packing_regime:
        if cand["warm_same"]:
            return min(cand["warm_same"], key=lambda g: sim.gpus[g].n)
        if cand["repurpose"]:
            return cand["repurpose"][0]
    return None


def lambda_controller(lam: float, theta_up: float = 1.0, dwell: float = 300.0,
                      deadline_aware: bool = True) -> "_LambdaController":
    """Factory for the interpretable N-controller. `lam` in [0,1] sets the
    throughput/latency operating point; `theta_up`/`dwell` damp chatter."""
    return _LambdaController(lam, theta_up, dwell, deadline_aware)


def assert_no_load_shed_eviction(sim: CLoraDataPlane) -> None:
    """Invariant check (plan Flaw 1 / verification 3): the interpretable
    controller is placement-only, so NO running adapter should ever carry the
    `force_evicted` flag — that flag is set only by the rescue/SRPT path, never by
    load-shedding to a lower N_target."""
    assert not any(getattr(j, "force_evicted", False) for j in sim.running), (
        "Physical-law violation: a job was force-evicted under the placement-only "
        "N-controller (load-shed eviction on the downswing is forbidden)."
    )
