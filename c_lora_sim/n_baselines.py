"""
Standard feedback-control baselines for N-selection, to situate the lambda-controller
against known engineering solutions (reviewer Reason 5).

  AIMDController          -- Additive-Increase / Multiplicative-Decrease, the TCP
                             congestion-control law. The canonical feedback rule for
                             an asymmetric scale-up/scale-down resource problem.
  ThresholdHysteresis     -- Two-threshold hysteretic autoscaler (the structure
                             Kubernetes HPA / classic autoscaling uses): step the
                             target toward N_MAX above an upper backlog threshold,
                             toward 1 below a lower one, hold in the deadband.

Both reuse the lambda-controller's placement rule (`place_to_target`) and the same
asymmetric actuator (target gates placement; realized N falls only by drain), so the
ONLY thing that differs is the target LAW. This makes the comparison about the
control law, not the mechanics.

NOTE on AIMD fit: TCP-AIMD is designed for the OPPOSITE asymmetry -- cheap to shrink
the window, careful (additive) to grow it, fast (multiplicative) to back off. Our
actuator is the reverse: growing N is instant/cheap, shrinking is slow/impossible.
So AIMD's slow additive increase under-packs during bursts (it cannot ramp N as fast
as the backlog demands), which is exactly what we test below.
"""

from __future__ import annotations

from typing import Optional

from c_lora_sim.data_plane import CLoraDataPlane, LoraJob
from c_lora_sim.n_controller import place_to_target, N_MAX


class _StatefulController:
    def __init__(self, deadline_aware: bool = True):
        self.deadline_aware = deadline_aware
        self._sim_id: Optional[int] = None
        self.target = 1.0
        self._last_t = 0.0
        self.trace = []

    def _maybe_reset(self, sim):
        if id(sim) != self._sim_id:
            self._sim_id = id(sim)
            self.target = 1.0
            self._last_t = sim.wall_time
            self.trace = []


class AIMDController(_StatefulController):
    """target += ai when backlog is high; target *= md when backlog is low.
    Updated once per event (guarded by wall-time) so multiple placements in one
    decision phase do not over-step the law."""
    def __init__(self, ai: float = 1.0, md: float = 0.5,
                 hi: float = 1.0, lo: float = 0.5, deadline_aware: bool = True):
        super().__init__(deadline_aware)
        self.ai, self.md, self.hi, self.lo = ai, md, hi, lo

    def __call__(self, sim: CLoraDataPlane, job: LoraJob):
        self._maybe_reset(sim)
        if sim.wall_time > self._last_t:          # one update per event
            self._last_t = sim.wall_time
            rho = len(sim.pending) / max(1, sim.num_gpus)
            if rho > self.hi:
                self.target = min(N_MAX, self.target + self.ai)
            elif rho < self.lo:
                self.target = max(1.0, self.target * self.md)
        self.trace.append((sim.wall_time, self.target,
                           sum(g.n for g in sim.gpus.values())))
        return place_to_target(sim, job, int(round(self.target)), self.deadline_aware)


class ThresholdHysteresis(_StatefulController):
    """Two-threshold hysteretic autoscaler. Above `up` backlog, step the target up
    toward N_MAX; below `down`, step down toward 1; hold in the deadband (down<up)."""
    def __init__(self, up: float = 1.0, down: float = 0.4, step: int = 2,
                 deadline_aware: bool = True):
        super().__init__(deadline_aware)
        self.up, self.down, self.step = up, down, step

    def __call__(self, sim: CLoraDataPlane, job: LoraJob):
        self._maybe_reset(sim)
        if sim.wall_time > self._last_t:
            self._last_t = sim.wall_time
            rho = len(sim.pending) / max(1, sim.num_gpus)
            if rho > self.up:
                self.target = min(N_MAX, self.target + self.step)
            elif rho < self.down:
                self.target = max(1.0, self.target - self.step)
        self.trace.append((sim.wall_time, self.target,
                           sum(g.n for g in sim.gpus.values())))
        return place_to_target(sim, job, int(round(self.target)), self.deadline_aware)


def aimd_controller(**kw):
    return AIMDController(**kw)


def threshold_hysteresis(**kw):
    return ThresholdHysteresis(**kw)
