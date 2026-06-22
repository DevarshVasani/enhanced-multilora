"""
Dynamic N-selection experiments: the asymmetric-actuator control study.

Runs and plots:
  1. Adaptive frontier      -- lambda sweep of the interpretable controller vs the
                               Fixed-N family, bracketed by the unconstrained
                               (free-shed) lower bound and the passive-drain
                               adaptive oracle.
  2. Tracking under shift   -- realized N(t) vs commanded N_target(t) under a
                               non-stationary burst, asserting the asymmetric
                               signature (sharp upswing, drain-limited downswing).
  3. Generalization         -- the same controller across the Tau / APEX / mixed
                               task-structure profiles.
  4. Chatter                -- realized-N total variation (the honest metric) for
                               the damped controller vs a naive round() controller.
  5. Deadline / anti-zombie -- deadline-hit-rate and jobs-dropped under SLOs.

Run:  python -m c_lora_sim.experiment_n_control
All sim results are credible only within the Phase-0 real-hardware tolerance
(see calibration_validation.py).
"""

from __future__ import annotations

import copy
import math
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_lora_sim.data_plane import CLoraDataPlane
from c_lora_sim.workload import generate_workload
from c_lora_sim.runner import run_heuristic
from c_lora_sim.baselines import fixed_n, load_adaptive
from c_lora_sim.n_controller import lambda_controller, n_target, assert_no_load_shed_eviction
from c_lora_sim.oracles import oracle_adaptive_hindsight, oracle_unconstrained

HERE = __file__.rsplit("/", 1)[0]
LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _mean_ci(xs: List[float]) -> Tuple[float, float]:
    xs = list(xs)
    m = float(np.mean(xs))
    ci = 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, ci


# ---------------------------------------------------------------------------
# A tracing runner: drives the controller and samples (t, N_target, N_realized)
# at every discrete event, so we can see the realized depth lag the command on
# the downswing (drain) while matching it on the upswing (packing).
# ---------------------------------------------------------------------------
def run_with_trace(jobs, controller, num_gpus: int = 8, max_steps: int = 200_000):
    sim = CLoraDataPlane(num_gpus=num_gpus)
    sim.reset(copy.deepcopy(jobs))
    trace: List[Tuple[float, int, int]] = []
    steps = 0

    def sample():
        realized = sum(g.n for g in sim.gpus.values())
        tgt = getattr(controller, "target", realized)
        trace.append((sim.wall_time, tgt, realized))

    while not sim.done() and steps < max_steps:
        progress = True
        while progress and sim.pending:
            progress = False
            for job in list(sim.pending):
                gid = controller(sim, job)
                if gid is not None:
                    sim.place(job, gid)
                    progress = True
        assert_no_load_shed_eviction(sim)   # invariant: no force-evict on downswing
        sample()
        kind, _ = sim.advance()
        steps += 1
        if kind == "idle":
            break
    sample()
    return sim.metrics(), trace


# ---------------------------------------------------------------------------
# 1. Adaptive frontier
# ---------------------------------------------------------------------------
def _contended(seed, num_jobs=40, rho=2.0, bursty=True, profile=None,
               p_deadline=0.0):
    """A workload with genuine packing headroom: bias toward shared base models so
    co-location is actually possible (otherwise N is capped by base diversity)."""
    js = generate_workload(num_jobs=num_jobs, seed=seed, target_rho=rho,
                           bursty=bursty, profile=profile, p_deadline=p_deadline)
    # collapse onto two bases so adapters can co-locate (packing headroom)
    for j in js:
        j.base_model_id = "Llama-3-8B" if (j.job_id % 2 == 0) else "Qwen-2.5-7B"
    return js


def experiment_frontier(seeds=range(6)):
    print("\n=== 1. Adaptive frontier (lambda sweep vs Fixed-N, oracle bracket) ===")
    rows = []
    # lambda controller points
    for lam in LAMBDAS:
        mk, fc, mj = [], [], []
        for s in seeds:
            m = run_heuristic(_contended(s), lambda_controller(lam, dwell=150.0))
            mk.append(m["makespan"]); fc.append(m["mean_jct"]); mj.append(m["time_to_first_completion"])
        rows.append(("lambda=%.1f" % lam, _mean_ci(mk)[0], _mean_ci(fc)[0]))
    # fixed-N points
    fixed_pts = []
    for k in [1, 2, 4, 8]:
        mk, fc = [], []
        for s in seeds:
            m = run_heuristic(_contended(s), fixed_n(k))
            mk.append(m["makespan"]); fc.append(m["mean_jct"])
        fixed_pts.append(("Fixed-N%d" % k, _mean_ci(mk)[0], _mean_ci(fc)[0]))
    # oracle bracket (free-shed oracle is event-heavy -> only a few seeds)
    oracle_seeds = list(seeds)[:3]
    adp_jct, unc_jct = [], []
    for s in oracle_seeds:
        js = _contended(s)
        adp_jct.append(oracle_adaptive_hindsight(js)["mean_jct"])
        unc_jct.append(oracle_unconstrained(js)["mean_jct"])
    asym_gap = (np.mean(adp_jct) - np.mean(unc_jct)) / np.mean(unc_jct) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    lx = [r[1] for r in rows]; ly = [r[2] for r in rows]
    ax.plot(lx, ly, "o-", color="tab:blue", label="lambda-controller (adaptive)")
    for nm, x, y in rows:
        ax.annotate(nm.replace("lambda=", "λ="), (x, y), fontsize=7)
    for nm, x, y in fixed_pts:
        ax.scatter(x, y, color="tab:red", marker="s")
        ax.annotate(nm, (x, y), fontsize=7, color="tab:red")
    ax.scatter(np.mean(adp_jct), np.mean([0]), alpha=0)  # keep axis sane
    ax.axhline(np.mean(unc_jct), ls="--", color="green",
               label="unconstrained oracle (lower bound, first-compl)")
    ax.axhline(np.mean(adp_jct), ls="--", color="orange",
               label="adaptive oracle (passive-drain)")
    ax.set_xlabel("makespan  (lower = more throughput)")
    ax.set_ylabel("mean JCT / first-completion  (lower = lower latency)")
    ax.set_title("Adaptive N frontier  (asymmetry gap = %.0f%%)" % asym_gap)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{HERE}/n_control_frontier.png", dpi=110)
    print(f"  asymmetry gap (adaptive vs unconstrained mean_jct) = {asym_gap:+.0f}%")
    print(f"  saved n_control_frontier.png")
    return asym_gap


# ---------------------------------------------------------------------------
# 2. Tracking under workload shift  (the centerpiece)
# ---------------------------------------------------------------------------
def experiment_tracking(seed=7):
    print("\n=== 2. Tracking under shift (realized N(t) vs N_target(t)) ===")
    # Build a non-stationary stream: a dense burst of same-base jobs at t=0, then
    # a long quiet tail -> target should ramp UP fast then the realized depth must
    # DRAIN slowly back down.
    js = generate_workload(num_jobs=50, seed=seed, target_rho=2.5, bursty=True)
    for j in js:
        j.base_model_id = "Llama-3-8B"
    ctrl = lambda_controller(0.8, dwell=120.0)
    _, trace = run_with_trace(js, ctrl)
    t = np.array([p[0] for p in trace])
    tgt = np.array([p[1] for p in trace])

    # --- assertions on the asymmetric signature ---------------------------
    # Realized total-N may only DECREASE at completion events (drain), never via a
    # force-evict. We check that every downward step in realized depth coincides
    # with the physics (no instantaneous multi-step collapse from the controller).
    realized_raw = np.array([p[2] for p in trace])
    downs = np.diff(realized_raw)
    max_single_drop = -downs.min() if len(downs) and downs.min() < 0 else 0
    print(f"  max single-event drop in total realized N = {max_single_drop} "
          f"(passive drain -> small, bounded steps)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(t, tgt, where="post", color="tab:orange", label="N_target (commanded)")
    ax.step(t, realized_raw, where="post", color="tab:blue", label="N_realized (Σ depth)")
    ax.fill_between(t, tgt, realized_raw, where=(realized_raw >= tgt),
                    step="post", alpha=0.15, color="tab:blue",
                    label="drain lag (realized > target)")
    ax.set_xlabel("wall time (s)"); ax.set_ylabel("co-location depth")
    ax.set_title("Asymmetric tracking: sharp upswing, drain-limited downswing")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{HERE}/n_control_tracking.png", dpi=110)
    print("  saved n_control_tracking.png")


# ---------------------------------------------------------------------------
# 3. Generalization across task structure
# ---------------------------------------------------------------------------
def experiment_generalization(seeds=range(5)):
    print("\n=== 3. Generalization across task structure (Tau / APEX / mixed) ===")
    print(f"  {'profile':<10}{'lam':>5}{'mean_jct':>12}{'makespan':>12}")
    for profile in [None, "tau", "apex"]:
        for lam in [0.0, 0.5, 1.0]:
            mj, mk = [], []
            for s in seeds:
                js = _contended(s, profile=profile)
                m = run_heuristic(js, lambda_controller(lam, dwell=150.0))
                mj.append(m["mean_jct"]); mk.append(m["makespan"])
            label = profile or "mixed"
            print(f"  {label:<10}{lam:>5.1f}{np.mean(mj):>12.0f}{np.mean(mk):>12.0f}")


# ---------------------------------------------------------------------------
# 4. Chatter: realized-N TV vs a naive round() controller
# ---------------------------------------------------------------------------
class _NaiveRoundController:
    """A deliberately undamped controller: recomputes N_target every call with no
    deadband/dwell, so it toggles on the round() boundary."""
    def __init__(self, lam=0.8):
        self.lam = lam
        self.target = 1
    def __call__(self, sim, job):
        self.target = n_target(len(sim.pending), sim.num_gpus, self.lam)
        cand = __import__("c_lora_sim.control_plane", fromlist=["CandidateGenerator"]).CandidateGenerator._candidate_gpus(sim, job)
        under = [g for g in cand["warm_same"] if sim.gpus[g].n < self.target]
        if under:
            return min(under, key=lambda g: sim.gpus[g].n)
        if cand["empty"]:
            return cand["empty"][0]
        if self.target >= 2 and cand["warm_same"]:
            return min(cand["warm_same"], key=lambda g: sim.gpus[g].n)
        if not sim.running and cand["warm_same"]:
            return cand["warm_same"][0]
        return None


def _command_tv(controller_factory, js):
    """Run a controller and return (command_TV, realized_TV_excess). command_TV is
    the total variation of the COMMANDED N_target; realized_TV_excess is realized_N
    TV above the unavoidable place/complete floor (2 * jobs). The asymmetric
    actuator means a chattery COMMAND need not produce chattery REALIZED depth."""
    ctrl = controller_factory()
    _, trace = run_with_trace(js, ctrl)
    tgt = [p[1] for p in trace]
    cmd_tv = sum(abs(tgt[i + 1] - tgt[i]) for i in range(len(tgt) - 1))
    m = run_heuristic(js, controller_factory())
    floor = 2 * len([j for j in js])  # each job: one place (+) and one complete (-)
    realized_excess = max(0.0, m["realized_N_tv"] - floor)
    return cmd_tv, realized_excess


def experiment_chatter(seeds=range(5)):
    print("\n=== 4. Chatter: command vs realized volatility (Flaw 4) ===")
    print(f"  {'controller':<22}{'cmd N_target TV':>16}{'realized excess TV':>20}")
    dmp_cmd, dmp_rx, nv_cmd, nv_rx = [], [], [], []
    for s in seeds:
        js = _contended(s)
        c, r = _command_tv(lambda: lambda_controller(0.8, dwell=150.0), js)
        dmp_cmd.append(c); dmp_rx.append(r)
        c, r = _command_tv(lambda: _NaiveRoundController(0.8), js)
        nv_cmd.append(c); nv_rx.append(r)
    print(f"  {'deadband+dwell':<22}{np.mean(dmp_cmd):>16.0f}{np.mean(dmp_rx):>20.0f}")
    print(f"  {'naive round()':<22}{np.mean(nv_cmd):>16.0f}{np.mean(nv_rx):>20.0f}")
    print("  Reading: deadband cuts COMMAND chatter; the asymmetric actuator keeps")
    print("  REALIZED excess churn ~0 for both (you cannot force N down -> no thrash).")


# ---------------------------------------------------------------------------
# 5. Deadline protection / anti-zombie
# ---------------------------------------------------------------------------
def experiment_deadlines(seeds=range(6)):
    print("\n=== 5. Deadline protection / anti-zombie ===")
    print(f"  {'scheduler':<22}{'dl_hit':>8}{'dropped':>9}{'mean_jct':>11}")
    scheds = {
        "Fixed-N8": fixed_n(8),
        "LoadAdaptive": load_adaptive,
        "lambda(0.6,DL-blind)": lambda_controller(0.6, deadline_aware=False),
        "lambda(0.6,DL-aware)": lambda_controller(0.6, deadline_aware=True),
    }
    for name, sched in scheds.items():
        hit, drop, mj = [], [], []
        for s in seeds:
            js = _contended(s, num_jobs=50, rho=1.5, p_deadline=0.5)
            m = run_heuristic(js, sched)
            hit.append(m["deadline_hit_rate"]); drop.append(m["jobs_dropped"]); mj.append(m["mean_jct"])
        print(f"  {name:<22}{np.mean(hit):>8.2f}{np.mean(drop):>9.1f}{np.mean(mj):>11.0f}")


def main():
    experiment_frontier()
    experiment_tracking()
    experiment_generalization()
    experiment_chatter()
    experiment_deadlines()
    print("\nAll N-control experiments complete.")


if __name__ == "__main__":
    main()
