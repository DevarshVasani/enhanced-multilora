"""
Evaluate the trained C-LoRA RL scheduler against all baselines and emit the
credible result tables + plots.

Evaluation design (ARCHITECTURE.md Section 5 success criteria):

  Metric set: mean_jct, makespan, p95_jct, p99_jct, total_cold_start,
              cluster_throughput_x, mux_speedup -- all mean ± 95% CI over seeds.

  Load sweep: target_rho in {0.5, 0.75, 0.9, 1.1, 1.5} relative to N=1 capacity,
              both stationary Poisson and bursty MMPP.

  Comparators: FIFO, BestFit, LocalityGreedy, Fixed-N1, Fixed-N2, Fixed-N8,
               LoadAdaptive (the BC teacher), RL (ours), ORACLE
               where ORACLE = min over {Fixed-N1..N8} chosen *per episode* with
               hindsight -- the hardest static comparator.

  Success criteria (pre-registered in ARCHITECTURE.md):
    1. RL < ORACLE on mean_jct at critical/bursty regimes (≥ 0% margin)
    2. RL < LoadAdaptive on mean_jct at ≥ one rho with non-overlapping CIs
    3. RL bounded backlog in bursts where Fixed-N1/N2 diverge
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from c_lora_sim.baselines import BASELINES, NO_MULTIPLEX_REF, run_srpt_preemptive
from c_lora_sim.ppo_clora import make_policy, run_episode
from c_lora_sim.runner import run_heuristic
from c_lora_sim.workload import generate_workload
from c_lora_sim.n_controller import lambda_controller
from c_lora_sim.oracles import oracle_adaptive_hindsight, oracle_unconstrained

METRICS = ["mean_jct", "makespan", "time_to_first_completion", "p95_jct", "p99_jct",
           "total_cold_start", "cluster_throughput_x", "deadline_hit_rate",
           "jobs_dropped", "realized_N_tv"]

# The default operating point for the interpretable controller in the table.
CONTROLLER_LAMBDA = 0.6

# Fixed-N variants used for the per-episode oracle
ORACLE_NS = [1, 2, 3, 4, 5, 6, 7, 8]


def ci95(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(len(xs))


def load_policy(path: str, device, architecture: str | None = None):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    # Auto-detect architecture from the checkpoint's tensor names unless overridden.
    # GAT checkpoints carry a graph-attention context query; the residual encoder
    # has a LayerNorm-projected input stack instead.
    if architecture is None:
        architecture = "gat" if any("ctx_query" in k or "gat" in k.lower() for k in sd) else "residual"
    pol = make_policy(hidden_dim=ck["hidden_dim"], num_layers=ck["layers"],
                      device=device, architecture=architecture)
    pol.load_state_dict(sd)
    pol.eval()
    return pol


def run_rl(policy, jobs, num_gpus, device):
    _, m, _ = run_episode(policy, jobs, num_gpus=num_gpus, deterministic=True,
                          collect=False, device=device)
    return m


def evaluate_at_load(policy, num_jobs, target_rho, bursty, seeds, num_gpus, device,
                     p_deadline=0.0):
    """Return {scheduler: {metric: {mean, ci}}} aggregated over seeds.

    `policy` may be None (controller-only / no trained RL), in which case the RL
    row is skipped — the interpretable lambda-controller is the primary 'ours'.
    """
    raw: Dict[str, Dict[str, List[float]]] = {}

    def add(name, m, ref_makespan):
        d = raw.setdefault(name, {k: [] for k in METRICS + ["mux_speedup"]})
        for k in METRICS:
            d[k].append(m.get(k, 0.0))
        d["mux_speedup"].append(ref_makespan / m["makespan"] if m["makespan"] > 0 else 0.0)

    for seed in seeds:
        jobs = generate_workload(num_jobs=num_jobs, seed=seed,
                                 target_rho=target_rho, bursty=bursty,
                                 p_deadline=p_deadline)
        ref = run_heuristic(jobs, NO_MULTIPLEX_REF, num_gpus=num_gpus)
        ref_mk = ref["makespan"]
        for name, sched in BASELINES.items():
            add(name, run_heuristic(jobs, sched, num_gpus=num_gpus), ref_mk)
        add("SRPT-Preemptive", run_srpt_preemptive(jobs, num_gpus=num_gpus), ref_mk)
        add("N-Controller (ours)",
            run_heuristic(jobs, lambda_controller(CONTROLLER_LAMBDA), num_gpus=num_gpus), ref_mk)
        if policy is not None:
            add("RL (ours)", run_rl(policy, jobs, num_gpus, device), ref_mk)
        add("ORACLE-adaptive", oracle_adaptive_hindsight(jobs, num_gpus=num_gpus), ref_mk)
        add("ORACLE-unconstrained", oracle_unconstrained(jobs, num_gpus=num_gpus), ref_mk)

    out: Dict[str, Dict[str, dict]] = {}
    for name, md in raw.items():
        out[name] = {k: {"mean": float(np.mean(v)), "ci": ci95(v)} for k, v in md.items()}
    return out


def print_table(label, results):
    order = ["FIFO", "BestFit", "LocalityGreedy", "Fixed-N1", "Fixed-N2",
             "Fixed-N8", "LoadAdaptive", "SRPT-Preemptive", "N-Controller (ours)",
             "RL (ours)", "ORACLE-adaptive", "ORACLE-unconstrained"]
    hdr = (f"{'scheduler':<22}{'mean_jct':>13}{'makespan':>12}{'first_compl':>12}"
           f"{'mux_x':>7}{'dl_hit':>7}{'N_tv':>7}")
    print(f"\n=== {label} ===")
    print(hdr)
    for name in order:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<22}"
              f"{r['mean_jct']['mean']:>9.0f}±{r['mean_jct']['ci']:<3.0f}"
              f"{r['makespan']['mean']:>12.0f}"
              f"{r['time_to_first_completion']['mean']:>12.0f}"
              f"{r['mux_speedup']['mean']:>7.2f}"
              f"{r['deadline_hit_rate']['mean']:>7.2f}"
              f"{r['realized_N_tv']['mean']:>7.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Optional trained RL checkpoint. If omitted, the "
                         "interpretable N-Controller is the primary 'ours' and the "
                         "RL row is skipped.")
    ap.add_argument("--architecture", default=None, choices=[None, "residual", "gat"],
                    help="Override policy architecture; auto-detected from the checkpoint if omitted.")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--p-deadline", type=float, default=0.0,
                    help="Fraction of jobs with an SLO deadline (for the deadline columns).")
    ap.add_argument("--result-dir", default="c_lora_sim/results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu")
    policy = load_policy(args.model, device, architecture=args.architecture) if args.model else None
    seeds = [20_000 + i for i in range(args.seeds)]
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # rho sweep: stationary and bursty variants at the critical regimes
    LOADS = [
        ("underloaded",        40,  0.50, False),
        ("light",              50,  0.75, False),
        ("critical",           60,  0.90, False),
        ("critical_bursty",    60,  0.90, True),
        ("overloaded",         70,  1.50, False),
        ("overloaded_bursty",  70,  1.50, True),
    ]

    sweep: Dict[str, Dict] = {}
    for label, nj, rho, bursty in LOADS:
        tag = f"rho={rho:.2f}{'_bursty' if bursty else ''}"
        print(f"evaluating {label} ({tag}, jobs={nj}) over {len(seeds)} seeds ...")
        res = evaluate_at_load(policy, nj, rho, bursty, seeds, args.num_gpus, device,
                               p_deadline=args.p_deadline)
        sweep[label] = {
            "num_jobs": nj, "target_rho": rho, "bursty": bursty,
            "results": res,
        }
        print_table(f"{label} ({tag})", res)

    with (result_dir / "evaluation.json").open("w") as f:
        json.dump(sweep, f, indent=2)

    # ---- Controller (and RL, if present) vs the oracle bracket -----------
    # Primary claim (Option B+): the interpretable controller tracks the
    # passive-drain adaptive oracle, and the ASYMMETRY GAP (adaptive vs the
    # unconstrained free-rebalance lower bound) quantifies the cost of the
    # actuator constraint.
    print("\n===== N-Controller vs oracle bracket (mean_jct) =====")
    summary = {}
    primary = "N-Controller (ours)"
    for label, _, rho, bursty in LOADS:
        res = sweep[label]["results"]
        ours = res[primary]["mean_jct"]["mean"]
        adp = res["ORACLE-adaptive"]["mean_jct"]["mean"]
        unc = res["ORACLE-unconstrained"]["mean_jct"]["mean"]
        gap_to_oracle = 100.0 * (ours - adp) / adp if adp > 0 else 0.0
        asym_gap = 100.0 * (adp - unc) / unc if unc > 0 else 0.0
        summary[label] = {
            "controller": ours, "oracle_adaptive": adp, "oracle_unconstrained": unc,
            "controller_gap_to_oracle_pct": gap_to_oracle, "asymmetry_gap_pct": asym_gap,
        }
        print(f"{label:<22} ctrl={ours:>9.0f}  ORACLE-adp={adp:.0f}({gap_to_oracle:+.1f}%)  "
              f"asymmetry-gap={asym_gap:+.1f}%")

    with (result_dir / "controller_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_plots:
        make_plots(sweep, LOADS, result_dir)
        print(f"\nplots written to {result_dir}")


def make_plots(sweep, loads, result_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l[0] for l in loads]
    order = ["Fixed-N1", "Fixed-N2", "Fixed-N8", "LoadAdaptive",
             "SRPT-Preemptive", "N-Controller (ours)", "ORACLE-adaptive",
             "ORACLE-unconstrained"]
    colors = ["#aaa", "#bbb", "#f85", "#fb3", "#a0f", "#39f", "#e22", "#2a2"]
    styles = ["-", "-", "-", "--", "--", "-", ":", ":"]
    lws   = [1.0, 1.0, 1.0, 1.5, 1.8, 2.5, 2.0, 1.6]

    # 1) mean_jct vs load regime
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(labels)))
    for name, c, ls, lw in zip(order, colors, styles, lws):
        ys, es = [], []
        for lab in labels:
            r = sweep[lab]["results"].get(name)
            if r:
                ys.append(r["mean_jct"]["mean"])
                es.append(r["mean_jct"]["ci"])
            else:
                ys.append(float("nan"))
                es.append(0)
        ax.errorbar(x, ys, yerr=es, marker="o", label=name, color=c,
                    linestyle=ls, linewidth=lw, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlabel("load regime (ρ)")
    ax.set_ylabel("mean experiment time (s)")
    ax.set_title("Mean JCT vs load regime — RL-BSBF vs baselines + oracle\n(lower is better; dashed=LoadAdaptive teacher)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(result_dir / "load_sweep_jct.png", dpi=130)
    plt.close(fig)

    # 2) throughput vs tail-latency tradeoff at critical (bursty)
    tgt = "critical_bursty" if "critical_bursty" in sweep else labels[0]
    med = sweep[tgt]["results"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, c in zip(order, colors):
        r = med.get(name)
        if not r:
            continue
        ax.scatter(r["mux_speedup"]["mean"], r["p95_jct"]["mean"],
                   s=140, color=c, edgecolors="k", zorder=3)
        ax.annotate(name, (r["mux_speedup"]["mean"], r["p95_jct"]["mean"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("multiplexing speedup (×, higher better)")
    ax.set_ylabel("p95 JCT (s, lower better)")
    ax.set_title(f"Throughput vs tail-latency — {tgt}\n(RL should sit bottom-right of LoadAdaptive)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(result_dir / "tradeoff.png", dpi=130)
    plt.close(fig)

    # 3) controller gap-to-oracle and the asymmetry gap, per regime
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gap_ctrl, gap_asym = [], []
    for lab in labels:
        res = sweep[lab]["results"]
        ours = res["N-Controller (ours)"]["mean_jct"]["mean"]
        adp = res["ORACLE-adaptive"]["mean_jct"]["mean"]
        unc = res["ORACLE-unconstrained"]["mean_jct"]["mean"]
        gap_ctrl.append(100.0 * (ours - adp) / adp if adp > 0 else 0.0)
        gap_asym.append(100.0 * (adp - unc) / unc if unc > 0 else 0.0)
    for ax, vals, title in zip(
        axes, [gap_ctrl, gap_asym],
        ["Controller gap to adaptive oracle (%)\n(lower = closer to oracle)",
         "Asymmetry gap: adaptive vs unconstrained (%)\n(cost of the actuator constraint)"],
    ):
        bars = ax.bar(labels, vals, color=["#39f" if v <= 5 else "#e55" for v in vals])
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(result_dir / "controller_vs_oracle.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
