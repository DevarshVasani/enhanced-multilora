"""
Plot the RL learning curve: eval mean_jct and the RL->adaptive-oracle GAP vs
training episode (plan Flaw 7 — a learning curve, NOT a tautological bound). The
meaningful claim is that the gap NARROWS with training; a flat/widening gap means
the policy is not learning anything useful.

    python -m c_lora_sim.gen_learning_curve --csv c_lora_sim/results/training_metrics.csv
"""

from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="c_lora_sim/results/training_metrics.csv")
    ap.add_argument("--out", default="c_lora_sim/results/learning_curve.png")
    args = ap.parse_args()

    eps, evals, gaps = [], [], []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("eval_mean_jct") not in (None, ""):
                eps.append(int(row["episode"]))
                evals.append(float(row["eval_mean_jct"]))
                g = row.get("oracle_gap_pct", "")
                gaps.append(float(g) if g not in (None, "") else float("nan"))

    if not eps:
        print("no eval checkpoints logged yet")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(eps, evals, "o-", color="tab:blue", label="eval mean_jct")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("eval mean_jct (s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(eps, gaps, "s--", color="tab:red", label="gap to adaptive oracle (%)")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("RL → adaptive-oracle gap (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("RL learning curve — eval JCT and oracle gap vs training\n"
                  "(gap should NARROW with training)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"final eval mean_jct={evals[-1]:.0f}  final oracle gap={gaps[-1]:+.1f}%  "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
