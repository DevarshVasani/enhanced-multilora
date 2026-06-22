"""
Generate the critique-response figures from the JSON produced by
critique_experiments.py. Writes PNGs into c_lora_sim/results/critiques/.

  oracle_ladder.png          -- Critique 1: RL margin vs the oracle ladder
  extrapolation_invariance.png -- Critique 4: RL invariant to N>8 curve, rarely packs >8
  jitter_robustness.png      -- Critique 2: RL margin survives real-world jitter
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CDIR = Path("c_lora_sim/results/critiques")
SHORT = {
    "underloaded": "under\n0.50", "light": "light\n0.75", "critical": "crit\n0.90",
    "critical_bursty": "crit+b\n0.90", "overloaded": "over\n1.50",
    "overloaded_bursty": "over+b\n1.50",
}
ORDER = ["underloaded", "light", "critical", "critical_bursty", "overloaded", "overloaded_bursty"]


def plot_oracle():
    d = json.load((CDIR / "oracle.json").open())
    labels = [SHORT[k] for k in ORDER]
    oracles = [("ORACLE-fixedN", "static Fixed-N oracle", "#94a3b8"),
               ("ORACLE-library", "best-of-library oracle", "#fb923c"),
               ("ORACLE-adaptive", "clairvoyant adaptive oracle", "#2563eb")]
    x = np.arange(len(ORDER)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (key, name, col) in enumerate(oracles):
        means = [d[k][key]["rl_margin_pct"]["mean"] for k in ORDER]
        cis = [d[k][key]["rl_margin_pct"]["ci"] for k in ORDER]
        ax.bar(x + (i - 1) * w, means, w, yerr=cis, capsize=3, label=name, color=col,
               edgecolor="#1e293b", linewidth=0.5)
    ax.axhline(0, color="#111", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("RL margin  (%, + = RL beats the oracle)")
    ax.set_title("Critique 1 — RL-BSBF vs the oracle ladder\n"
                 "RL beats the static oracle AND matches/beats a clairvoyant within-episode "
                 "adaptive oracle (using only causal info)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(CDIR / "oracle_ladder.png", dpi=140); plt.close(fig)
    print("wrote oracle_ladder.png")


def plot_extrapolation():
    d = json.load((CDIR / "extrapolation.json").open())
    labels = [SHORT[k] for k in ORDER]
    variants = ["linear", "default", "harsh", "hardcap8"]
    vcol = {"linear": "#a3a3a3", "default": "#2563eb", "harsh": "#dc2626", "hardcap8": "#0d9488"}
    x = np.arange(len(ORDER)); w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: RL mean JCT under each extrapolation shape -> invariance.
    for i, v in enumerate(variants):
        means = [d[k]["variants"][v]["RL"]["mean"] for k in ORDER]
        ax1.bar(x + (i - 1.5) * w, means, w, label=v, color=vcol[v],
                edgecolor="#1e293b", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("RL mean JCT (s)")
    ax1.set_title("RL performance is invariant to the N>8 penalty shape\n(<0.4% change across linear/default/harsh/hardcap)")
    ax1.legend(fontsize=9, title="N>8 extrapolation"); ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(d[k]["variants"]["default"]["RL"]["mean"] for k in ORDER) * 1.18)

    # Panel B: how often the policy actually packs past N=8 (default physics).
    frac = [100 * d[k]["variants"]["default"]["frac_placements_N_gt_8"]["mean"] for k in ORDER]
    maxd = [d[k]["variants"]["default"]["max_depth"]["mean"] for k in ORDER]
    bars = ax2.bar(x, frac, 0.55, color="#2563eb", edgecolor="#1e293b", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("% of placements with N > 8")
    ax2.set_title("The agent almost never enters the extrapolated region\n(annotated: mean max co-location depth reached)")
    ax2.set_ylim(0, max(frac) * 1.5 + 0.5)
    for b, f, m in zip(bars, frac, maxd):
        ax2.annotate(f"{f:.1f}%\nmaxN={m:.1f}", (b.get_x() + b.get_width() / 2, f),
                     ha="center", va="bottom", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(CDIR / "extrapolation_invariance.png", dpi=140); plt.close(fig)
    print("wrote extrapolation_invariance.png")


def plot_jitter():
    path = CDIR / "jitter.json"
    if not path.exists():
        print("skip jitter_robustness.png (no jitter.json)")
        return
    d = json.load(path.open())
    levels = ["noiseless", "mild", "realistic", "harsh"]
    lcol = {"noiseless": "#a3a3a3", "mild": "#60a5fa", "realistic": "#2563eb", "harsh": "#dc2626"}
    x = np.arange(len(ORDER)); w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    # Panel A: RL margin vs Fixed-N2 across noise levels.
    for i, lv in enumerate(levels):
        means = [d[k]["levels"][lv]["Fixed-N2"]["rl_margin_pct"]["mean"] for k in ORDER]
        cis = [d[k]["levels"][lv]["Fixed-N2"]["rl_margin_pct"]["ci"] for k in ORDER]
        ax1.bar(x + (i - 1.5) * w, means, w, yerr=cis, capsize=2, label=lv, color=lcol[lv],
                edgecolor="#1e293b", linewidth=0.4)
    ax1.axhline(0, color="#111", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels([SHORT[k] for k in ORDER])
    ax1.set_ylabel("RL margin vs Fixed-N2 (%)")
    ax1.set_title("RL edge vs Fixed-N2 is preserved under jitter")
    ax1.legend(fontsize=8, title="noise level"); ax1.grid(axis="y", alpha=0.3)

    # Panel B: RL margin vs the static Fixed-N oracle across noise levels.
    for i, lv in enumerate(levels):
        means = [d[k]["levels"][lv]["ORACLE-fixedN"]["rl_margin_pct"]["mean"] for k in ORDER]
        cis = [d[k]["levels"][lv]["ORACLE-fixedN"]["rl_margin_pct"]["ci"] for k in ORDER]
        ax2.bar(x + (i - 1.5) * w, means, w, yerr=cis, capsize=2, label=lv, color=lcol[lv],
                edgecolor="#1e293b", linewidth=0.4)
    ax2.axhline(0, color="#111", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels([SHORT[k] for k in ORDER])
    ax2.set_ylabel("RL margin vs static Fixed-N oracle (%)")
    ax2.set_title("RL still beats the hindsight static oracle under jitter")
    ax2.legend(fontsize=8, title="noise level"); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Critique 2 — robustness to real-world jitter (step CV up to 0.25, cold-start CV up to 0.35)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(CDIR / "jitter_robustness.png", dpi=140); plt.close(fig)
    print("wrote jitter_robustness.png")


def plot_seeds():
    path = CDIR / "seed_study.json"
    if not path.exists():
        print("skip seed_variance.png (no seed_study.json)")
        return
    d = json.load(path.open())
    agg = d["aggregate"]
    x = np.arange(len(ORDER)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: margin vs static Fixed-N oracle (the correct claim).
    means = [agg[k]["vs_oracle_fixedN_pct"]["mean"] for k in ORDER]
    cis   = [agg[k]["vs_oracle_fixedN_pct"]["ci"]   for k in ORDER]
    frac  = [agg[k]["vs_oracle_fixedN_pct"]["frac_positive"] for k in ORDER]
    cols  = ["#2563eb" if m > 0 else "#dc2626" for m in means]
    bars  = ax1.bar(x, means, w, yerr=cis, capsize=4, color=cols, edgecolor="#1e293b", lw=0.5)
    for b, f in zip(bars, frac):
        ax1.annotate(f"{100*f:.0f}%", (b.get_x() + b.get_width() / 2, b.get_height() + cis[list(bars).index(b)]),
                     ha="center", va="bottom", fontsize=8)
    ax1.axhline(0, color="#111", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels([SHORT[k] for k in ORDER])
    ax1.set_ylabel("RL margin vs static Fixed-N oracle (%, +5 seeds)")
    ax1.set_title("Critique 3 — vs static oracle\n(annotation = % of 5 training seeds beating oracle)")
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: margin vs clairvoyant adaptive oracle.
    means2 = [agg[k]["vs_oracle_adaptive_pct"]["mean"] for k in ORDER]
    cis2   = [agg[k]["vs_oracle_adaptive_pct"]["ci"]   for k in ORDER]
    cols2  = ["#2563eb" if m > 0 else "#94a3b8" for m in means2]
    ax2.bar(x, means2, w, yerr=cis2, capsize=4, color=cols2, edgecolor="#1e293b", lw=0.5)
    ax2.axhline(0, color="#111", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels([SHORT[k] for k in ORDER])
    ax2.set_ylabel("RL margin vs clairvoyant adaptive oracle (%)")
    ax2.set_title("Critique 3 — vs adaptive oracle\n(most seeds within ~1% of this much harder ceiling)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout(); fig.savefig(CDIR / "seed_variance.png", dpi=140); plt.close(fig)
    print("wrote seed_variance.png")


if __name__ == "__main__":
    plot_oracle()
    plot_extrapolation()
    plot_jitter()
    plot_seeds()
