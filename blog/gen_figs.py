"""Generate the three figures for the optimal-N blog post.

All numbers are read from / consistent with c_lora_sim/results/*.json:
  - phase0_qwen.json          -> Figure 1 (validation scatter)
  - asymmetry_gap_ci.json     -> Figure 2 (bimodal histogram)
  - frontier_rho15.json +
    baselines_generalization.json -> Figure 3 (controller frontier)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "c_lora_sim", "results")

# Blog palette (matches styles.css)
BG = "#f7f9e8"
INK = "#253241"
ACCENT = "#70792a"
HILITE = "#c0392b"
GREEN = "#2b7a3b"
GRID = "#aeb8f2"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 13,
    "font.family": "sans-serif",
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- Figure 1
def fig_validation():
    d = load("phase0_qwen.json")
    rows = d["step_scaling_rows"]
    pred = [r["pred_scaling"] for r in rows]
    meas = [r["meas_scaling"] for r in rows]
    ns = [r["N"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    lo, hi = 0.8, 2.95
    ax.plot([lo, hi], [lo, hi], "--", color=INK, lw=1.3, alpha=0.6,
            label="y = x  (perfect prediction)")
    ax.scatter(pred, meas, s=170, color=ACCENT, edgecolor=INK,
               linewidth=1.3, zorder=5)
    for p, m, n, r in zip(pred, meas, ns, rows):
        dy = 0.10 if n != 2 else -0.16
        ax.annotate(f"N={n}\n({r['residual']*100:.1f}%)", (p, m),
                    textcoords="offset points", xytext=(10, 8 if n != 2 else -28),
                    fontsize=11, color=INK, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted step scaling  (simulator)")
    ax.set_ylabel("Measured step scaling  (real Qwen2.5-0.5B)")
    ax.set_title("Plant validation: simulator vs real hardware",
                 fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.grid(True, color=GRID, alpha=0.35)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_validation.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


# ---------------------------------------------------------------- Figure 2
def fig_bimodal():
    d = load("asymmetry_gap_ci.json")
    gaps = np.array(d["per_seed_gap"])
    edges = [0, 5, 50, 100, 200, np.inf]
    labels = ["<5%", "5–50%", "50–100%", "100–200%", ">200%"]
    counts = []
    for i in range(len(edges) - 1):
        counts.append(int(np.sum((gaps >= edges[i]) & (gaps < edges[i + 1]))))

    colors = ["#9aa3c9", "#9aa3c9", ACCENT, ACCENT, HILITE]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    bars = ax.bar(labels, counts, color=colors, edgecolor=INK, linewidth=1.2,
                  width=0.74)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.12, str(c),
                ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_ylim(0, max(counts) + 1.6)
    ax.set_ylabel("Number of workloads (seeds)")
    ax.set_xlabel("Value of free eviction over passive drain  (% JCT improvement)")
    ax.set_title("Free eviction's value is bimodal  (ρ = 1.5, 20 seeds)",
                 fontsize=15, fontweight="bold", pad=12)
    ax.text(0.02, 0.97, "6 workloads <5%\nno short-behind-long contention",
            transform=ax.transAxes, fontsize=10, color="#5b6280", ha="left",
            va="top")
    ax.text(0.98, 0.97, "8 workloads >150%\nshort jobs stuck behind long",
            transform=ax.transAxes, fontsize=10, color=HILITE, ha="right",
            va="top", fontweight="bold")
    ax.grid(True, axis="y", color=GRID, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_bimodal.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


# ---------------------------------------------------------------- Figure 3
def fig_frontier():
    fr = load("frontier_rho15.json")
    gen = load("baselines_generalization.json")
    oracle = fr["oracle_asymmetric"]

    rows = [
        ("ORACLE (drain, no eviction)", oracle),
        ("λ-controller", fr["lambda"]["1.0"]),
        ("AIMD / TCP", oracle * (1 + gen["aimd"]["gap_pct"] / 100)),
        ("Fixed-N4", fr["fixed"]["Fixed-N4"]),
        ("LoadAdaptive", fr["load_adaptive"]),
        ("Threshold-hysteresis", oracle * (1 + gen["threshold"]["gap_pct"] / 100)),
        ("Fixed-N2", fr["fixed"]["Fixed-N2"]),
        ("Fixed-N8", fr["fixed"]["Fixed-N8"]),
        ("Fixed-N1", fr["fixed"]["Fixed-N1"]),
    ]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    gaps = [(v / oracle - 1) * 100 for v in vals]

    y = np.arange(len(rows))[::-1]  # top = best
    colors = []
    for n in names:
        if n.startswith("λ"):
            colors.append(GREEN)
        elif n.startswith("ORACLE"):
            colors.append(INK)
        else:
            colors.append("#9aa3c9")

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(y, vals, color=colors, edgecolor=INK, linewidth=1.1, height=0.66)
    ax.axvline(oracle, ls="--", color=INK, lw=1.2, alpha=0.55)
    for yi, v, g, n in zip(y, vals, gaps, names):
        lbl = f"{v:,.0f}" + (f"  (+{g:.1f}%)" if g > 0.05 else "  (0%)")
        ax.text(v + 1500, yi, lbl, va="center", fontsize=11,
                fontweight="bold" if n.startswith("λ") else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11.5)
    ax.set_xlim(0, 158000)
    ax.set_xlabel("Mean job completion time  (lower is better)")
    ax.set_title("Controller frontier at ρ = 1.5 (unseen; tuned on ρ = 0.9)",
                 fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, axis="x", color=GRID, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_frontier.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    fig_validation()
    fig_bimodal()
    fig_frontier()
