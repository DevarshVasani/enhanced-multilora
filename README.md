# Dynamic N-Selection for Multi-LoRA Training

**Question.** Co-locating `N` adapter-training jobs on one GPU trades throughput against
latency. What is the optimal N, and can we adapt it online as the workload shifts?

**Answer.** An interpretable λ-controller lands within 4.4% of the passive-drain
oracle at every load with zero infrastructure. Free eviction is worth building only
for oversubscribed clusters with mixed-length workloads; below capacity, passive drain
plus the λ-controller is sufficient.

Full write-up: [`blog/index.html`](blog/index.html). Technical report: [`c_lora_sim/MEMO.md`](c_lora_sim/MEMO.md).

---

## Folder layout

```
multilora_n_selection/
├── c_lora_sim/          # simulation package (importable as `c_lora_sim`)
│   ├── n_controller.py  # λ-controller (primary result)
│   ├── n_baselines.py   # fixed-N and LoadAdaptive baselines
│   ├── oracles.py       # passive-drain and free-eviction oracles
│   ├── workload.py      # job generator (rho, profiles)
│   ├── runner.py        # heuristic evaluation harness
│   ├── clora_cluster.py / clora_env.py / clora_job.py / data_plane.py / control_plane.py
│   │                    # physics layer (calibrated to Qwen2.5-0.5B / RTX 3050)
│   ├── calibration.py   # step_scaling(N), calibrated from hardware
│   ├── calibration_validation.py   # validates sim against phase0 measurements
│   ├── expert.py        # load-adaptive expert (BC teacher)
│   ├── bc_pretrain.py   # behaviour-clone training
│   ├── ppo_clora.py     # PPO fine-tuning (currently unstable vs BC baseline)
│   ├── train_clora.py   # training entry point
│   ├── evaluate_clora.py
│   ├── phase0_qwen_measure.py   # real-hardware measurement script
│   ├── phase_asymmetry_gap.py   # Exp: free-eviction value by load (5 seeds)
│   ├── phase_ci_rho15.py        # Exp: CI on asymmetry gap (20 seeds, ρ=1.5)
│   ├── phase_frontier_rho15.py  # Exp: full scheduler ladder at ρ=1.5
│   ├── phase_baselines_generalization.py  # tune on ρ=0.9, eval on ρ=1.5
│   ├── phase_gap_decompose.py   # decompose gap into placement vs preemption
│   ├── experiment_n_control.py  # main experiment (frontier + tracking + chatter)
│   ├── critique_experiments.py  # reviewer-challenge experiments
│   ├── seed_aggregate.py        # aggregate multi-seed training variance
│   ├── results/         # pre-computed JSON results + plots
│   │   └── critiques/   # oracle ladder, jitter, extrapolation, seed variance
│   ├── seed_study/      # per-seed trained models + results (seeds 1,2,3,4,42)
│   ├── models/          # BC-trained checkpoint (bc_init.pt, best.pt, latest.pt)
│   ├── fonts/           # Slam font family for plots
│   └── MEMO.md          # technical summary
├── blog/
│   ├── index.html       # blog post (self-contained HTML)
│   ├── styles.css
│   ├── gen_figs.py      # regenerate blog figures from results/
│   ├── fig_validation.png   # Phase-0 hardware scatter
│   ├── fig_bimodal.png      # asymmetry-gap bimodal histogram
│   └── fig_frontier.png     # controller frontier at ρ=1.5
├── requirements.txt
└── README.md
```

---

## Reproduce

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Phase 0 — hardware validation (needs Qwen2.5-0.5B + GPU)

```bash
python -m c_lora_sim.phase0_qwen_measure
# writes c_lora_sim/results/phase0_qwen.json
```

### Core experiments (CPU-only, ~minutes each)

```bash
# Free-eviction value by load regime (5 seeds)
python -m c_lora_sim.phase_asymmetry_gap

# CI on the overload gap (20 seeds, ρ=1.5, ~15 min)
python -m c_lora_sim.phase_ci_rho15

# Full scheduler ladder at ρ=1.5
python -m c_lora_sim.phase_frontier_rho15

# Generalization: tune on ρ=0.9, eval on unseen ρ=1.5
python -m c_lora_sim.phase_baselines_generalization

# Gap decomposition (placement order vs preemption)
python -m c_lora_sim.phase_gap_decompose

# Main experiment: frontier + tracking + chatter plots
python -m c_lora_sim.experiment_n_control
```

### Reviewer-challenge experiments

```bash
python -m c_lora_sim.critique_experiments oracle
python -m c_lora_sim.critique_experiments extrapolation
python -m c_lora_sim.critique_experiments jitter
bash c_lora_sim/seed_study/run_seeds.sh
python -m c_lora_sim.seed_aggregate
```

### BC / PPO training (GPU recommended)

```bash
# Behaviour-clone warm-start (~30k samples)
python -m c_lora_sim.bc_pretrain

# PPO fine-tune (currently degrades BC baseline — documented finding)
python -m c_lora_sim.train_clora
```

### Regenerate blog figures

```bash
cd blog
python gen_figs.py
# writes fig_validation.png, fig_bimodal.png, fig_frontier.png
```

---

## Key findings

| claim | script | pre-run result |
|---|---|---|
| Sim validated on Qwen/RTX 3050, ≤10.5% residual | `phase0_qwen_measure.py` | `results/phase0_qwen.json` |
| Depth-rebalancing gap = 0% (passive drain is free) | `phase_asymmetry_gap.py` | `results/asymmetry_gap.json` |
| Free-eviction gap at ρ=1.5: 66%, CI [32%, 122%], bimodal | `phase_ci_rho15.py` | `results/asymmetry_gap_ci.json` |
| λ-controller (λ=1) within 4.4% of drain oracle at ρ=1.5 | `phase_frontier_rho15.py` | `results/frontier_rho15.json` |
| Same λ=1, unseen ρ=1.5 (generalization test) | `phase_baselines_generalization.py` | `results/baselines_generalization.json` |
| Placement headroom = 0% (gap is preemption only) | `phase_gap_decompose.py` | (logged) |
| PPO degrades BC clone; BC clone = +13.5% from oracle | `train_clora.py` / `evaluate_clora.py` | `models/best.pt` |
