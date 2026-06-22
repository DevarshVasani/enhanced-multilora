"""
Train the C-LoRA PPO scheduling policy.

Usage:
    python -m c_lora_sim.train_clora --episodes 800

Curriculum: each episode samples a random utilisation level (target_rho) AND
a random burst flag, so the policy must learn to adapt its co-location depth
to queue pressure and within-episode bursts -- exactly the "dynamic load
balancing across jobs" C-LoRA names as an open problem (ARCHITECTURE.md
Theorems 1-2). The policy is initialised from a load-adaptive BC teacher
(expert.py) that already handles the regime shift; PPO must improve beyond it.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from c_lora_sim.ppo_clora import evaluate, make_policy, run_episode
from c_lora_sim.workload import generate_workload

# Held-out evaluation specs covering all three regimes (ARCHITECTURE.md §5).
# Each spec is passed directly to generate_workload as kwargs.
EVAL_SPECS = [
    dict(num_jobs=40,  seed=9000, target_rho=0.50, bursty=False),   # underloaded
    dict(num_jobs=50,  seed=9001, target_rho=0.75, bursty=False),   # light
    dict(num_jobs=60,  seed=9002, target_rho=0.90, bursty=False),   # critical
    dict(num_jobs=60,  seed=9003, target_rho=0.90, bursty=True),    # critical+burst
    dict(num_jobs=70,  seed=9004, target_rho=1.50, bursty=False),   # overloaded
    dict(num_jobs=70,  seed=9005, target_rho=1.50, bursty=True),    # overloaded+burst
]

# Curriculum rho levels; sampled uniformly each episode.
RHO_LEVELS  = [0.50, 0.75, 0.90, 1.10, 1.50]
# Probability of a bursty episode.
P_BURSTY    = 0.40


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",    type=int,   default=800)
    p.add_argument("--batch-size",  type=int,   default=10)
    p.add_argument("--num-gpus",    type=int,   default=8)
    p.add_argument("--hidden-dim",  type=int,   default=128)
    p.add_argument("--layers",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--entropy",     type=float, default=0.02)
    p.add_argument("--reward-scale",type=float, default=50_000.0)
    p.add_argument("--tail-coef",   type=float, default=0.0)
    p.add_argument("--feature-mode",type=str,   default="v3",
                   help="v1 | v2 | v3 (v3 adds deadline + asymmetry features)")
    p.add_argument("--deadline-shaping-coef", type=float, default=0.5)
    p.add_argument("--miss-penalty",          type=float, default=0.2)
    p.add_argument("--churn-coef",            type=float, default=0.002)
    p.add_argument("--evict-coef",            type=float, default=1.0)
    p.add_argument("--p-deadline",            type=float, default=0.3)
    p.add_argument("--eval-interval",type=int,  default=25)
    p.add_argument("--max-steps",   type=int,   default=20_000,
                   help="Decision cap per episode. Lower bounds the cost of "
                        "pathological early-policy episodes on CPU.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--init-model",  type=str,   default=None,
                   help="Checkpoint to warm-start from (e.g. bc_init.pt)")
    p.add_argument("--model-dir",   type=str,   default="c_lora_sim/models")
    p.add_argument("--result-dir",  type=str,   default="c_lora_sim/results")
    return p.parse_args()


# Overloaded specs (rho>=1.5) produce million-second makespans -> millions of
# events, making CPU eval/oracle-baseline intractable. For a tractable, VALID
# learning curve on CPU we evaluate on SMALL workloads that fully complete within a
# modest decision budget (so capping training episodes does NOT truncate the eval
# metric). These still span under-/critically-loaded adaptation.
EVAL_SUBSET = [
    dict(num_jobs=24, seed=9100, target_rho=0.50, bursty=False),
    dict(num_jobs=24, seed=9101, target_rho=0.75, bursty=False),
    dict(num_jobs=28, seed=9102, target_rho=0.90, bursty=False),
    dict(num_jobs=28, seed=9103, target_rho=0.90, bursty=True),
]


def eval_policy(policy, num_gpus, device, specs=None, max_steps=20_000):
    # Use the completion-robust metric (charges unfinished jobs their elapsed
    # time) so a policy that stalls and completes only short jobs is penalised,
    # not flattered (plan Flaw 7: a valid learning-curve signal).
    flows = []
    for spec in (specs if specs is not None else EVAL_SUBSET):
        jobs = generate_workload(**spec)
        m = evaluate(policy, jobs, num_gpus=num_gpus, device=device, max_steps=max_steps)
        flows.append(m["mean_flow_all"])
    return float(np.mean(flows))


def _oracle_baseline(num_gpus, specs=None):
    """Mean adaptive-oracle JCT over the eval specs, computed once. The training
    loop logs the RL->oracle gap against this so we can SEE the policy close the
    gap with training (plan Flaw 7: a learning curve, not a tautological bound).
    `specs` may subsample EVAL_SPECS to keep the (clairvoyant, 18-config) oracle
    sweep tractable on CPU."""
    from c_lora_sim.oracles import oracle_adaptive_hindsight
    specs = specs if specs is not None else EVAL_SPECS
    # mean_flow_all == mean_jct for the oracle (it finishes everything), but using
    # the same metric as eval_policy keeps the gap exactly comparable.
    flows = [oracle_adaptive_hindsight(generate_workload(**spec), num_gpus=num_gpus)["mean_flow_all"]
             for spec in specs]
    return float(np.mean(flows))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir  = Path(args.model_dir)
    result_dir = Path(args.result_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    from gpu_rl.ppo import ppo_update
    policy    = make_policy(hidden_dim=args.hidden_dim, num_layers=args.layers,
                            device=device, feature_mode=args.feature_mode)
    oracle_base = _oracle_baseline(args.num_gpus, specs=EVAL_SUBSET)
    print(f"adaptive-oracle baseline mean_jct over eval specs = {oracle_base:.0f}")
    if args.init_model:
        ck = torch.load(args.init_model, map_location=device, weights_only=False)
        policy.load_state_dict(ck["model_state_dict"])
        print(f"warm-started from {args.init_model} (source={ck.get('source','?')})")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    best_eval  = float("inf")
    metrics_path = result_dir / "training_metrics.csv"
    handle = metrics_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=[
        "episode", "rho", "bursty", "return", "mean_jct", "makespan",
        "loss", "policy_loss", "value_loss", "entropy", "eval_mean_jct",
        "oracle_gap_pct",
    ])
    writer.writeheader()

    batch = []
    last  = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for ep in range(1, args.episodes + 1):
        rho    = random.choice(RHO_LEVELS)
        bursty = random.random() < P_BURSTY
        num_jobs = random.choice([40, 50, 60, 70, 80])
        jobs = generate_workload(
            num_jobs=num_jobs,
            seed=random.randint(0, 2**31 - 1),
            target_rho=rho,
            bursty=bursty,
            p_deadline=args.p_deadline,
        )
        steps, summary, _ = run_episode(
            policy, jobs, num_gpus=args.num_gpus, deterministic=False,
            reward_scale=args.reward_scale,
            tail_coef=args.tail_coef,
            deadline_shaping_coef=args.deadline_shaping_coef,
            miss_penalty=args.miss_penalty,
            churn_coef=args.churn_coef,
            evict_coef=args.evict_coef,
            max_steps=args.max_steps,
            device=device, collect=True,
        )
        batch.extend(steps)

        if ep % args.batch_size == 0 or ep == args.episodes:
            last  = ppo_update(policy, optimizer, batch, train_epochs=4,
                               entropy_coef=args.entropy, device=device)
            batch = []

        eval_jct = ""
        if ep % args.eval_interval == 0 or ep == args.episodes:
            # Eval is UNCAPPED (small specs complete fully and fast) so the
            # reported mean_jct / oracle gap is valid even when TRAINING episodes
            # are capped for speed.
            eval_jct = eval_policy(policy, args.num_gpus, device, max_steps=50_000)
            ckpt = {
                "model_state_dict": policy.state_dict(),
                "hidden_dim": args.hidden_dim, "layers": args.layers,
                "episode": ep, "eval_mean_jct": eval_jct,
            }
            torch.save(ckpt, model_dir / "latest.pt")
            if eval_jct < best_eval:
                best_eval = eval_jct
                torch.save(ckpt, model_dir / "best.pt")

        gap_pct = ""
        if eval_jct != "" and oracle_base > 0:
            gap_pct = (float(eval_jct) - oracle_base) / oracle_base * 100.0
        writer.writerow({
            "episode": ep, "rho": rho, "bursty": int(bursty),
            "return": sum(s.reward for s in steps),
            "mean_jct": summary["mean_jct"], "makespan": summary["makespan"],
            "loss": last["loss"], "policy_loss": last["policy_loss"],
            "value_loss": last["value_loss"], "entropy": last["entropy"],
            "eval_mean_jct": eval_jct, "oracle_gap_pct": gap_pct,
        })
        handle.flush()
        if ep % 10 == 0 or ep == 1:
            print(f"ep={ep:4d} rho={rho:.2f} burst={int(bursty)} "
                  f"mean_jct={summary['mean_jct']:.0f} "
                  f"loss={last['loss']:.3f} ent={last['entropy']:.3f} eval={eval_jct}")

    handle.close()
    print(f"\nBest eval mean_jct: {best_eval:.1f}")
    with (result_dir / "train_summary.json").open("w") as f:
        json.dump({"best_eval_mean_jct": best_eval, "episodes": args.episodes}, f, indent=2)


if __name__ == "__main__":
    main()
