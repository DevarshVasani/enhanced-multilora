"""
Behaviour-cloning warm start: train the policy to imitate the load-adaptive
teacher (expert.py) over the full utilisation sweep.

The teacher is strictly stronger than any Fixed-N(k) across the rho sweep
(ARCHITECTURE.md Theorems 1-2). PPO fine-tunes from here, so any improvement
over the teacher is attributable purely to online RL.

    python -m c_lora_sim.bc_pretrain --workloads 100 --epochs 8
    -> c_lora_sim/models/bc_init.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from c_lora_sim.expert import collect_bc_samples
from c_lora_sim.ppo_clora import evaluate, make_policy
from c_lora_sim.workload import generate_workload
from c_lora_sim.train_clora import EVAL_SPECS, RHO_LEVELS, P_BURSTY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workloads",   type=int,   default=100)
    p.add_argument("--epochs",      type=int,   default=8)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--hidden-dim",  type=int,   default=128)
    p.add_argument("--layers",      type=int,   default=3)
    p.add_argument("--num-gpus",    type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=7)
    p.add_argument("--feature-mode",default="v3", help="v1 | v2 | v3")
    p.add_argument("--p-deadline",  type=float, default=0.3)
    p.add_argument("--max-samples", type=int,   default=30_000,
                   help="Cap on BC decision samples (the per-sample loop is "
                        "CPU-overhead bound). 0 = no cap.")
    p.add_argument("--model-dir",   default="c_lora_sim/models_preemptive")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"collecting BC samples from load-adaptive teacher across {args.workloads} workloads ...")
    samples = []
    for k in range(args.workloads):
        rho    = random.choice(RHO_LEVELS)
        bursty = random.random() < P_BURSTY
        nj     = random.choice([40, 50, 60, 70, 80])
        jobs = generate_workload(num_jobs=nj, seed=random.randint(0, 2**31 - 1),
                                 target_rho=rho, bursty=bursty,
                                 p_deadline=args.p_deadline)
        # clairvoyant=False: BC and PPO share the same EMA-based feature
        # distribution, ensuring the warm-start transfers cleanly to RL. The
        # feature_mode must match the PPO policy's (Flaw 6).
        samples.extend(collect_bc_samples(jobs, num_gpus=args.num_gpus,
                                          clairvoyant=False,
                                          feature_mode=args.feature_mode))
    print(f"  {len(samples)} decision samples collected")
    # The per-sample (non-batched) BC loop is CPU-overhead bound, so cap the set
    # to keep each epoch tractable. A few tens of thousands of expert decisions is
    # ample to imitate the load-adaptive teacher.
    if args.max_samples and len(samples) > args.max_samples:
        samples = random.sample(samples, args.max_samples)
        print(f"  subsampled to {len(samples)} samples (--max-samples)")

    policy    = make_policy(hidden_dim=args.hidden_dim, num_layers=args.layers,
                            device=device, feature_mode=args.feature_mode)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        total_loss, correct = 0.0, 0
        optimizer.zero_grad()
        for i, (feats, target) in enumerate(samples, 1):
            ft     = torch.as_tensor(feats, dtype=torch.float32, device=device)
            logits, _ = policy.forward(ft)
            loss   = F.cross_entropy(logits.unsqueeze(0),
                                     torch.tensor([target], device=device))
            (loss / 16).backward()
            total_loss += float(loss.item())
            correct    += int(torch.argmax(logits).item() == target)
            if i % 16 == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()
        acc = correct / len(samples)
        eval_jct = 0
        print(f"epoch {epoch}: ce_loss={total_loss/len(samples):.4f} "
              f"imitation_acc={acc:.3f} eval_mean_jct={eval_jct:.0f}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "hidden_dim": args.hidden_dim, "layers": args.layers,
        "feature_mode": args.feature_mode,
        "source": "bc_load_adaptive",
    }, model_dir / "bc_init.pt")
    print(f"saved {model_dir/'bc_init.pt'}")


if __name__ == "__main__":
    main()
