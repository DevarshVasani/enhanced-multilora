#!/usr/bin/env bash
# Multi-training-seed variance study (addresses Critique 3: single-seed fragility).
#
# Each seed is an INDEPENDENT PPO run warm-started from the SAME BC teacher
# (bc_init.pt). The only thing that varies across runs is the PPO training seed,
# which controls: rollout action sampling, minibatch shuffling, and torch weight
# init of the freshly-added value head. This is exactly the source of variance the
# critique flags -- "did this checkpoint get lucky in gradient descent, or does the
# pipeline reliably converge to a margin-positive policy?"
#
# Memory-aware: this box has ~3GB free, and each torch process is ~0.8-1GB
# resident, so we cap concurrency at MAXJOBS instead of fanning out all seeds.
set -u
PY=.venv/bin/python
SEEDS=(42 1 2 3 4)
EPISODES=${EPISODES:-600}
MAXJOBS=${MAXJOBS:-2}
ROOT=c_lora_sim/seed_study

running=0
for s in "${SEEDS[@]}"; do
  mkdir -p ${ROOT}/models_s${s} ${ROOT}/results_s${s}
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY -m c_lora_sim.train_clora \
    --episodes $EPISODES --seed $s \
    --init-model c_lora_sim/models/bc_init.pt \
    --model-dir ${ROOT}/models_s${s} \
    --result-dir ${ROOT}/results_s${s} \
    > ${ROOT}/train_s${s}.log 2>&1 &
  running=$((running+1))
  if [ "$running" -ge "$MAXJOBS" ]; then
    wait -n            # wait for any one seed to finish before launching the next
    running=$((running-1))
  fi
done
wait
echo "ALL_SEEDS_DONE"
