#!/bin/bash
# Hyperparameter sweep for RTU-PPO with ReLU activations on
# ForagaxSquareWaveTwoBiome-v11, tuned at the short 1M horizon (tune-short /
# eval-long convention; the selected config runs at 10M in ../../foragax/).
#
# Grid (mirrors the tanh RTU sweep, X30-ForagaxSquareWaveTwoBiome): actor alpha
# x actor beta2 x critic lr_scale x critic beta2 x entropy_coef = 3*2*3*2*3 =
# 108 cells, x --runs seeds. gae_lambda fixed at 0.95.
#
# --tasks 5 vmaps 5 runs/GPU: d_hidden=512 + LayerNorm + the plasticity/grad-norm
# probes have the same per-run footprint as the 10M eval, which OOMs a single
# L40S above --tasks 5. scripts/slurm.py is idempotent — re-run after timeouts to
# fill missing seeds.

for fov in 9; do
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 03:00:00 --runs 5 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-rtu-plasticity/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPReLU.json
done
