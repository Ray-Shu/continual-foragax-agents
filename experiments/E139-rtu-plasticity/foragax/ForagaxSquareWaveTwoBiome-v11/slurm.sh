#!/bin/bash
# Submit RTU-PPO at 30 seeds × 10M steps on ForagaxSquareWaveTwoBiome-v11.
# Mirrors the canonical RTU-PPO run-shape (vmap'd GPU, 32G, ~6h).
# scripts/slurm.py is idempotent — re-run after timeouts to fill missing seeds.

python scripts/slurm.py \
    --cluster clusters/vulcan-gpu-vmap-32G.json \
    --tasks 30 --time 06:00:00 --runs 30 --force \
    --entry src/rtu_ppo.py \
    -e experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/9/RealTimeActorCriticMLP.json
