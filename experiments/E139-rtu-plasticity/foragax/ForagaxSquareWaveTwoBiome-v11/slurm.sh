#!/bin/bash
# Submit RTU-PPO and the vanilla PPO baseline at 30 seeds × 10M steps on
# ForagaxSquareWaveTwoBiome-v11.
# --tasks 5 vmaps 5 seeds/GPU -> 6 jobs (d_hidden=512 + LayerNorm + the
# plasticity/grad-norm probes OOM a single L40S at --tasks 10/30).
# scripts/slurm.py is idempotent — re-run after timeouts to fill missing seeds,
# and an agent whose seeds are all done schedules nothing.

for fov in 9; do
    # RTU-PPO.
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLP.json

    # Vanilla PPO baseline (feedforward; uses the same rtu_ppo.py trainer so the
    # plasticity/grad-norm probes are collected). Lighter than the RTU, so
    # --tasks could likely go to 10; kept at 5 to match the RTU and stay safe.
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP.json
done
