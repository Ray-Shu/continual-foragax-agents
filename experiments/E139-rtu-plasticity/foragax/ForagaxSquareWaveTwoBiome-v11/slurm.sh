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

    # Vanilla PPO baseline
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP.json

    # RTU-PPO with ReLU activations at every layer (dormancy + persistent
    # dormancy measured at each post-ReLU layer).
    # NOTE: run only AFTER the ReLU sweep + process_hypers has written the
    # selected hypers into this config (see ../../foragax-sweep/.../slurm.sh).
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPReLU.json

    # Vanilla PPO with ReLU everywhere (incl. the wide mid layer); dormancy +
    # persistent dormancy at all three post-ReLU sites.
    # NOTE: run only AFTER the ReLU sweep + process_hypers has written the
    # selected hypers into this config (see ../../foragax-sweep/.../slurm.sh).
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLPReLU.json
done
