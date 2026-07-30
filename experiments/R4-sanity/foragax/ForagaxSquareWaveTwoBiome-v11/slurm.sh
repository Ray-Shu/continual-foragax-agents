#!/bin/bash

EXP=experiments/R4-sanity/foragax/ForagaxSquareWaveTwoBiome-v11

for fov in 9; do
    python scripts/slurm.py \
       --cluster clusters/vulcan-gpu-vmap-32G.json \
       --tasks 3 --time 06:00:00 --runs 30 --force \
       --entry src/rtu_ppo.py \
       -e ${EXP}/${fov}/RealTimeActorCriticMLPReLU.json

    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 3 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e ${EXP}/${fov}/RealTimeActorCriticMLP_decay.json
done
