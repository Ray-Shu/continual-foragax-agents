#!/bin/bash

EXP=experiments/R5-tppos/foragax/ForagaxSquareWaveTwoBiome-v11

for fov in 9; do
    # PPO
    python scripts/slurm.py \
       --cluster clusters/vulcan-gpu-vmap-32G.json \
       --tasks 5 --time 12:00:00 --runs 30 --force \
       --entry src/tppo.py \
       -e ${EXP}/${fov}/TransformerPPO_layersTest2.json
done
