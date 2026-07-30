#!/bin/bash
set -e

EXP=experiments/R4-sanity/foragax/ForagaxSquareWaveTwoBiome-v11

python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --plot-name double_check_20m \
    --filter-alg-apertures RealTimeActorCriticMLPReLU:9 RealTimeActorCriticMLP_decay:9 \
    --end-frame 20000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --disable-fov \
    --font-size 20
