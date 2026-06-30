#!/bin/bash
# Render plasticity figures for R2-plasticity (PPO-ReLU vs RTU-PPO-ReLU).
# Assumes process_data.py has already produced
# results/R2-plasticity/.../data.parquet.

set -e

EXP=experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward curve (sanity / baseline), RTU vs PPO overlaid. Top-level cross-alg
# comparison; the per-alg plasticity figures go in plots/<alg>/ below.
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLPReLU:9 ActorCriticMLPReLU:9 \
    --end-frame 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --disable-fov \
    --font-size 20
