#!/bin/bash
# Render plasticity figures for R2-plasticity (PPO-ReLU vs RTU-PPO-ReLU).
# Assumes process_data.py has already produced
# results/R2-plasticity/.../data.parquet.

set -e

EXP=experiments/R5-tppos/foragax/ForagaxSquareWaveTwoBiome-v11


python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --plot-name ACMLP \
    --filter-alg-apertures TransformerPPO_test:9 TransformerPPO_layersTest:9 \
    --end-frame 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --disable-fov \
    --font-size 20
