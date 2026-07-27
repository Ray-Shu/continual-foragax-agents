#!/bin/bash
# Render plasticity figures for R2-plasticity (PPO-ReLU vs RTU-PPO-ReLU).
# Assumes process_data.py has already produced
# results/R2-plasticity/.../data.parquet.

set -e

EXP=experiments/R4-exp_rtus/foragax/ForagaxSquareWaveTwoBiome-v11


# python src/learning_curve.py "$EXP" \
#     --metrics ewm_reward \
#     --plot-name ACMLP \
#     --filter-alg-apertures ActorCriticMLP:9 \
#     --end-frame 10000000 \
#     --legend-on-bar \
#     --plot-avg \
#     --horizontal-bars \
#     --disable-fov \
#     --font-size 20

# python src/learning_curve.py "$EXP" \
#     --metrics ewm_reward \
#     --plot-name RTACMLP_09 \
#     --filter-alg-apertures RealTimeActorCriticMLP_09:9 \
#     --end-frame 10000000 \
#     --legend-on-bar \
#     --plot-avg \
#     --horizontal-bars \
#     --disable-fov \
#     --font-size 20

# python src/learning_curve.py "$EXP" \
#     --metrics ewm_reward \
#     --plot-name RTACMLP_099 \
#     --filter-alg-apertures RealTimeActorCriticMLP_099:9 \
#     --end-frame 10000000 \
#     --legend-on-bar \
#     --plot-avg \
#     --horizontal-bars \
#     --disable-fov \
#     --font-size 20

# python src/learning_curve.py "$EXP" \
#     --metrics ewm_reward \
#     --plot-name RTACMLP_0999 \
#     --filter-alg-apertures RealTimeActorCriticMLP_0999:9 \
#     --end-frame 10000000 \
#     --legend-on-bar \
#     --plot-avg \
#     --horizontal-bars \
#     --disable-fov \
#     --font-size 20

python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --plot-name compare_3 \
    --filter-alg-apertures RealTimeActorCriticMLPReLU:9 RealTimeActorCriticMLP_09:9 RealTimeActorCriticMLP_099:9 RealTimeActorCriticMLP_0999:9 RealTimeActorCriticMLP_09999:9 \
    --end-frame 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --disable-fov \
    --font-size 20
