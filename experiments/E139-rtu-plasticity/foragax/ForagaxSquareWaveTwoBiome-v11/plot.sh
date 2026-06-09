#!/bin/bash
# Render plasticity figures for E139 RTU-PPO. Assumes process_data.py has
# already produced results/E139-rtu-plasticity/.../data.parquet.

set -e

EXP=experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward curve (sanity / baseline).
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLP:9 \
    --end-frame 10000000 \
    --vertical-lines 500000 1000000 1500000 2000000 2500000 3000000 3500000 4000000 4500000 5000000 5500000 6000000 6500000 7000000 7500000 8000000 8500000 9000000 9500000

# Effective feature rank at all 6 probe sites.
python src/learning_curve.py "$EXP" \
    --metrics eff_rank_actor_pre1 eff_rank_critic_pre1 eff_rank_actor_rtu eff_rank_critic_rtu eff_rank_actor_pre2 eff_rank_critic_pre2 \
    --filter-alg-apertures RealTimeActorCriticMLP:9 \
    --end-frame 10000000

# Tanh saturation rate at the 4 pre-tanh probe sites (paper's "dormancy" under tanh).
python src/learning_curve.py "$EXP" \
    --metrics sat_rate_actor_pre1 sat_rate_critic_pre1 sat_rate_actor_pre2 sat_rate_critic_pre2 \
    --filter-alg-apertures RealTimeActorCriticMLP:9 \
    --end-frame 10000000

# Sokar τ-dormant fraction at the 2 post-ReLU RTU-output probe sites.
python src/learning_curve.py "$EXP" \
    --metrics dormant_actor_rtu dormant_critic_rtu \
    --filter-alg-apertures RealTimeActorCriticMLP:9 \
    --end-frame 10000000
