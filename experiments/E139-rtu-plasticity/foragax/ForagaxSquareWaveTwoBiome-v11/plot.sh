#!/bin/bash
# Render plasticity figures for E139 RTU-PPO. Assumes process_data.py has
# already produced results/E139-rtu-plasticity/.../data.parquet.

set -e

EXP=experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward switches every 250k steps (square wave: half of the 500k period).
SWITCHES=$(seq 250000 250000 9750000)

# Reward curve (sanity / baseline).
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLP:9 \
    --end-frame 10000000 \
    --vertical-lines $SWITCHES

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

# Per-layer gradient norms (l0/l1/l2), first-rollout-normalized and
# parameter-count-weighted. Reads the per-seed .npz directly (not the parquet),
# so run this where the raw results live.
python src/grad_norm_curve.py "$EXP"
