#!/bin/bash
# Render plasticity figures for E139 RTU-PPO. Assumes process_data.py has
# already produced results/E139-rtu-plasticity/.../data.parquet.

set -e

EXP=experiments/E139-rtu-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward switches every 250k steps (square wave: half of the 500k period).
# Use %.0f so macOS/BSD seq emits plain integers (it defaults to %g, which
# renders e.g. 1000000 as "1e+06" and breaks --vertical-lines int parsing).
SWITCHES=$(seq -f "%.0f" 250000 250000 9750000)

# Reward curve (sanity / baseline).
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLP:9 \
    --end-frame 10000000 \
    --vertical-lines $SWITCHES

# Full-run actor-vs-critic curves (absolute time), one subplot per layer, with
# environment switches marked by faint dotted vertical lines. Overview companion
# to the switch-triggered folds below. eff_rank is width-normalized.
python src/plasticity_compare.py "$EXP"

# Switch-triggered ("peri-switch") plasticity analysis. Every metric family is
# folded onto steps-relative-to-switch, centered so the pre-switch plateau is on
# screen next to tau=0: effective rank (width-normalized: pre1/pre2 = 64, RTU =
# 2*d_hidden = 1024), tanh saturation rate, and RTU dormancy. Actor vs critic.
#   - fold:         canonical average response over a mid-training window.
#   - fold-overlay: the same fold at early/mid/late windows, to expose how the
#                   transient changes over training (plasticity loss).
# Biomes are symmetric, so switches are folded together on the 250k half-period.
python src/plasticity_compare.py "$EXP" --mode fold --window 4500000:5500000:500
python src/plasticity_compare.py "$EXP" --mode fold-overlay \
    --windows 1000000:2000000:500 4500000:5500000:500 9000000:10000000:500 \
    --window-labels early mid late

# Per-layer gradient norms (l0/l1/l2), first-rollout-normalized and
# parameter-count-weighted. Reads the per-seed .npz directly (not the parquet),
# so run this where the raw results live.
python src/grad_norm_curve.py "$EXP" --log-scale
