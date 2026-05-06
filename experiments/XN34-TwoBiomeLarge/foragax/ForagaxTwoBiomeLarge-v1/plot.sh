EXP=experiments/XN34-TwoBiomeLarge/foragax/ForagaxTwoBiomeLarge-v1

# Per-FOV learning curves of DQN against the three baselines
for fov in 3 5 7 9 11 13 15; do
    python src/learning_curve.py $EXP \
        --plot-name fov_${fov}/DQN_vs_baselines \
        --filter-alg-apertures DQN:${fov} Search-Oracle Search-Nearest Random \
        --metric ewm_reward_5
    python src/learning_bar.py $EXP \
        --plot-name fov_${fov}/DQN_vs_baselines_bar \
        --filter-alg-apertures DQN:${fov} Search-Oracle Search-Nearest Random \
        --metric ewm_reward_5
done

# DQN compared across all FOVs (the figure-style FOV sweep)
python src/learning_curve.py $EXP \
    --plot-name DQN_fov_sweep \
    --filter-alg-apertures \
        DQN:3 DQN:5 DQN:7 DQN:9 DQN:11 DQN:13 DQN:15 \
        Search-Oracle Search-Nearest Random \
    --metric ewm_reward_5
python src/learning_bar.py $EXP \
    --plot-name DQN_fov_sweep_bar \
    --filter-alg-apertures \
        DQN:3 DQN:5 DQN:7 DQN:9 DQN:11 DQN:13 DQN:15 \
        Search-Oracle Search-Nearest Random \
    --metric ewm_reward_5
