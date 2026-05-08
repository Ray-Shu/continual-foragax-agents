EXP=experiments/XN34-TwoBiomeLarge/foragax/ForagaxTwoBiomeLarge-v1

# Per-FOV learning curves of DQN against the three baselines
for fov in 9; do
    python src/learning_curve.py $EXP \
        --plot-name fov_${fov}/DQN_vs_baselines \
        --filter-alg-apertures DQN:${fov} Search-Oracle Search-Nearest Random \
        --metric ewm_reward_5
    python src/learning_bar.py $EXP \
        --plot-name fov_${fov}/DQN_vs_baselines_bar \
        --filter-alg-apertures DQN:${fov} Search-Oracle Search-Nearest Random \
        --metric ewm_reward_5
done
