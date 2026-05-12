EXP=experiments/XN34-TwoBiomeLarge/foragax/ForagaxTwoBiomeLarge-v1

'''
# Plot AUC vs Field of View across all FOVs
python src/auc_fov.py $EXP \
    --plot-name auc_fov


# Dont need this for now. 

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
'''

# Plot learning curves for all DQN FOV variants on the same plot
python src/learning_curve.py $EXP \
    --plot-name dqn_all_fovs \
    --filter-alg-apertures DQN:3 DQN:5 DQN:7 DQN:9 DQN:11 DQN:13 DQN:15 Search-Oracle Search-Nearest Random \
    --metric ewm_reward_5 \
    --legend
