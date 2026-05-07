EXP=experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1

python src/learning_curve.py $EXP \
    --plot-name fov_9/DQN_vs_baselines \
    --filter-alg-apertures DQN:9 \
    --metric ewm_reward_5
python src/learning_bar.py $EXP \
    --plot-name fov_9/DQN_vs_baselines_bar \
    --filter-alg-apertures DQN:9 \
    --metric ewm_reward_5
