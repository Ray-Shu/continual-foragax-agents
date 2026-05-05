python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-reward-trace-frozen-nonfrozen --grid 1,3 ActorCriticMLP-reward-trace_frozen_5M:5+ActorCriticMLP-reward-trace:5 ActorCriticMLP-reward-trace_frozen_5M:9+ActorCriticMLP-reward-trace:9 ActorCriticMLP-reward-trace_frozen_5M:15+ActorCriticMLP-reward-trace:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-vanilla-reward-trace-frozen-nonfrozen --grid 1,2 ActorCriticMLP_frozen_5M:9+ActorCriticMLP:9 ActorCriticMLP-reward-trace_frozen_5M:9+ActorCriticMLP-reward-trace:9 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name rtu-mlp-conv-frozen-nonfrozen --grid 1,2 RealTimeActorCriticMLP_frozen_5M:9+RealTimeActorCriticMLP:9 RealTimeActorCriticConv_frozen_5M:9+RealTimeActorCriticConv:9 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-vanilla-reward-trace-rtu-frozen --grid 1,1 RealTimeActorCriticMLP_frozen_5M:9+ActorCriticMLP_frozen_5M:9+ActorCriticMLP-reward-trace_frozen_5M:9 --legend

python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-rt-rtu-frozen-nonfrozen --grid 1,3 ActorCriticMLP_frozen_5M:9+ActorCriticMLP:9 ActorCriticMLP-reward-trace_frozen_5M:9+ActorCriticMLP-reward-trace:9 RealTimeActorCriticMLP_frozen_5M:9+RealTimeActorCriticMLP:9 --legend

# python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name rtu --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP:5 RealTimeActorCriticMLP:9 RealTimeActorCriticMLP:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name rtu --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP:5 RealTimeActorCriticMLP:9 RealTimeActorCriticMLP:15 --legend

# python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name rtu-frozen --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP_frozen_5M:5 RealTimeActorCriticMLP_frozen_5M:9 RealTimeActorCriticMLP_frozen_5M:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name rtu-frozen --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP_frozen_5M:5 RealTimeActorCriticMLP_frozen_5M:9 RealTimeActorCriticMLP_frozen_5M:15 --legend

# python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo --filter-alg-apertures Search-Oracle ActorCriticMLP:5 ActorCriticMLP:9 ActorCriticMLP:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo --filter-alg-apertures Search-Oracle ActorCriticMLP:5 ActorCriticMLP:9 ActorCriticMLP:15 --legend

# python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-l2 --filter-alg-apertures Search-Oracle ActorCriticMLP-l2:5 ActorCriticMLP-l2:9 ActorCriticMLP-l2:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-l2 --filter-alg-apertures Search-Oracle ActorCriticMLP-l2:5 ActorCriticMLP-l2:9 ActorCriticMLP-l2:15 --legend

# python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-reward-trace --filter-alg-apertures Search-Oracle ActorCriticMLP-reward-trace:5 ActorCriticMLP-reward-trace:9 ActorCriticMLP-reward-trace:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-reward-trace --filter-alg-apertures Search-Oracle ActorCriticMLP-reward-trace:5 ActorCriticMLP-reward-trace:9 ActorCriticMLP-reward-trace:15 --legend

# python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-reward-trace-frozen --filter-alg-apertures Search-Oracle ActorCriticMLP-reward-trace_frozen_5M:5 ActorCriticMLP-reward-trace_frozen_5M:9 ActorCriticMLP-reward-trace_frozen_5M:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ppo-reward-trace-frozen --filter-alg-apertures Search-Oracle ActorCriticMLP-reward-trace_frozen_5M:5 ActorCriticMLP-reward-trace_frozen_5M:9 ActorCriticMLP-reward-trace_frozen_5M:15 --legend

python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name dqn --filter-alg-apertures Search-Oracle DQN:5 DQN:9 DQN:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name dqn_frozen_5M --filter-alg-apertures Search-Oracle DQN_frozen_5M:5 DQN_frozen_5M:9 DQN_frozen_5M:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name dqn-l2 --filter-alg-apertures Search-Oracle DQN_L2:5 DQN_L2:9 DQN_L2:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name dqn-l2-init --filter-alg-apertures Search-Oracle DQN_L2_Init:5 DQN_L2_Init:9 DQN_L2_Init:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name dqn-ln --filter-alg-apertures Search-Oracle DQN_LN:5 DQN_LN:9 DQN_LN:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name dqn-hare-and-tortoise --filter-alg-apertures Search-Oracle DQN_Hare_and_Tortoise:5 DQN_Hare_and_Tortoise:9 DQN_Hare_and_Tortoise:15 --legend
python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name dqn-crelu --filter-alg-apertures Search-Oracle DQN_CReLU:5 DQN_CReLU:9 DQN_CReLU:15 --legend

python src/learning_curve.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ForagaxSquareWaveTwoBiome-v11-learning-curve-pt-dqn --filter-alg-apertures PT_DQN:9 PT_DQN_64:9 --metric ewm_reward_3 --ylim "-0.25" 1.65
python src/learning_bar.py experiments/X32-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name ForagaxSquareWaveTwoBiome-v11-bar-plot-pt-dqn --filter-alg-apertures PT_DQN:9 PT_DQN_64:9
