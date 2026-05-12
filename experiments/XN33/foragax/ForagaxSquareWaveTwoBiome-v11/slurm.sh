for fov in 9;
do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 06:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/XN33/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/DQN.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 06:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/XN33/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/DQN_ReDo.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 06:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/XN33/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/DQN_ReDo_PreActLN.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 06:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/XN33/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/DQN_ReDo_PostLNScore.json
done
