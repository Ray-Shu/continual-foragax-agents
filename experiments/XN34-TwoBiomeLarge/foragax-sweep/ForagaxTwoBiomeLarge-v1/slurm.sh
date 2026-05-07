for fov in 7; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 00:30:00 --runs 10 --entry src/continuing_main.py --force -e experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1/${fov}/DQN.json
done
