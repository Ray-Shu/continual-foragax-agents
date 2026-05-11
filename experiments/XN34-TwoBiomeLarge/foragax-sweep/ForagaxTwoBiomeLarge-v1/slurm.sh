for fov in 3 5 7 9 11 13 15; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1/${fov}/DQN.json
done
