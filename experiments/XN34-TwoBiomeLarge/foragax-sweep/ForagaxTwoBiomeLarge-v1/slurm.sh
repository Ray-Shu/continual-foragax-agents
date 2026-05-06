for fov in 3 5 7 9 11 13 15; do
    python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1/${fov}/DQN.json
done

python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1/Baselines/Search-Oracle.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1/Baselines/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1/Baselines/Random.json
