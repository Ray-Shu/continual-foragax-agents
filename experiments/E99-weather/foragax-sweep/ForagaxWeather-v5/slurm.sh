python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=1:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/9/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=1:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/9/DQN_CReLU.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=1:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/9/DQN_Hare_and_Tortoise.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=1:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/9/DQN_L2.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=1:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/9/DQN_L2_Init.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=1:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/9/DQN_LN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-2G.json --time=1:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/9/DQN_Reset_Head.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=1:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/9/DQN_Shrink_and_Perturb.json

python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=2:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/15/DQN.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=2:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/15/DQN_privileged.json
python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap.json --time=2:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E99-weather/foragax-sweep/ForagaxWeather-v5/15/DQN_world.json
