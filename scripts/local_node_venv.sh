#!/bin/bash

#SBATCH --account=aip-whitem
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --export=path="/home/stang5/scratch/continual-foragax-agents"

module load python/3.11 arrow/19 gcc opencv rust swig

cp $path/pyproject.toml $SLURM_TMPDIR/
cd $SLURM_TMPDIR
python -m venv .venv
source .venv/bin/activate

pip install -e .

cp -r .venv $path/

pip freeze
