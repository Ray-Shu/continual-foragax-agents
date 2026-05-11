#!/bin/bash
#SBATCH --account=aip-whitem
#SBATCH --job-name=XN34-TwoBiomeLarge_foragax-sweep_ForagaxTwoBiomeLarge-v1_hypers
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --output=../slurm-%j.out
#SBATCH --time=01:00:00

module load arrow/19

cp -R .venv $SLURM_TMPDIR

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORMS=cpu

$SLURM_TMPDIR/.venv/bin/python experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1/hypers.py

for fov in 3 5 7 9 11 13 15; do
    $SLURM_TMPDIR/.venv/bin/python scripts/generate_frozen_configs.py experiments/XN34-TwoBiomeLarge/foragax/ForagaxTwoBiomeLarge-v1/${fov}
done
