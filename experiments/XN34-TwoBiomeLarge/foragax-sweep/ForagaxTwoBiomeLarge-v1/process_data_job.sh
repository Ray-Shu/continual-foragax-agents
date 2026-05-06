#!/bin/bash
#SBATCH --account=rrg-whitem
#SBATCH --job-name=XN34-TwoBiomeLarge_foragax-sweep_ForagaxTwoBiomeLarge-v1_process_data
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=16
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

$SLURM_TMPDIR/.venv/bin/python src/process_data.py experiments/XN34-TwoBiomeLarge/foragax-sweep/ForagaxTwoBiomeLarge-v1
