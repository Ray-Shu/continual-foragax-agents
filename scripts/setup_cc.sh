#!/bin/bash

echo "scheduling a job to install project dependencies"
sbatch --ntasks=1 --mem-per-cpu="12G" --export=path="$(pwd)" scripts/local_node_venv.sh
