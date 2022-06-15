#!/bin/bash

#SRUN --qos=io
#SRUN --partition=io
#SRUN --time=00:05:00
#SRUN --nodes=1
#SRUN --ntasks=1
#SRUN --cpus-per-task=1
#SRUN --output=update-deps-%j.stdout
#SRUN --error=update-deps-%j.stderr

conda_env="uf-ml"
repo_dir="/p/projects/eubucco/ufo-prediction"

module load anaconda

conda env list | grep "$conda_env" || conda create -n "$conda_env" python=3.8

source activate "$conda_env"

pip install -r "${repo_dir}/cluster-utils/requirements.txt"
pip install -r "${repo_dir}/requirements.txt"
