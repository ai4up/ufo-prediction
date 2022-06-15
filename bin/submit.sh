#!/bin/bash

#SBATCH --job-name=ml-training
#SBATCH --account=eubucco
#SBATCH --qos=short
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=245000
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=ml-training-%j.stdout
#SBATCH --error=ml-training-%j.stderr
#SBATCH --workdir=/p/tmp/floriann/ml-training

conda_env="uf-ml"
repo_dir="/p/projects/eubucco/ufo-prediction"

module load anaconda
module load cuda

conda env list | grep "$conda_env" || conda create -n "$conda_env" python=3.8

source activate "$conda_env"

# srun \
#   --partition io \
#   --qos io \
#   --time 5 \
#   --cpus-per-task=1 \
#   /bin/bash -c "
#   pip install -r "${repo_dir}/requirements.txt"; \
#   pip install -r "${repo_dir}/cluster-utils/requirements.txt""

if [[ -z "${PROFILE_MEMORY}" ]]; then
  python "${repo_dir}/bin/train.py"
else
  mprof run "${repo_dir}/bin/train.py"
  mprof plot --output mem_usage_over_time.png
fi
