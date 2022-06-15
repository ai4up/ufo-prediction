#!/bin/bash

#SBATCH --job-name=ml-training
#SBATCH --account=eubucco
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=ml-training-%j.stdout
#SBATCH --error=ml-training-%j.stderr
#SBATCH --workdir=/p/tmp/floriann/ml-training

conda_env="uf-ml"
repo_dir="/p/projects/eubucco/ufo-prediction"

srun --partition io --qos io "${repo_dir}/bin/update-dependencies.sh"

module load anaconda

source activate "$conda_env"

if [[ -z "${PROFILE_MEMORY}" ]]; then
  python "${repo_dir}/bin/train.py"
else
  mprof run "${repo_dir}/bin/train.py"
  mprof plot --output mem_usage_over_time.png
fi
