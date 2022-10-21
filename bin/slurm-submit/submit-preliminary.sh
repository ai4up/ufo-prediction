#!/bin/bash

#SBATCH --job-name=ml-preliminary
#SBATCH --account=eubucco
#SBATCH --qos=short
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=55000
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=ml-preliminary-%j.stdout
#SBATCH --error=ml-preliminary-%j.stderr
#SBATCH --workdir=/p/tmp/floriann/ml-preliminary

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
  export RQ=prelim; python "${repo_dir}/bin/run_experiments.py"
else
  export RQ=prelim; mprof run --output "mprofile_${SLURM_JOBID}_${RQ}.dat" "${repo_dir}/bin/run_experiments.py"
fi
