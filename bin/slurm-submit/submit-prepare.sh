#!/bin/bash

#SBATCH --job-name=fts-eng-prepare
#SBATCH --account=eubucco
#SBATCH --qos=medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=245000
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=fts-eng-prepare-%j.stdout
#SBATCH --error=fts-eng-prepare-%j.stderr
#SBATCH --workdir=/p/tmp/floriann/fts-eng-prepare

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
  python "${repo_dir}/bin/prepare.py"
else
  mprof run --exit-code --output "mprofile_${SLURM_JOBID}.dat" "${repo_dir}/bin/prepare.py"
fi
