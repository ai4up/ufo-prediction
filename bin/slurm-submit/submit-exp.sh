#!/bin/bash

#SBATCH --job-name=ml-exp
#SBATCH --account=eubucco
#SBATCH --qos=gpushort
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=120000
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=ml-exp-%j.stdout
#SBATCH --error=ml-exp-%j.stderr
#SBATCH --workdir=/p/tmp/floriann/ml-exp

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

until [ -f /p/tmp/floriann/ml-exp/start.txt ]
do
     echo "Waiting for file to be created..."
     sleep 60
done

export METHOD='regression'

if [[ -z "${PROFILE_MEMORY}" ]]; then
  echo "Memory profiling deactivated."
  # export RQ=1; python "${repo_dir}/bin/run_experiments.py"
  # export RQ=2; python "${repo_dir}/bin/run_experiments.py"
  # export RQ=3; python "${repo_dir}/bin/run_experiments.py"
  # export RQ=spatial-distance; python "${repo_dir}/bin/run_experiments.py"
  # export RQ=additional-data; python "${repo_dir}/bin/run_experiments.py"
  # export RQ=SI; python "${repo_dir}/bin/run_experiments.py"
  export RQ=construction-type; python "${repo_dir}/bin/run_experiments.py"
else
  echo "Memory profiling activated."
  # export RQ=1; mprof run --output "mprofile_${SLURM_JOBID}_${RQ}.dat" "${repo_dir}/bin/run_experiments.py"
  # export RQ=2; mprof run --output "mprofile_${SLURM_JOBID}_${RQ}.dat" "${repo_dir}/bin/run_experiments.py"
  # export RQ=3; mprof run --output "mprofile_${SLURM_JOBID}_${RQ}.dat" "${repo_dir}/bin/run_experiments.py"
  # export RQ=spatial-distance; mprof run --output "mprofile_${SLURM_JOBID}_${RQ}.dat" "${repo_dir}/bin/run_experiments.py"
  # export RQ=additional-data; mprof run --output "mprofile_${SLURM_JOBID}_${RQ}.dat" "${repo_dir}/bin/run_experiments.py"
  # export RQ=SI; mprof run --output "mprofile_${SLURM_JOBID}_${RQ}.dat" "${repo_dir}/bin/run_experiments.py"
  export RQ=construction-type; mprof run --output "mprofile_${SLURM_JOBID}_${RQ}.dat" "${repo_dir}/bin/run_experiments.py"
fi
