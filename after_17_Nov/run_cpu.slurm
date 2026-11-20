#!/bin/bash
# 17 Nov tasks, CPU part.   STEPS="weight_search.py grid" sbatch run_cpu.slurm
#SBATCH --job-name=n17cpu
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"; echo "host: $(hostname)  $(date)"
IFS=';' read -ra RUN <<< "$STEPS"
for s in "${RUN[@]}"; do echo "===== $s  $(date +%T)"; python $s; done
echo "done $(date)"
