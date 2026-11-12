#!/bin/bash
# FairVLMed: finish the SVM search that reached the time limit in job 19429774 (48 of 100 trials).
# 6 workers continue the same study (svm_auprc in optuna_raw.db) until 100 trials are complete.
#   cd ~/glaucoma/fairvlmed && PREPROC=raw sbatch run_svm_resume.slurm
#SBATCH --job-name=fv_svm
#SBATCH --partition=shared
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"; echo "worker $SLURM_ARRAY_TASK_ID  host: $(hostname)  PREPROC=${PREPROC:-stretch}  $(date)"
RESUME=1 python step3_auprc.py svm
echo "done $(date)"
