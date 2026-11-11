#!/bin/bash
# FairVLMed: input choice, then logistic regression / SVM / kNN searches (AUPRC objective). CPU.
#   cd ~/glaucoma/fairvlmed && sbatch run_cv.slurm      (MLP: run_mlp_workers.slurm; comparison: STEPS=step5_compare.py)
#SBATCH --job-name=fv_cv
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"; echo "host: $(hostname)  PREPROC=${PREPROC:-stretch}  $(date)"
if [ -n "${STEPS:-}" ]; then IFS=';' read -ra RUN <<< "$STEPS"; else
    RUN=("step3_auprc.py logreg" "step3_auprc.py knn" "step3_auprc.py svm"); fi
for s in "${RUN[@]}"; do echo "===== $s  $(date +%T)"; python $s; done
echo "done $(date)"
