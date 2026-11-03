#!/bin/bash
# Learning curve (CPU): tuned logistic regression on frozen RETFound-Green features with 25/50/100/200 training images.
#   cd <pipeline folder> && sbatch run_learning_curve.slurm
#SBATCH --job-name=rfg_learning_curve
#SBATCH --partition=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nobackup/%u/retfound_green/logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
F=/nobackup/$USER/retfound_green/features; R=/nobackup/$USER/retfound_green/results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"; echo "code: $(pwd)  host: $(hostname)  $(date)"
for pp in ${PREPROCS:-resize disc_crop}; do
    python learning_curve.py --papila-labels papila_labels.csv --papila-feats $F/papila_rfg_${pp}.npz \
        --chaksu-labels chaksu_labels.csv --chaksu-feats $F/chaksu_rfg_${pp}.npz --jobs "$SLURM_CPUS_PER_TASK" --out $R/learning_curve_${pp}
done
echo "done $(date)"
