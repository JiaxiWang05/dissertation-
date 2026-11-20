#!/bin/bash
# Small-MLP search with F1 objective and the class weight searched, 8 workers sharing optuna_weight_f1.log (200 trials).
#SBATCH --job-name=n17f1
#SBATCH --partition=shared
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$SLURM_SUBMIT_DIR"; WORKER=1 python weight_search.py optuna
