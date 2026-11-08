#!/bin/bash
# check_mlp_edges.py: one array task per setting (6 settings), then: python check_mlp_edges.py summary
# cd ~/glaucoma/after_3_Nov && sbatch run_edges.slurm
#SBATCH --job-name=edges
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"; echo "code: $(pwd)  host: $(hostname)  cell $SLURM_ARRAY_TASK_ID  $(date)"
CELL=$SLURM_ARRAY_TASK_ID python check_mlp_edges.py
echo "done $(date)"
