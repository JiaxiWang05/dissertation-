#!/bin/bash
# Small-MLP Optuna search with 16 parallel workers sharing optuna_mlp_small.log (stops when 300 trials are complete).
# Create the study first:  python step3_auprc.py mlp_small   then:  sbatch run_mlp_workers.slurm
#SBATCH --job-name=fv_mlp
#SBATCH --partition=shared
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"; echo "worker $SLURM_ARRAY_TASK_ID  host: $(hostname)  $(date)"
WORKER=1 python step3_auprc.py mlp_small
echo "done $(date)"
