#!/bin/bash
# 17 Nov GPU work. Cache:      MODE=cache sbatch run_gpu.slurm
# Workers (array):             MODE="cnn_cv.py chaksu" sbatch --array=0-1 run_gpu.slurm   (or "unfreeze_cv.py")
#SBATCH --job-name=n17gpu
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
cd "$SLURM_SUBMIT_DIR"; echo "host: $(hostname)  MODE=$MODE  task=${SLURM_ARRAY_TASK_ID:-}  $(date)"; nvidia-smi -L || true
if [ "$MODE" = cache ]; then python cache_images.py chaksu; python cache_images.py fairvlmed; else WORKER=1 python $MODE; fi
echo "done $(date)"
