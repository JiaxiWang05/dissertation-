#!/bin/bash
# 24 Nov GPU jobs.  CMD="python small_cnn.py" WORKER=1 sbatch --array=0-3 run_gpu.slurm   (Optuna workers)
#                   CMD="python vit_runs.py" CURVES=1 MODEL=lora sbatch run_gpu.slurm
#SBATCH --job-name=n24gpu
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl_1g.18gb:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
cd "$SLURM_SUBMIT_DIR"; echo "host: $(hostname)  CMD=$CMD  CFG=${CFG:-}  $(date)"; nvidia-smi -L || true
for c in ${CFG:-none}; do CFG=$c $CMD; done
echo "done $(date)"
