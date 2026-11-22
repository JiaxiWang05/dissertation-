#!/bin/bash
# 22 Nov GPU jobs.  CMD="python layer_features.py" sbatch run_gpu.slurm
#                   CMD="python grounding_cv.py" CFG=mask1 sbatch run_gpu.slurm
#SBATCH --job-name=n22gpu
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl_1g.18gb:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
cd "$SLURM_SUBMIT_DIR"; echo "host: $(hostname)  CMD=$CMD  CFG=${CFG:-}  $(date)"; nvidia-smi -L || true
for c in ${CFG:-none}; do CFG=$c $CMD; done
echo "done $(date)"
