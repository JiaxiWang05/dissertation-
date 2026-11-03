#!/bin/bash
# Unfreezing ablation (GPU): best tuned setting for 0/1/2/4/6/12 unfrozen blocks, from the finished Optuna studies.
#SBATCH --job-name=rfg_ablation_ft
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=04:00:00
#SBATCH --output=/nobackup/%u/retfound_green/logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
R=/nobackup/$USER/retfound_green/results
cd "$SLURM_SUBMIT_DIR"; echo "code: $(pwd)  host: $(hostname)  $(date)"; nvidia-smi -L
python ablation_optuna.py finetune --study-dir $R/optuna_finetune_resize --papila-labels papila_labels.csv \
    --chaksu-labels chaksu_labels.csv --weights /nobackup/$USER/weights/retfoundgreen_statedict.pth --baseline $R/frozen_lr_tuned_resize \
    --out $R/ablation_finetune
