#!/bin/bash
# Step 2b: Optuna over unfrozen RETFound-Green blocks + head (nested on PAPILA's official folds), then
# tune on all PAPILA and test on Chaksu. One GPU, a few hours; resumes if resubmitted.
#   cd <pipeline folder> && sbatch run_optuna_finetune.slurm
# Hamilton: don't request CPUs or memory for GPU jobs. `--gres=gpu:1` is a 1/8 H200 slice (enough for
# ViT-S) but those are often all busy; a whole H200 (`gpu:h200_nvl:1`) usually starts at once.
#SBATCH --job-name=rfg_optuna_ft
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=12:00:00
#SBATCH --output=/nobackup/%u/retfound_green/logs/%x_%j.out
set -euo pipefail
set +u  # conda activation scripts read unset variables
source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh
conda activate glaucoma
set -u
ROOT=/nobackup/$USER/retfound_green
cd "$SLURM_SUBMIT_DIR"
echo "code: $(pwd)  host: $(hostname)  $(date)"; nvidia-smi -L
python optuna_finetune.py --papila-labels papila_labels.csv --chaksu-labels chaksu_labels.csv \
    --weights /nobackup/$USER/weights/retfoundgreen_statedict.pth --n-trials ${N_TRIALS:-30} \
    --workers 4 --out $ROOT/results/optuna_finetune_resize
