#!/bin/bash
# FairVLMed: label table, then RETFound-Green features for both inputs on one GPU slice.
#   cd ~/glaucoma/fairvlmed && sbatch run_extract.slurm
#SBATCH --job-name=fv_extract
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
cd "$SLURM_SUBMIT_DIR"; echo "host: $(hostname)  $(date)"; nvidia-smi -L || true
[ -s fairvlmed_labels.csv ] || python make_labels.py
for pp in raw stretch; do python extract_features.py --preproc $pp --workers 2; done
echo "done $(date)"
