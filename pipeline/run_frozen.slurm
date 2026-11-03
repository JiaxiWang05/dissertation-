#!/bin/bash
# Step 1 + 2a: frozen RETFound-Green features -> logistic regression (paper protocol and tuned) and
# the Optuna-tuned head, for both preprocessings. CPU only, about 30 min.
#   cd <pipeline folder> && sbatch run_frozen.slurm
#SBATCH --job-name=rfg_frozen
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/nobackup/%u/retfound_green/logs/%x_%j.out
set -euo pipefail
set +u  # conda activation scripts read unset variables
source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh
conda activate glaucoma
set -u
ROOT=/nobackup/$USER/retfound_green
W=/nobackup/$USER/weights/retfoundgreen_statedict.pth
cd "$SLURM_SUBMIT_DIR"
echo "code: $(pwd)  host: $(hostname)  $(date)"

for pp in ${PREPROCS:-resize fov_square}; do
    for ds in papila chaksu; do
        f=$ROOT/features/${ds}_rfg_${pp}.npz
        [ -s "$f" ] || python extract_features.py --labels ${ds}_labels.csv --weights "$W" --out "$f" \
            --preproc $pp --workers "$SLURM_CPUS_PER_TASK"
    done
    for proto in paper tuned; do
        python run_linear_probe.py --papila-labels papila_labels.csv --papila-feats $ROOT/features/papila_rfg_${pp}.npz \
            --chaksu-labels chaksu_labels.csv --chaksu-feats $ROOT/features/chaksu_rfg_${pp}.npz \
            --protocol $proto --out $ROOT/results/frozen_lr_${proto}_${pp}
        python plot_curves.py $ROOT/results/frozen_lr_${proto}_${pp}
    done
    python optuna_head.py --papila-labels papila_labels.csv --papila-feats $ROOT/features/papila_rfg_${pp}.npz \
        --chaksu-labels chaksu_labels.csv --chaksu-feats $ROOT/features/chaksu_rfg_${pp}.npz \
        --compare-to $ROOT/results/frozen_lr_tuned_${pp} --n-trials 40 --threads 4 --out $ROOT/results/optuna_head_${pp}
done
