#!/bin/bash
# RETFound-Green frozen features -> logistic regression (PAPILA CV + Chaksu external).
#   sbatch run_pipeline.slurm
#SBATCH --job-name=rfg_frozen_lr
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nobackup/%u/retfound_green/logs/%x_%j.out

set -euo pipefail
ROOT=/nobackup/$USER/retfound_green
PY=$ROOT/env/bin/python
CODE=$ROOT/pipeline
W=$ROOT/weights/retfoundgreen_statedict.pth
cd "$CODE"

for ds in papila chaksu; do
    if [ ! -s "$ROOT/features/${ds}_rfg.npz" ]; then
        $PY extract_features.py --labels ${ds}_labels.csv --weights "$W" \
            --out "$ROOT/features/${ds}_rfg.npz" --workers "$SLURM_CPUS_PER_TASK"
    fi
done

$PY run_linear_probe.py \
    --papila-labels papila_labels.csv --papila-feats "$ROOT/features/papila_rfg.npz" \
    --chaksu-labels chaksu_labels.csv --chaksu-feats "$ROOT/features/chaksu_rfg.npz" \
    --out "$ROOT/results/rfg_frozen_lr"
