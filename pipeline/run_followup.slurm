#!/bin/bash
# Follow-up to the 20 Oct meeting (CPU): optic-disc crop features, 90%-sensitivity threshold, Fairlearn, head ablation.
#   cd <pipeline folder> && sbatch run_followup.slurm
#SBATCH --job-name=rfg_followup
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/nobackup/%u/retfound_green/logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
ROOT=/nobackup/$USER/retfound_green; F=$ROOT/features; R=$ROOT/results
W=/nobackup/$USER/weights/retfoundgreen_statedict.pth; B=/nobackup/$USER/data/labels/disc_boxes.csv
cd "$SLURM_SUBMIT_DIR"; echo "code: $(pwd)  host: $(hostname)  $(date)"

# 1. input: crop around the optic disc (3 disc diameters)
for ds in papila chaksu; do
    [ -s "$F/${ds}_rfg_disc_crop.npz" ] || python extract_features.py --labels ${ds}_labels.csv --weights "$W" \
        --out "$F/${ds}_rfg_disc_crop.npz" --preproc disc_crop --boxes "$B" --crop-scale 3 --workers "$SLURM_CPUS_PER_TASK"
done
# 2. tuned logistic regression with the threshold at 90% sensitivity, for every input
for pp in resize fov_square disc_crop; do
    python run_linear_probe.py --papila-labels papila_labels.csv --papila-feats $F/papila_rfg_${pp}.npz \
        --chaksu-labels chaksu_labels.csv --chaksu-feats $F/chaksu_rfg_${pp}.npz \
        --protocol tuned --threshold sens90 --out $R/frozen_lr_tuned_${pp}_sens90 > /dev/null
    python plot_curves.py $R/frozen_lr_tuned_${pp}_sens90
done
# 3. fairness (Fairlearn) at both thresholds
for run in frozen_lr_tuned_resize frozen_lr_tuned_resize_sens90 frozen_lr_tuned_disc_crop_sens90; do
    python fairness.py --result $R/$run --papila-labels papila_labels.csv --out $R/fairness_$run > /dev/null
done
# 4. head ablation (0-3 hidden layers) from the finished Optuna studies
python ablation_optuna.py head --study-dir $R/optuna_head_resize --papila-labels papila_labels.csv \
    --chaksu-labels chaksu_labels.csv --papila-feats $F/papila_rfg_resize.npz --chaksu-feats $F/chaksu_rfg_resize.npz \
    --out $R/ablation_head
# 5. paired comparison of the inputs (each crop minus the squashed full image)
python compare_inputs.py --results $R --chaksu-labels chaksu_labels.csv --out $R/input_comparison
# 6. fairness mitigation: thresholds set on a few labelled Chakshu images per camera
python threshold_calibration.py --runs $R/frozen_lr_tuned_resize $R/frozen_lr_tuned_resize_sens90 \
    --chaksu-labels chaksu_labels.csv --out $R/threshold_calibration_resize
python threshold_calibration.py --runs $R/frozen_lr_tuned_disc_crop_sens90 \
    --chaksu-labels chaksu_labels.csv --out $R/threshold_calibration_disc_crop
# 7. single-number summaries at the chosen threshold (F1, balanced accuracy, Youden's J, MCC)
python combined_measures.py --results $R --chaksu-labels chaksu_labels.csv --out $R/combined_measures
echo "done $(date)"
