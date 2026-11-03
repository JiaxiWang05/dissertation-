#!/bin/bash
# Rebuild every statistics table (CPU, no GPU training): linear probes, fairness, paired comparisons, ablation
# tables from their saved predictions, threshold calibration, single-number summaries. Run after a change to rfg.py.
#   cd <pipeline folder> && sbatch run_stats.slurm
#SBATCH --job-name=rfg_stats
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/nobackup/%u/retfound_green/logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
F=/nobackup/$USER/retfound_green/features; R=/nobackup/$USER/retfound_green/results
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SLURM_SUBMIT_DIR"; echo "code: $(pwd)  host: $(hostname)  $(date)"

probe() {  # preproc protocol threshold
    local out=$R/frozen_lr_$2_$1$([ "$3" = sens90 ] && echo _sens90)
    python run_linear_probe.py --papila-labels papila_labels.csv --papila-feats $F/papila_rfg_$1.npz \
        --chaksu-labels chaksu_labels.csv --chaksu-feats $F/chaksu_rfg_$1.npz --protocol $2 --threshold $3 \
        --out $out > /dev/null && python plot_curves.py $out
}
if [ "${SKIP_PROBES:-0}" != 1 ]; then
    probe resize paper youden & probe fov_square paper youden & probe resize tuned youden & probe fov_square tuned youden &
    wait
    probe resize tuned sens90 & probe fov_square tuned sens90 & probe disc_crop tuned sens90 &
    wait
fi
for run in frozen_lr_tuned_resize_sens90 frozen_lr_tuned_fov_square_sens90 frozen_lr_tuned_disc_crop_sens90; do
    [ -s $R/$run/curves.png ] && [ $R/$run/curves.png -nt $R/$run/metrics.json ] || python plot_curves.py $R/$run
done
for run in frozen_lr_tuned_resize frozen_lr_tuned_resize_sens90 frozen_lr_tuned_disc_crop_sens90; do
    python fairness.py --result $R/$run --papila-labels papila_labels.csv --out $R/fairness_$run > /dev/null
done
python compare_inputs.py --results $R --chaksu-labels chaksu_labels.csv --out $R/input_comparison > /dev/null
python ablation_optuna.py head --study-dir $R/optuna_head_resize --papila-labels papila_labels.csv \
    --chaksu-labels chaksu_labels.csv --predictions-from $R/ablation_head --out $R/ablation_head > /dev/null
[ -s $R/ablation_finetune/papila_oof_predictions.csv ] && python ablation_optuna.py finetune \
    --study-dir $R/optuna_finetune_resize --papila-labels papila_labels.csv --chaksu-labels chaksu_labels.csv \
    --baseline $R/frozen_lr_tuned_resize --predictions-from $R/ablation_finetune --out $R/ablation_finetune > /dev/null
python threshold_calibration.py --runs $R/frozen_lr_tuned_resize $R/frozen_lr_tuned_resize_sens90 \
    --chaksu-labels chaksu_labels.csv --out $R/threshold_calibration_resize > /dev/null
python threshold_calibration.py --runs $R/frozen_lr_tuned_disc_crop_sens90 \
    --chaksu-labels chaksu_labels.csv --out $R/threshold_calibration_disc_crop > /dev/null
python combined_measures.py --results $R --chaksu-labels chaksu_labels.csv --out $R/combined_measures > /dev/null
echo "done $(date)"
