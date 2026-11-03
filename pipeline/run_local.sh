#!/bin/bash
# Same as run_frozen.slurm, on the laptop (low priority so the Mac stays usable).
# Expects images under $DATA (same layout as /nobackup/tggq61/data on Hamilton).
#   bash run_local.sh
set -euo pipefail
DATA=${DATA:-$HOME/data/dissertation}
PY=${PY:-$HOME/.venvs/rfg/bin/python}
W=$DATA/weights/retfoundgreen_statedict.pth
F=$DATA/features
R=$(cd "$(dirname "$0")" && pwd)/results
run() { taskpolicy -b "$PY" "$@"; }
cd "$(dirname "$0")"
for pp in ${PREPROCS:-resize fov_square}; do
    for ds in papila chaksu; do
        [ -s "$F/${ds}_rfg_${pp}.npz" ] || run extract_features.py --labels ${ds}_labels.csv --weights "$W" \
            --out "$F/${ds}_rfg_${pp}.npz" --preproc $pp --data-root "$DATA" --workers 2
    done
    for proto in paper tuned; do
        run run_linear_probe.py --papila-labels papila_labels.csv --papila-feats "$F/papila_rfg_${pp}.npz" \
            --chaksu-labels chaksu_labels.csv --chaksu-feats "$F/chaksu_rfg_${pp}.npz" \
            --protocol $proto --out "$R/frozen_lr_${proto}_${pp}" > /dev/null
        run plot_curves.py "$R/frozen_lr_${proto}_${pp}" > /dev/null
    done
    run optuna_head.py --papila-labels papila_labels.csv --papila-feats "$F/papila_rfg_${pp}.npz" \
        --chaksu-labels chaksu_labels.csv --chaksu-feats "$F/chaksu_rfg_${pp}.npz" \
        --compare-to "$R/frozen_lr_tuned_${pp}" --n-trials ${N_TRIALS:-40} --out "$R/optuna_head_${pp}" > /dev/null
done
echo "done: $R"
