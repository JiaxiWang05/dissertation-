#!/bin/bash
# One-off setup on a Hamilton LOGIN node (needs internet). Reuses the existing `glaucoma` conda env
# (python 3.11, torch 2.5.1+cu121, timm 0.9.16) and adds Optuna + scikit-learn; downloads the
# RETFound-Green weights and checks their sha256.
#   bash setup_hamilton.sh
set -euo pipefail
ROOT=/nobackup/$USER/retfound_green
mkdir -p "$ROOT/features" "$ROOT/results" "$ROOT/logs" /nobackup/$USER/weights

set +u  # conda activation scripts read unset variables
source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh
conda activate glaucoma
set -u
python -c "import optuna, sklearn" 2>/dev/null || pip install -q optuna scikit-learn
python -c "import torch, torchvision, timm, sklearn, optuna, pandas; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'timm', timm.__version__, 'sklearn', sklearn.__version__, 'optuna', optuna.__version__, 'pandas', pandas.__version__)"

W=/nobackup/$USER/weights/retfoundgreen_statedict.pth
SHA=431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1  # recorded 2026-10-17, 87,396,676 bytes
[ -s "$W" ] || curl -sSL -o "$W" https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
echo "$SHA  $W" | sha256sum -c -
