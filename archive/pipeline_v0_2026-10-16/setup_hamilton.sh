#!/bin/bash
# One-off setup on a Hamilton LOGIN node (needs internet): Python env + RETFound-Green weights.
#   bash setup_hamilton.sh
set -euo pipefail
ROOT=/nobackup/$USER/retfound_green
ENV=$ROOT/env
mkdir -p "$ROOT/weights" "$ROOT/features" "$ROOT/results" "$ROOT/logs"

module purge
module load python 2>/dev/null || true
python3 --version

if [ ! -x "$ENV/bin/python" ]; then
    python3 -m venv "$ENV"
fi
"$ENV/bin/pip" install -q --upgrade pip
# CPU wheels are enough: ~1.8k images through a ViT-S takes minutes.
"$ENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$ENV/bin/pip" install -q "timm==0.9.12" pandas scikit-learn
"$ENV/bin/python" -c "import torch, timm, sklearn; print('torch', torch.__version__, 'timm', timm.__version__, 'sklearn', sklearn.__version__)"

W=$ROOT/weights/retfoundgreen_statedict.pth
if [ ! -s "$W" ]; then
    curl -L -o "$W" https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
fi
ls -lh "$W"
