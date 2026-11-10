#!/bin/bash
# 10 Nov tasks on the real features: AUPRC-tuned classifiers, small MLP, loss check, learning curve, comparison.
# The test part is never used.   cd ~/glaucoma/after_10_Nov && sbatch run_10nov.slurm   (logs in logs/)
#SBATCH --job-name=nov10
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"; echo "code: $(pwd)  host: $(hostname)  $(date)"
python -c "import torch, sklearn, optuna, pandas; print('torch', torch.__version__, 'sklearn', sklearn.__version__, 'optuna', optuna.__version__, 'pandas', pandas.__version__)"
# STEPS overrides the list, separated by ";", e.g. STEPS="step5_compare.py" sbatch run_10nov.slurm
if [ -n "${STEPS:-}" ]; then IFS=';' read -ra RUN <<< "$STEPS"; else
    RUN=("step3_auprc.py logreg" "step3_auprc.py svm" "step3_auprc.py knn" "step3_auprc.py mlp_small"
         check_losses.py step4_learning_curve.py step5_compare.py); fi
for s in "${RUN[@]}"; do
    echo "===== $s  $(date +%T)"
    python $s
done
echo "done $(date)"
