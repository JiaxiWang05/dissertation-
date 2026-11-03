#!/bin/bash
# Chaksu frozen-feature guide on the real features: steps 2-5 (split check, Optuna x4, MLP learning curve, comparison).
# Step 6 (the one test run) is NOT in this job.   cd ~/glaucoma/3_Nov && sbatch run_learn.slurm
#SBATCH --job-name=learn_chaksu
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"; echo "code: $(pwd)  host: $(hostname)  $(date)"
python -c "import torch, sklearn, optuna, pandas; print('torch', torch.__version__, 'sklearn', sklearn.__version__, 'optuna', optuna.__version__, 'pandas', pandas.__version__)"
for s in ${STEPS:-step2_check_split step3_logreg step3_svm step3_mlp step3_knn step4_learning_curve step5_compare}; do
    echo "===== $s  $(date +%T)"
    python "$s.py"
done
echo "done $(date)"
