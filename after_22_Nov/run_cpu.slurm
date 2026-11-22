#!/bin/bash
# 22 Nov CPU job (cache).  CMD="python cache_extra.py" sbatch run_cpu.slurm
#SBATCH --job-name=n22cpu
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
set +u; source /nobackup/$USER/envs/miniforge3/etc/profile.d/conda.sh; conda activate glaucoma; set -u
cd "$SLURM_SUBMIT_DIR"; echo "host: $(hostname)  CMD=$CMD  $(date)"
$CMD
echo "done $(date)"
