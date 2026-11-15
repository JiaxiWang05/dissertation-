# RETFound-Green: frozen features + logistic regression

First end-to-end pipeline, from the supervisor meeting:
- one model (RETFound-Green), fully frozen, with a simple head
- performance reported for each dataset

| Step | What it does |
|---|---|
| `extract_features.py` | RETFound-Green (`vit_small_patch14_reg4_dinov2`, 392 px, mean/std 0.5, avg pool) → one 384-d vector per image, saved as `.npz` |
| `run_linear_probe.py` | PAPILA binary glaucoma vs healthy (suspects excluded): 5-fold **patient-grouped** CV, with C tuned by grouped inner CV. Then trains on all of PAPILA and tests on **Chákṣu without retraining**, overall and per camera. Reports 95% CIs (1000× cluster bootstrap), image vs clinical (age, sex) vs image+clinical, and a safe/dangerous split of Chákṣu errors |
| `setup_hamilton.sh` | Run once on a login node: creates the venv in `/nobackup/$USER/retfound_green/env` and downloads the weights |
| `run_pipeline.slurm` | Runs both extractions and then the probe, on the `shared` CPU partition |

## Run on Hamilton
```bash
rsync -av ~/Desktop/dissertation/pipeline/ tggq61@hamilton8.dur.ac.uk:/nobackup/tggq61/retfound_green/pipeline/
ssh tggq61@hamilton8.dur.ac.uk
cd /nobackup/tggq61/retfound_green/pipeline && bash setup_hamilton.sh && sbatch run_pipeline.slurm
```
Results are written to `/nobackup/tggq61/retfound_green/results/rfg_frozen_lr/`: `summary.md`, `metrics.json` and the prediction CSVs.

## Choices to revisit
- Images are resized directly to 392×392, as in the official RETFound-Green example. There is no fundus crop. Chákṣu Remidio images are portrait (2448×3264), so they get squashed.
- Chákṣu label = majority of 5 experts (`label_majority`). The 7 MD5 duplicates are dropped. Paired Bosch images (`bosch_group`) are resampled together in the bootstrap.
- The clinical baseline uses age and sex only. IOP, visual-field MD and similar measures are part of the diagnosis and would leak the label.
- Next step (meeting): Optuna over unfrozen layers and head size/depth.
