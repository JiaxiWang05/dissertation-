# RETFound-Green glaucoma pipeline

One model end to end (supervisor meeting, 16 Oct 2026): RETFound-Green, frozen backbone with a simple head
first, then Optuna over unfrozen layers and head size. Binary glaucoma vs healthy.

| Step | Script | What it does | Where it runs |
|---|---|---|---|
| 0 | `extract_features.py` | Frozen RETFound-Green → one 384-d vector per image (`.npz`) | laptop or Hamilton |
| 1 | `run_linear_probe.py` | Logistic regression on the features. `--protocol paper` = the paper's probe (sklearn defaults); `--protocol tuned` = C tuned, balanced classes | laptop or Hamilton |
| 2a | `optuna_head.py` | Optuna over the head on frozen features: logistic regression (C) or a 1–3 layer MLP (width, dropout, lr, L2, epochs) | laptop or Hamilton |
| 2b | `optuna_finetune.py` | Optuna over unfrozen ViT blocks (0, 1, 2, 4, 6, 12) + head + learning rates | Hamilton GPU |
| 3a | `disc_boxes.py` | Optic-disc centre and diameter from the experts' disc outlines (input to `--preproc disc_crop`) | Hamilton |
| 3b | `fairness.py` | Fairlearn `MetricFrame`: Chákṣu by camera, PAPILA by sex and age, with bootstrap CIs | laptop or Hamilton |
| 3c | `ablation_optuna.py` | Best tuned setting for each head depth (`head`) or number of unfrozen blocks (`finetune`) | Hamilton (`finetune` on GPU) |
| 3d | `compare_inputs.py` | Paired AUROC difference of each crop vs the squashed full image | laptop or Hamilton |
| | `plot_curves.py` | ROC and precision-recall curves (PAPILA; Chákṣu by camera) with operating points and guessing lines | laptop or Hamilton |
| | `rfg.py` | Shared code: model loading, preprocessing, data, metrics, bootstrap | |
| | `tests/make_synthetic.py` | Fake images and features to check the code end to end (numbers mean nothing) | laptop |

## Design
- **Model:** timm `vit_small_patch14_reg4_dinov2`, 392×392, mean/std 0.5, average of the patch tokens
  (official README). Weights `retfoundgreen_statedict.pth` v0.1, sha256 `431de5db…942e1`.
- **Preprocessing:** `resize` = the paper's plain resize to a square (default). `fov_square` = crop to the
  retina and pad to a square first, to check whether squashing Chákṣu's portrait Remidio photos matters.
- **PAPILA:** glaucoma (87) vs healthy (333), suspects excluded. PAPILA's own patient-level 5 folds
  (`fold_test2`), so both eyes of a patient are always in the same fold. Out-of-fold predictions.
- **No leakage:** every tuning step (C, Optuna) uses only the training patients of the current fold.
- **Chákṣu:** train on all of PAPILA, test on all 1,338 unique Chákṣu images without retraining; overall,
  per camera, per expert agreement (5–0 / 4–1 / 3–2).
- **Reported measures** (as in the Week 3 lecture): AUROC; AUPRC next to its guessing floor (the share of
  glaucoma images); sensitivity, specificity, precision (PPV) and NPV at the chosen threshold, with the
  confusion matrix; sensitivity at 90% specificity; and the PPV the same model would have in screening, where
  3.5% of people aged 40–80 have glaucoma (Tham et al. 2014).
- **Threshold:** chosen by Youden's J (or 90% sensitivity, `--threshold sens90`) on the training patients' inner cross-validated predictions, never on
  the test fold; frozen from PAPILA for Chákṣu.
- **Statistics:** 95% CIs by 2000× cluster bootstrap (patients / eyes). Paired bootstrap of the AUROC
  difference to compare two methods: if the interval contains 0, there is no clear winner. The paper's
  metric (median AUROC of 100 bootstrap resamples) is also reported.
- **Failures:** Chákṣu errors split into *dangerous* (experts unanimous, model wrong) and *safe* (experts split).
- **Clinical baseline:** age and sex only, scaled and concatenated with the image features. IOP, visual
  field and cup-to-disc ratio are part of the diagnosis and would leak the label.

## Run on the laptop
Images are expected under `~/data/dissertation` with the same layout as `/nobackup/tggq61/data`.
```bash
bash run_local.sh
```
Results go to `results/<run>/summary.md` (tables), `metrics.json` and prediction CSVs.

## Run on Hamilton
```bash
bash setup_hamilton.sh            # login node, once: Optuna + scikit-learn into the glaucoma env, weights
sbatch run_frozen.slurm           # steps 0, 1, 2a for both preprocessings (CPU)
sbatch run_optuna_finetune.slurm  # step 2b (GPU); resubmit to resume
sbatch run_followup.slurm         # steps 3b-3d after the disc boxes exist: disc crop, 90% sensitivity, fairness (CPU)
sbatch run_ablation_ft.slurm      # step 3c finetune (GPU)
```
Outputs: `/nobackup/tggq61/retfound_green/{features,results,logs}`.

## Environments
- Laptop: `~/.venvs/rfg` (Python 3.13, torch 2.14.1, timm 1.0.15, scikit-learn 1.9.1, Optuna 5.0.0).
- Hamilton: conda env `glaucoma` (Python 3.11, torch 2.5.1+cu121, timm 0.9.16) + Optuna, scikit-learn.
