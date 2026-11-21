# After the 17 Nov meeting: CNN baseline, class weight, percentage confusion matrices, unfreezing

Status: IN PROGRESS (started 20 Nov; updated after every step). Test sets are not used anywhere in this folder.

## Tasks from the 17 Nov meeting and where they are done

| Task | Script | State |
| --- | --- | --- |
| Confusion matrices as percentages (Chakshu and FairVLMed); Chakshu split by camera | `confusion_percent.py` | done |
| Class weight: try smaller weights; let Optuna choose it with F1 or AUPRC | `weight_search.py grid` (job 19545173), `weight_search.py optuna` (array 19545174, F1 objective) | done |
| Own small CNN from scratch as the baseline (Optuna: blocks, channels, kernel, ...), both datasets | `cache_images.py`, `img_common.py`, `cnn_cv.py`, `final_compare.py` | done (Result 5) |
| Repeat on FairVLMed (no class weight: 50 % glaucoma) | `confusion_percent.py`, `cnn_cv.py`, `final_compare.py fairvlmed` | done |
| If time: open 0-3 RETFound-Green blocks (Optuna) | `unfreeze_cv.py`, `final_compare.py chaksu` | done (Result 5) |

## Result 1: confusion matrices in percent (`confusion_percent.csv`, `figures/fig1_confusion_percent_*.png`)

Rows normalised by the true class: of all glaucoma images, % found / missed; of all healthy images, % flagged / cleared.
Out-of-fold scores (Chakshu 1,141 images, FairVLMed 8,000 patients).

| Dataset | Classifier | 90 % sensitivity: healthy flagged | Youden's J: glaucoma found | Youden's J: healthy flagged |
| --- | --- | --- | --- | --- |
| Chakshu | Logistic regression | 48 % | 74 % | 24 % |
| Chakshu | SVM | 49 % | 75 % | 24 % |
| Chakshu | kNN | 55 % | 92 % | 55 % |
| Chakshu | Small MLP | 45 % | 80 % | 30 % |
| FairVLMed | Logistic regression | 61 % | 64 % | 20 % |
| FairVLMed | SVM | 59 % | 72 % | 28 % |
| FairVLMed | kNN | 71 % | 67 % | 33 % |
| FairVLMed | Small MLP | 60 % | 65 % | 22 % |

At the 90 % sensitivity threshold every model finds 90 % of the glaucoma images by construction; the difference is in
the healthy row: about half of the healthy images are flagged on Chakshu and about 60 % on FairVLMed.

## Result 2: Chakshu by camera (`confusion_by_camera.csv`, `figures/fig2_confusion_by_camera.png`)

Logistic regression (the chosen model), each classifier's own out-of-fold thresholds:

| Camera | Images (glaucoma) | 90 % sensitivity: glaucoma found | healthy flagged | Youden's J: glaucoma found | healthy flagged |
| --- | --- | --- | --- | --- | --- |
| Remidio | 910 (127) | 91 % | 47 % | 76 % | 23 % |
| Bosch | 125 (18) | 89 % | 55 % | 72 % | 23 % |
| Forus | 106 (15) | 80 % | 48 % | 60 % | 30 % |

- Rahele asked whether the healthy images called glaucoma come mainly from one camera: **no**. The share of healthy
  images flagged is similar on all three cameras (47-55 %), so the low specificity is not a Remidio effect.
- Forus misses the most glaucoma (80 % found at the 90 % threshold, 60 % at Youden), but it has only 15 glaucoma images.
- The same holds for the SVM and the small MLP (kNN flags 81 % of healthy Bosch images).

## Result 3: class weight (`weight_grid.csv`; `weight_search.py`, jobs 19545173 and 19545174)

Weight w on the glaucoma class (6.13 = healthy / glaucoma). Out-of-fold, 1,141 images; MLP = mean of 5 seeds.

| Model | Weight | AUPRC | AUROC | Sensitivity at 0.5 | Specificity at 0.5 | F1 at 0.5 | Best F1 (any threshold) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.25 | 0.384 | 0.806 | 0.000 | 1.000 | 0.000 | 0.460 |
| Logistic regression | 1 | 0.405 | 0.813 | 0.112 | 0.983 | 0.185 | 0.470 |
| Logistic regression | 2 | 0.410 | 0.814 | 0.375 | 0.936 | 0.424 | 0.477 |
| Logistic regression | 4 | 0.414 | 0.814 | 0.588 | 0.847 | 0.465 | 0.474 |
| Logistic regression | 6.13 | 0.414 | 0.813 | 0.688 | 0.788 | 0.460 | 0.474 |
| Logistic regression | 12.27 | 0.412 | 0.811 | 0.812 | 0.667 | 0.421 | 0.479 |
| Small MLP | 1 | 0.304 | 0.731 | 0.000 | 1.000 | 0.000 | 0.405 |
| Small MLP | 4 | 0.325 | 0.756 | 0.454 | 0.847 | 0.332 | 0.428 |
| Small MLP | 6.13 | 0.386 | 0.789 | 0.702 | 0.735 | 0.424 | 0.450 |
| Small MLP | 12.27 | 0.406 | 0.804 | 0.842 | 0.591 | 0.387 | 0.467 |

(all nine weights in `weight_grid.csv`)

- **The weight moves the trade-off, not the ranking.** For logistic regression AUPRC, AUROC and the best possible F1
  hardly change between weight 2 and 12; what changes is where the default 0.5 threshold falls: a lower weight gives
  fewer false alarms (specificity 0.85 at w = 4 against 0.79 at 6.13) and fewer glaucoma found (0.59 against 0.69).
  F1 at 0.5 is highest at w about 4 (0.465; 6.13 gives 0.460).
- For the small MLP, weights below 4 make it call almost everything healthy (early stopping keeps an early epoch);
  it needs the weight.
- Optuna with the F1 objective (200 trials, 8 workers, 6 min): best F1 at 0.5 = 0.507 with weight 2.8 (scale 0.46),
  64-64 neurons (search best on the same folds, so optimistic).
- Because the evaluation thresholds (90 % sensitivity, Youden's J) are set on the scores anyway, the weight mainly
  matters for a model used at its default threshold.

## Result 4 (search stage): CNN from scratch and unfrozen RETFound-Green (5-fold mean validation AUPRC)

| Search | Trials | Best setting | Search AUPRC (folds) | Mean AUROC | Parameters (trainable) |
| --- | --- | --- | --- | --- | --- |
| CNN, Chakshu | 61 | 5 blocks, 32 channels doubling, kernel 5, dropout 0.02, weight x 0.36 | 0.544 (0.513 / 0.530 / 0.519 / 0.479 / 0.678) | 0.857 | 4,356,897 |
| CNN, FairVLMed | 31 | 5 blocks, 32 channels, kernel 5, dropout 0.23 | 0.818 (0.805 / 0.803 / 0.846 / 0.832 / 0.804) | 0.808 | 4,355,297 |
| RETFound-Green, last 2 blocks open, Chakshu | 26 | linear head, dropout 0.37, weight x 0.30 | 0.592 (0.734 / 0.593 / 0.473 / 0.447 / 0.712) | 0.884 | 3,551,617 |

- Frozen-feature reference: Chakshu SVM 0.427 / 0.817; FairVLMed small MLP 0.805 / 0.791.
- Unfreezing search by depth: 2 blocks in 15 trials (best 0.592), 1 block in 9 (best 0.583), 3 blocks once (0.550),
  0 blocks (frozen backbone, head trained on images) once: 0.382, AUROC 0.802 - the same level as the frozen-feature
  classifiers, a useful sanity check.
- CNN Chakshu: even the median trial (0.467) is above the frozen models.
- **These are search scores and flatter the image models**: the best of many trials, scored on the folds that
  chose them, with early stopping on the scored fold. `final_compare.py` therefore retrains each best setting with
  3 seeds, each fold for the median kept epoch of the other folds (the scored images choose nothing), and compares
  with the frozen models on the same images (jobs 19550043 Chakshu, 19550044 FairVLMed).
- A second check (`FOLDS=pseudo`; first attempt 19550045 used the name `GROUPS`, a reserved bash variable, so it silently repeated the plain run - kept as `final_chaksu_run2.csv`; redone as job 19550114): the same comparison with the likely fellow eyes kept in the same
  fold, because flexible image models could exploit a patient's other eye in training (frozen models did not: 0.004).

## Jobs (20 Nov)

- 19545173 (CPU): class-weight grid. 19545174 (8 CPU workers): small-MLP search with the F1 objective, 200 trials.
- 19545209 (CPU): image cache (Chakshu 256 and 392 px, FairVLMed 256 px, in `/nobackup/tggq61/retfound_green/cache`).
- 19545210 / 19545211 (2 GPU workers each): CNN searches, 60 trials (Chakshu) and 30 (FairVLMed).
- 19545212 (3 GPU workers): unfreezing search, 24 trials.

## Result 5: fair comparison (`final_compare.py`, `final_<tag>.csv`, `figures/fig3_final_comparison.png`)

Each best setting retrained in every fold with 3 seeds; each fold trains for the median kept epoch of the other four
folds and is scored at its last epoch (the scored images choose nothing). Reference: the best frozen-feature model
on the same images and folds. Difference and CI: seed 0, paired, 2,000 resamples of the image groups.

| Data | Model | Trainable parameters | Search AUPRC | Fair AUPRC, 3 seeds (range) | AUROC (seed 0) | Healthy flagged at 90 % sens. | Minus frozen AUPRC (95 % CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chakshu | Frozen features, SVM | - | 0.427 | 0.427 | 0.817 | 49 % | reference |
| Chakshu | CNN from scratch | 4,356,897 | 0.544 | 0.371 (0.306-0.440) | 0.828 | 43 % | +0.013 (-0.076 to 0.106), tied |
| Chakshu | RETFound-Green, last 2 blocks open | 3,551,617 | 0.592 | 0.521 (0.505-0.552) | 0.856 | 38 % | +0.078 (0.003 to 0.152), better |
| FairVLMed | Frozen features, small MLP | - | 0.808 | 0.805 | 0.791 | 60 % | reference |
| FairVLMed | CNN from scratch | 4,355,297 | 0.818 | 0.734 (0.669-0.783) | 0.749 | 64 % | -0.053 (-0.065 to -0.041), worse |

- Repeat run on Chakshu (job 19550045, same settings, `final_chaksu_run2.csv`): CNN 0.408 (0.383-0.423), fine-tuned
  0.522 (0.504-0.553). GPU training is not bit-reproducible; the fine-tuned model is stable, the CNN is not.
- The search scores flatter the image models: the CNN drops by 0.13-0.17 (Chakshu) and 0.08 (FairVLMed) from search to
  fair score; the fine-tuned model by 0.07.
- **Fine-tuning the last two RETFound-Green blocks is the only change so far that beats the frozen features** (Chakshu).
- A CNN from scratch does not reach the frozen features on either dataset, even with 8,000 FairVLMed patients.
