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

### Result 5b: likely fellow eyes kept in the same fold (`FOLDS=pseudo`, job 19550228, `final_chaksu_pseudo.csv`)

Folds rebuilt with StratifiedGroupKFold on the pseudo-patients of `../after_10_Nov/pseudo_patients.csv` (433 likely
fellow-eye pairs, plus the Bosch pairs and copies); the frozen models are refitted on the same new folds.

| Model | Fair AUPRC, 3 seeds (range) | AUROC (seed 0) | Healthy flagged at 90 % sens. | Minus frozen SVM (95 % CI) | Same model, ordinary folds |
| --- | --- | --- | --- | --- | --- |
| Frozen SVM | 0.421 | 0.809 | - | reference | 0.427 |
| Frozen logistic regression | 0.418 | 0.811 | - | - | 0.415 |
| CNN from scratch | 0.322 (0.299-0.363) | 0.693 | 82 % | -0.122 (-0.207 to -0.035), worse | 0.371 / 0.408 |
| RETFound-Green, last 2 blocks open | 0.508 (0.453-0.558) | 0.881 | 29 % | +0.137 (0.060 to 0.210), better | 0.521 / 0.522 |

- The frozen models do not change (0.421 against 0.427): as found on 10 Nov, fellow eyes hardly matter for them.
- The fine-tuned model keeps its advantage (0.51 against 0.52 on the ordinary folds; still clearly above the frozen
  features), so its gain is not explained by fellow eyes on both sides of the split.
- The CNN drops further (0.32) and is now clearly worse than the frozen features.

## Result 6 (search stage): LoRA on all 12 RETFound-Green blocks (`lora_cv.py`, array 19550229, 45 trials)

Every pre-trained weight frozen; low-rank add-ons on the attention layers (qkv, proj) or also the MLP layers of all
12 blocks; linear head. 6 workers on 18 GB H200 slices, 75 min.

- Best search AUPRC 0.653 (folds 0.767 / 0.674 / 0.625 / 0.487 / 0.712), mean AUROC 0.911: rank 4, attention only,
  alpha / r = 1, LoRA learning rate 0.00135, head 0.0010, class weight 0.28 x 6.13 = 1.7; 110,977 trainable weights
  (3.5 M for the 2 open blocks).
- The top 8 trials all use attention-only add-ons with rank 4 or 8 and score 0.63-0.65 (2 open blocks: best 0.59).
- Fair comparison (3 seeds; jobs 19551235, 19551236, `MODELS=lora`; `final_chaksu_lora.csv`, `final_chaksu_pseudo_lora.csv`):

| Folds | LoRA fair AUPRC, 3 seeds (range) | AUROC (seed 0) | Healthy flagged at 90 % sens. | Minus frozen SVM (95 % CI) |
| --- | --- | --- | --- | --- |
| Ordinary | 0.544 (0.528-0.554) | 0.887 (0.861-0.910) | 27 % | +0.101 (0.026 to 0.177), better |
| Likely fellow eyes together | 0.534 (0.471-0.610) | 0.812 (0.771-0.850) | 56 % (seed 0) | +0.049 (-0.033 to 0.135), tied |

- LoRA against the 2 open blocks (seed 0 of each, paired, ordinary folds): AUPRC +0.024 (-0.032 to 0.085), tied;
  AUROC +0.029 (0.006 to 0.052), better. LoRA trains 111k weights against 3.55 M.
- On the fellow-eye folds the seed-0 LoRA run is the weakest of its three seeds (0.471 against 0.523 / 0.610), so its
  paired test is a tie although the 3-seed mean stays above the frozen features; the 2 open blocks remain clearly
  better there (+0.137). Both adaptations beat the frozen features on the ordinary folds.
