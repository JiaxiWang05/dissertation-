# After the 10 Nov meeting: AUPRC, a smaller MLP, loss functions, confusion matrices, left / right eye

Status: DONE 10 Nov 2026, 16:10 (all jobs finished; figures in `figures/`, made by `make_figures.py`).

Folder made on 10 Nov 2026 from `../after_3_Nov` (left unchanged: it holds the 8 Nov test run and its `TEST_DONE`
lock). Same data, same split file (`chaksu_split.csv`, byte-identical), same frozen RETFound-Green features
(`chaksu_rfg_resize.npz`, MD5 1189d844..., identical to `/nobackup/tggq61/retfound_green/features/`).
RETFound-Green stays frozen. **The test part (204 images) is not used anywhere in this folder**: it was scored once on
8 Nov, and scoring it again for new models would make it a second validation set.

## What Rahele and Roya asked on 10 Nov, and where it is done

| Request | What was done | File |
| --- | --- | --- |
| Find left / right eye (OD / OS) in the image headers; later test flipping | Read every EXIF tag, XMP / comment block and the text in the first 128 KB of all 1,345 files, plus file and folder names | `laterality_check.py` -> `laterality_check.csv` |
| Make the MLP smaller: 2 hidden layers, 32-128 neurons, widths may differ; report the parameter count | New Optuna search `mlp_small`: width of layer 1 and of layer 2 each from {32, 64, 96, 128} | `step3_auprc.py mlp_small` |
| Use AUPRC (or F1) instead of AUROC as the Optuna objective | All four classifiers re-tuned with the 5-fold mean validation AUPRC as the objective | `step3_auprc.py logreg / svm / knn / mlp_small` |
| Play with the loss function; give the rare class extra weight | Optuna chooses between plain cross-entropy, weighted cross-entropy and focal loss, and the size of the weight (0.5-2 x healthy / glaucoma); then a fixed comparison of 5 losses with 5 seeds x 5 folds | `step3_auprc.py`, `check_losses.py` |
| Confusion matrix of the best model | Out-of-fold confusion matrices of every classifier at two thresholds (90 % sensitivity, Youden's J), with F1 and MCC | `step5_compare.py` -> `step5_confusion.csv` |
| Learning curves that show less overfitting | 300-epoch learning curve of the best small MLP on the fixed split | `step4_learning_curve.py` |

Not done now, as agreed: nested cross-validation, augmentation, unfreezing layers, the large second dataset,
PCA / t-SNE.

## Things said in the meeting that were already in place (to tell Rahele)

- The loss is already weighted: the MLP uses `BCEWithLogitsLoss(pos_weight = healthy / glaucoma)`, about 6.1, and
  logistic regression and the SVM use `class_weight='balanced'` (so the SVM does handle the imbalance).
- Adam, AdamW, SGD and RMSprop were all in the 3-8 Nov search (trial 72 used RMSprop).
- Test set: 204 images, 28 glaucoma (not about 150 with 2-3). Each training fold has about 913 images, 128 glaucoma.
- The 8 Nov MLP (2 x 512) has 460,289 trainable parameters (384 x 512 + 512 + 512 x 512 + 512 + 512 + 1), not 3-13 M;
  still far too many for 1,141 images, which is her point.
- kNN with k = 3 or 5 was in the grid (odd k 1-99) and scored lower than k = 39.

## Parameter counts (2 hidden layers, 384 inputs, 1 output)

| Widths | Trainable parameters |
| --- | --- |
| 512, 512 (8 Nov MLP) | 460,289 |
| 128, 128 | 65,921 |
| 128, 64 | 57,601 |
| 64, 64 | 28,865 |
| 64, 32 | 26,753 |
| 32, 32 | 13,409 |

## How the new search works (step3_auprc.py)

- Same 5 folds as 3-8 Nov (`common.cv_split`: StratifiedGroupKFold, strata label x camera, seed 42). Groups: Chakshu has
  no patient or eye ID, so only the 30 Bosch file pairs (P1_Image1 / P1_Image2) and the 7 exact copies are kept together;
  every other image is its own group (1,308 groups).
- Objective: mean over the 5 folds of the validation AUPRC (average precision). Guessing gives the prevalence, 0.14.
  For the MLP each fold is scored by its validation AUPRC averaged over the kept epoch +- 5 (as on 3-8 Nov, so one lucky
  epoch cannot win). Early stopping still uses the validation loss (patience 30, up to 300 epochs).
- Trials: logistic regression 50 (C 1e-4..1e2), SVM 100 (linear or RBF, C 1e-5..1e2, gamma 1e-4..1), kNN all 200 grid
  settings, MLP 300.
- MLP search space: 2 hidden layers; width1, width2 in {32, 64, 96, 128}; dropout 0-0.7; optimiser Adam / AdamW / SGD /
  RMSprop; learning rate 1e-5..1e-1, constant or halve-on-plateau; L2 1e-6..1e-1; L1 1e-8..1e-2; batch 32-256;
  loss: plain cross-entropy, weighted cross-entropy or focal (weighted); weight scale 0.5-2 x (healthy / glaucoma);
  focal gamma 0.5-3.
- Changes to `common.py` (copied from `../after_3_Nov`): layers may have different widths; the validation AUPRC is
  recorded every epoch; loss `bce` (no weight) and `pos_weight_scale` added; `bootstrap_ci` and `paired_difference`
  work for any metric.

## Code test before the run (10 Nov, Mac, real training + validation data only, 2 trials each, on a scratch copy)

All four searches, the learning curve and the comparison ran end to end; the scratch copy was then discarded.

## Hamilton run

- Job 19425082 (`run_10nov.slurm`, 8 CPUs), 10 Nov 15:03: logistic regression, SVM and kNN searches finished
  (studies `logreg_auprc`, `svm_auprc`, `knn_auprc` in `optuna.db`). The small-MLP search then took about 85 s per trial,
  so 300 trials in one process would have needed about 7 hours; the job was cancelled at 15:22 after 2 MLP trials
  (a leftover study `mlp_small` with 2 trials remains in `optuna.db` and is not used).
- Job array 19425176 (`run_mlp_workers.slurm`, 8 workers x 2 CPUs), 15:25: the small-MLP search with 8 workers sharing
  one study in an Optuna journal file (`optuna_mlp_small.log`, `common.mlp_small_storage`); each worker has its own
  sampler seed (42 + worker number) and all stop when 300 trials are complete.
- Job 19425177 (`run_10nov.slurm` with `STEPS="check_losses.py;step4_learning_curve.py;step5_compare.py"`), started
  after the array: 27 min, finished 16:10. The 8 workers took 17 min each and finished 307 trials (a few workers
  started a trial just before the 300th one completed; all 307 count).
- `make_figures.py` draws the figures in `figures/` from the saved results.

## Results

## Result 1: left / right eye and fellow eyes (10 Nov, done)

**Headers (laterality_check.py, run on Hamilton on all 1,345 files): no laterality information.**
- EXIF tags present: Orientation, ColorSpace, image size, resolution, ExifVersion etc. (1,204-1,218 files); only the
  144 Bosch files also carry Make / Model (Optomed SmartScope M5), Software, DateTime, Copyright and exposure tags.
  No tag, XMP or comment block names the eye. The file and folder names have no OD / OS / L / R either.
- The script also flagged "OS" / "OU" in the header bytes of 665 files, but these are random letters inside the
  compressed image data (for example `OS@!x#`, `^D=OS`), not text fields. They carry no meaning.
- The dataset's own tables (`6.0_Glaucoma_Decision/...csv`: disc / cup areas and ratios per image) have no eye column,
  and the dataset README (`/nobackup/tggq61/data/raw/Chaksu_Readme.pdf`) uses "OD" only for optic disc; it says the
  personal information was anonymised and the images are mostly centred on the optic disc.

**Optic-disc position (laterality_pairs.py): left / right can be read for Remidio, not for Bosch or Forus.**
- Horizontal position of the expert disc mask centre (disc x / image width, 10 bins from 0 to 1):
  Remidio `[0, 6, 180, 291, 76, 44, 151, 272, 53, 1]` = two clear groups; Bosch `[0, 0, 8, 15, 42, 61, 16, 3, 0, 0]`
  and Forus `[0, 0, 6, 36, 25, 22, 26, 11, 0, 0]` = one group in the middle (photographs centred on the disc).
- Remidio: 518 images with the disc in the left part (x < 0.45), 499 in the right part (x > 0.55), 57 near the centre.
  Under the usual convention the disc is on the right of the image in a right eye (OD), but this is not confirmed for
  this camera, so the groups are named "disc left" / "disc right" (`remidio_disc_side.csv`).

**Fellow eyes: consecutive Remidio numbers are almost always the two eyes of one patient.**
- Of 807 pairs of Remidio files with consecutive numbers (IMG_n, IMG_n+1) and a clear disc side, 93 % show opposite
  sides; for two random Remidio images it would be 50 %. The two images of such a pair also share the glaucoma label
  more often than chance (85 % against 76 %).
- Taking disjoint consecutive pairs with opposite sides as pseudo patients gives 433 pairs (27 where both eyes have
  glaucoma). 216 of them are split between training / validation / test; **121 of the 204 test images have their likely
  fellow eye in training or validation.** The groups go from 1,308 (image groups) to 875 (pseudo patients);
  `pseudo_patients.csv`.
- **Effect on the results: small.** Logistic regression (C = 0.00725) with 5-fold cross-validation repeated over 10 fold
  seeds: folds grouped by image group, out-of-fold AUROC 0.814 (0.805-0.821), AUPRC 0.413 (0.396-0.430); folds grouped
  by pseudo patient, AUROC 0.810 (0.801-0.823), AUPRC 0.412 (0.392-0.444). The difference is inside the spread between
  fold seeds, so having fellow eyes on both sides of the split inflates the result by at most about 0.005 AUROC.
- So: the leakage exists, but the frozen-feature results hardly depend on it. For a final, cleaner analysis the split
  could be redone by pseudo patient (Rahele's decision: the test set has already been used once).
- Flipping all images to one side (Rahele's later experiment) is possible for Remidio from `remidio_disc_side.csv`;
  Bosch and Forus would need another way to tell the eyes apart (for example the direction of the main vessels).

## Result 2: classifiers tuned by AUPRC (step3_auprc.py; 5-fold mean validation AUPRC, guessing 0.14)

| Classifier | Trials | Best setting | 5-fold mean AUPRC | Folds | 5-fold mean AUROC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 50 | C = 0.00773 | 0.437 | 0.441 / 0.437 / 0.389 / 0.404 / 0.513 | 0.814 |
| SVM | 100 | RBF, C = 6.06, gamma = 0.00057 | 0.448 | 0.477 / 0.410 / 0.391 / 0.423 / 0.537 | 0.818 |
| kNN | 200 (grid) | k = 39, distance weights, cosine | 0.353 | 0.330 / 0.308 / 0.306 / 0.434 / 0.389 | 0.749 |
| Small MLP | 307 | 32 and 64 neurons, AdamW, lr 0.000136 (plateau), dropout 0.07, L2 0.0026, L1 0.0070, batch 32, weighted cross-entropy x 1.83 | 0.445 | 0.437 / 0.439 / 0.412 / 0.390 / 0.546 | 0.814 |

- Changing the objective from AUROC to AUPRC hardly changes the logistic regression (C 0.00725 -> 0.00773), the SVM
  (still RBF) or kNN (still k = 39, distance, cosine).
- **The small MLP has 14,497 parameters instead of 460,289 (32 x smaller)** and is as good as the big one: kept epochs
  104 / 94 / 83 / 52 / 111 per fold (the 8 Nov MLP stopped at 45-66 with a 9 x smaller learning rate; the 3 Nov one
  at about 14).
- Width (`figures/fig5_small_mlp_search.png`): the best trials used 32-64 or 128-64; 96-32 and 128-128 were worst.
- Loss in the search: Optuna chose weighted cross-entropy in 296 of 307 trials; plain cross-entropy (6 trials) and
  focal loss (5 trials) scored lower.

## Result 3: learning curve of the small MLP (step4_learning_curve.py; `figures/fig2_small_mlp_learning_curve.png`)

- Fixed split, all 300 epochs: the validation loss is lowest at epoch 92 (1.222) and is still 1.236 at epoch 300;
  the training loss levels off at 0.82 instead of falling to 0. Validation AUROC 0.818, AUPRC 0.349 at epoch 92, flat
  afterwards.
- So the overfitting is gone: compare the 8 Nov MLP (2 x 512), whose validation loss rose from 0.89 at epoch 61 to
  2.3 at epoch 300 while its training loss fell to 0.01. This is the "more confident" curve Rahele asked for. The AUROC
  is the same, as expected from the 3-8 Nov work.

## Result 4: loss functions (check_losses.py; best small MLP, 5 seeds x 5 folds; `figures/fig6_loss_functions.png`)

| Loss | AUPRC (mean of 5 seeds; range) | AUROC (mean; range) | Kept epoch (median) | Loss gap |
| --- | --- | --- | --- | --- |
| Weighted cross-entropy, weight 12.3 (2 x healthy / glaucoma) | 0.426 (0.407-0.434) | 0.803 (0.792-0.809) | 61 | 0.30 |
| As found by Optuna (weighted, x 1.83) | 0.426 (0.408-0.438) | 0.803 (0.792-0.810) | 59 | 0.28 |
| Weighted cross-entropy, weight 6.1 | 0.406 (0.377-0.421) | 0.794 (0.785-0.798) | 79 | 0.20 |
| Plain cross-entropy, no weight | 0.318 (0.143-0.404) | 0.734 (0.509-0.803) | 285 | 0.02 |
| Focal loss (gamma 2), weight 6.1 | 0.308 (0.268-0.344) | 0.717 (0.650-0.751) | 9 | 0.00 |

- **The class weight matters**: without it the AUPRC drops by about 0.11 and becomes unstable between seeds (one seed
  is at the guessing level, 0.14); a weight of about 2 x (healthy / glaucoma) is a little better than 1 x (+0.02).
- Focal loss is clearly worse here (it stops after about 9 epochs).
- With 5 seeds the kept model scores a bit lower (0.426) than the Optuna score (0.445, averaged over the kept epoch
  +- 5 at one seed): the search's best is optimistic, as on 3-8 Nov.

## Result 5: comparison and confusion matrices (step5_compare.py; out-of-fold, 1,141 images, 160 glaucoma)

| Classifier | AUPRC (95 % CI) | AUROC (95 % CI) | Best minus this, AUPRC (95 % CI) | Tied with best |
| --- | --- | --- | --- | --- |
| Logistic regression, 8 Nov (AUROC-tuned, C 0.00725) | 0.414 (0.344-0.505) | 0.814 (0.779-0.848) | 0.013 (-0.025 to 0.049) | yes |
| Logistic regression, AUPRC-tuned (C 0.00773) | 0.415 (0.345-0.507) | 0.814 (0.779-0.848) | 0.011 (-0.026 to 0.048) | yes |
| SVM, AUPRC-tuned | 0.427 (0.355-0.514) | 0.817 (0.785-0.850) | best | - |
| kNN | 0.337 (0.270-0.413) | 0.747 (0.708-0.785) | 0.090 (0.025 to 0.161) | no |
| Small MLP (14,497 parameters) | 0.414 (0.345-0.506) | 0.803 (0.764-0.841) | 0.013 (-0.030 to 0.049) | yes |

Confusion matrices (out-of-fold; `figures/fig4_confusion_sens90.png`, `fig4_confusion_youden.png`):

| Classifier | Threshold | Glaucoma found | Missed | Healthy flagged | Healthy cleared | Sensitivity | Specificity | PPV | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression (8 Nov) | 90 % sensitivity | 144 | 16 | 473 | 508 | 0.900 | 0.518 | 0.233 | 0.371 | 0.291 |
| Logistic regression (8 Nov) | Youden's J | 118 | 42 | 234 | 747 | 0.738 | 0.761 | 0.335 | 0.461 | 0.375 |
| SVM | 90 % sensitivity | 144 | 16 | 483 | 498 | 0.900 | 0.508 | 0.230 | 0.366 | 0.284 |
| SVM | Youden's J | 120 | 40 | 239 | 742 | 0.750 | 0.756 | 0.334 | 0.462 | 0.379 |
| kNN | 90 % sensitivity | 144 | 16 | 535 | 446 | 0.900 | 0.455 | 0.212 | 0.343 | 0.251 |
| Small MLP | 90 % sensitivity | 144 | 16 | 441 | 540 | 0.900 | 0.550 | 0.246 | 0.387 | 0.313 |
| Small MLP | Youden's J | 128 | 32 | 296 | 685 | 0.800 | 0.698 | 0.302 | 0.438 | 0.358 |

(The AUPRC-tuned logistic regression gives the same matrices as the 8 Nov one.)

- By the rule fixed in advance (tied if the CI of the AUPRC difference includes 0; simplest tied wins) the choice is
  still **logistic regression**: SVM, logistic regression and the small MLP are tied; kNN is clearly worse.
- At the screening threshold (90 % sensitivity) **the small MLP flags the fewest healthy images: 441 against 473
  (logistic regression) and 483 (SVM)**, i.e. specificity 0.55 against 0.52; F1 0.39 against 0.37. This is the kind of
  difference Rahele said matters (fewer unnecessary referrals at the same sensitivity), but it is small and has no CI
  yet, so it is a candidate, not a winner.
- None of the models calls everything healthy: at the 90 % threshold all find 144 of the 160 glaucoma images.

## Summary for the next meeting

1. Left / right: not in the headers; readable from the disc position on Remidio only. Consecutive Remidio numbers are
   almost always the two eyes of one patient (93 %); keeping those pairs together changes the cross-validated AUROC
   by about 0.004 only.
2. Smaller MLP: 32 and 64 neurons, 14,497 parameters (was 460,289); the learning curve no longer overfits (validation
   loss flat from epoch 92 to 300), with the same AUROC.
3. AUPRC as objective: same choices for logistic regression, SVM and kNN; AUPRC about 0.41-0.43 for the three tied
   models (guessing 0.14).
4. Loss: the class weight is essential (no weight: AUPRC 0.32 and unstable); focal loss is worse; weight about 2 x
   (healthy / glaucoma) is best.
5. Confusion matrices: at 90 % sensitivity the small MLP clears the most healthy images (540 of 981).
6. Still frozen; the test set was not used again.
