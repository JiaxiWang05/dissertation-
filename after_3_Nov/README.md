# After the 3 Nov meeting: new split, MLP overfitting

Same pipeline as `../3_Nov` (kept unchanged: that is what was shown to Rahele on 3 Nov), with her changes:

| Change | Where |
|---|---|
| The 7 exact copies are kept (1345 images, 1308 eyes); a copy always sits in the same part as its original | `common.py` (`load_labels`) |
| New split 70 / 15 / 15 (939 / 202 / 204), grouped by eye, stratified by label and camera, seed 42 | `step1_make_split.py` -> `chaksu_split.csv` |
| Check for Rahele: Optuna with 5-fold CV inside returns ONE set (best 5-fold mean); demo of the last-fold bug | `check_optuna_cv.py` (in memory, writes nothing) |
| MLP search widened against overfitting: optimiser, lr, L2, L1, dropout, batch size, loss (weighted BCE or focal), 1-5 layers; 300 trials; patience 30, up to 300 epochs | `step3_mlp.py`, `common.py` (`train_mlp`) |
| MLP trials scored by 5-fold CV on training + validation (folds grouped by eye, stratified by label and camera; scaler per fold); each fold scored by its validation AUROC averaged over the kept epoch +- 5; lr schedule searched (constant, or halve after 11 epochs without a lower validation loss); L1 up to 1e-2. The earlier search is kept as study `mlp_fixed_val` in `optuna.db` | `step3_mlp.py`, `common.py` (`cv_folds`, `train_mlp`) |
| Seed check: 4 MLP settings x 5 new seeds x the same 5 folds; AUROC of the kept model, kept epoch, validation minus training loss, AUROC jump per epoch | `check_mlp_seeds.py` -> `check_mlp_seeds.csv` |
| Check: the 3 Nov setting at its own lr and at lr / 10, same 5 folds | `check_lr_epoch.py` -> `check_lr_epoch.csv`, `check_lr_epoch_curves.csv` |
| Gaussian noise on the training features as a regulariser (training batches only; sigma 0 / 0.1 / 0.2 / 0.3 / 0.5 / 1 / 2): 3 settings x 7 levels x 5 seeds x the same 5 folds, then 300-epoch learning curves on the fixed split | `common.py` (`train_mlp`, `noise`), `check_noise.py` -> `check_noise.csv`, `check_noise_curves.csv` |
| Figures for the overfitting write-up (Notion page, private) | `plot_overfitting.py` -> `figures/overfitting_*.png` |
| Logistic regression, SVM and kNN also tuned by 5-fold CV (the same folds as the MLP). Step 5 refits each model's best setting in every fold, compares the 1,141 out-of-fold scores (AUROC, eye-grouped bootstrap CI, paired differences, tie rule), sets the thresholds from them and refits the chosen model on all training + validation images (an MLP would train for the median kept epoch of the 5 folds) | `step3_logreg.py`, `step3_svm.py`, `step3_knn.py`, `step5_compare.py` -> `step5_comparison.csv`, `step5_oof_scores.csv`, `chosen.json`, `chosen_model.joblib`; `common.py` (`cv_split`, `train_mlp` without a validation part) |

The test part has not been used. Run on Hamilton: `cd ~/glaucoma/after_3_Nov && sbatch run_learn.slurm` (logs in `logs/`).

## Search 1: scored on the one validation part (Hamilton job 19338403, 300 trials, 20 min; study `mlp_fixed_val`)

- Best trial 174: 3 x 64, RMSprop lr 0.011, dropout 0.64, L2 0.002, L1 0.00085, weighted BCE; validation AUROC 0.869 at epoch 76.
- Its learning curve (since replaced by search 2's): training and validation loss no longer split apart, but the curve is very noisy
  (validation AUROC jumps 0.70-0.87 between epochs, down to 0.19 at worst).
- Retrained with 5 seeds on the same split (Mac): trial 174 gives 0.828 on average (0.816-0.839), the old 3 Nov
  setting 0.836 (0.829-0.842, best epoch 15), trial 72 0.845 (0.838-0.859, best epoch about 33).
  So the 0.869 was a lucky epoch, not a better model.

## Search 2: scored by 5-fold CV (Hamilton job 19340041, 300 trials, 2 h 12 min; study `mlp`)

- Best trial 112: 2 x 512, Adam, lr 1.6e-5, constant lr, dropout 0.15, L2 4.9e-5, L1 off, weighted BCE.
  Score 0.815 (validation AUROC averaged over the kept epoch +- 5; folds 0.823 / 0.802 / 0.810 / 0.775 / 0.868),
  kept epochs 58 / 59 / 63 / 45 / 66.
- The score surface is flat: median trial 0.8126, best 0.8155; the top 20 trials sit at the edge of the range
  (lr 1.0-1.7e-5 at the lower bound 1e-5, width 512 at the upper bound). 281 of 300 trials used Adam.
- `mlp_learning_curve.png` (trial 112, one split, 300 epochs): smooth, but ordinary overfitting stretched in time.
  Lowest validation loss at epoch 61 (0.89), then it rises to 2.3 while training loss falls to 0.01;
  validation AUROC stays 0.81-0.84.
- Seed check (`check_mlp_seeds.py`, job 19340041; 5 seeds x 5 folds; AUROC of the kept model):

  | Setting | AUROC (mean of 5 seeds) | Seed range | Kept epoch (median) | Val minus train loss | AUROC jump per epoch |
  |---|---|---|---|---|---|
  | 3 Nov setting (Adam, 2 x 128) | 0.805 | 0.801-0.817 | 12 | 0.26 | 0.006 |
  | Search 1 best (trial 174) | 0.796 | 0.786-0.805 | 48 | 0.05 (underfits) | 0.038 |
  | Search 1 trial 72 (RMSprop, dropout 0.65) | 0.816 | 0.813-0.818 | 18 | 0.19 | 0.016 |
  | Search 2 best (trial 112) | 0.808 | 0.805-0.813 | 58 | 0.29 | 0.003 |

  Trial 72 beats the 3 Nov setting in 5 / 5 folds (+0.011); trial 112 in 3 / 5 (+0.003).
- `check_lr_epoch.py` (job 19341162): the 3 Nov setting with lr / 10 is kept at epoch 125 instead of 13, with the same
  AUROC (0.8165 both), the same loss gap (0.30 vs 0.28) and the same lowest validation loss (0.908 vs 0.905).
  So a later kept epoch only shows a smaller learning rate, not less overfitting.
- Conclusions: (1) every MLP sits at 0.80-0.82 in 5-fold CV, and logistic regression on the same folds gives
  0.806-0.814 for C 0.0011-0.05 (Mac check, not yet a step script); the frozen features and 160 glaucoma images set
  the limit, not overfitting. (2) The +- 5 epoch window penalises jumpy curves (trial 72: 0.797 window vs 0.811 kept
  epoch at seed 42), so search 2 drifted to the smallest lr and never sampled trial 72's region (1 RMSprop trial).
  (3) The ranking flips with the seed (seed 42: trial 112 0.814 > trial 72 0.811; seeds 0-4: 0.808 < 0.816):
  the MLP settings are tied.
- Caveats: the seed check reuses the search's folds (fold seed 42), so it favours trial 112 slightly; the numbers
  are selected on the same folds and are optimistic.
- Open question (not acted on): images with consecutive file numbers look alike (train + val, centred per camera:
  cosine 0.23 vs 0.00 for random pairs; Remidio both-glaucoma pairs 33 vs 12 by chance). They may be the two eyes of
  one patient, which the split cannot group because Chakshu gives no patient ID.

## Gaussian noise on the training features (Hamilton job 19361172, 1 h 34 min)

- Every training batch becomes x + sigma * N(0, 1) (features are standardised, so sigma = 1 is one standard deviation);
  validation is never changed. The noise has its own random generator, so batch order and dropout are the same at every
  level; sigma = 0 repeats all 75 runs of `check_mlp_seeds.py` exactly.
- Overfitting goes down: at sigma 1-2 training loss no longer reaches 0 and validation loss no longer explodes
  (fixed split, epoch 300: 3 Nov setting 4.81 -> 1.59 -> 0.95; trial 112 2.30 -> 0.93 -> 0.90). Gap at the kept epoch:
  3 Nov 0.259 -> 0.247 (sigma 1) -> 0.174 (sigma 2); trial 72 0.194 -> 0.150 -> 0.065; trial 112 0.292 -> 0.241 -> 0.149.
  Kept epoch: 3 Nov 12 -> 25 -> 73; trial 112 58 -> 126 -> 268.
- AUROC does not move: at most +0.004 (3 Nov, sigma 1; better in 2 of 5 folds), inside the seed range. At sigma 2
  trial 72 underfits (-0.016, better in none of the 5 folds); trial 112 did not converge (11 of 25 runs stopped by the
  300-epoch limit, so its -0.005 is for unfinished models); the 3 Nov setting is unchanged (+0.001).
- Conclusion: overfitting can be controlled (heavy dropout + L1/L2, or feature noise), but it is not what limits the MLP.
  Early stopping already keeps the model before it overfits; every MLP and logistic regression (0.8135 on the same folds)
  reach 0.80-0.82, so the limit is the information in the frozen features with 128 glaucoma images per fold.

## All four classifiers by 5-fold cross-validation (Hamilton jobs 19405597, 15 min, and 19405706, step 5 only)

- Tuning, 5-fold mean AUROC: logistic regression C 0.00725 -> 0.814 (folds 0.814 / 0.828 / 0.792 / 0.770 / 0.867; every C
  from 0.0029 to 0.015 is within 0.002); SVM RBF, C 4.16, gamma 0.00084 -> 0.818 (best linear SVM 0.805); kNN k 39,
  distance-weighted, cosine -> 0.749; MLP trial 112 from search 2.
- Step 5, out-of-fold scores of the 1,141 training + validation images (160 glaucoma):

  | Classifier | Out-of-fold AUROC (95% CI) | Best minus this (95% CI) | Tied | AUPRC | Specificity at 90% sensitivity |
  |---|---|---|---|---|---|
  | Logistic regression (385 parameters) | 0.814 (0.779-0.848) | 0.003 (-0.014 to 0.021) | yes | 0.414 | 0.518 |
  | kNN | 0.747 (0.708-0.785) | 0.070 (0.037 to 0.104) | no | 0.337 | 0.455 |
  | SVM, RBF (531 support vectors) | 0.817 (0.784-0.850) | best | - | 0.426 | 0.495 |
  | MLP (460,289 parameters) | 0.813 (0.780-0.846) | 0.004 (-0.013 to 0.021) | yes | 0.414 | 0.527 |

- Rule fixed before looking (tied if the CI of best minus this includes 0; simplest tied wins): logistic regression,
  refitted on all 1,141 images (`chosen.json`, `chosen_model.joblib`). Thresholds from its out-of-fold scores:
  0.2828 for 90% sensitivity (144 of 160 glaucoma found, 473 of 981 healthy flagged) and 0.4724 by Youden's J
  (118 of 160 found, 234 of 981 flagged).
- The MLP's out-of-fold scores come from runs trained for the median kept epoch of the other four folds; early stopping
  on the scored fold itself would have given it about +0.004 that the other classifiers do not get.
- Caveats: hyperparameters were tuned on the same 5 folds that compare them (no nested cross-validation), so every
  out-of-fold AUROC is a little optimistic, the MLP's most (300 trials over 10 hyperparameters). The test part gives the
  unbiased number (step 6, not run). Read `step5_oof_scores.csv` with `pd.read_csv(..., float_precision='round_trip')`:
  the default reader drops the last digit, and one glaucoma image sits exactly on the 90% threshold.
- `make_figures.py` (updated 8 Nov): figures 1-5 describe this split and the 5-fold procedure.


## Search ranges: edge check (8 Nov, Hamilton job 19405708, 12 min)

- Rule: if the best value sits at the edge of the search range, the search may have been cut short; widen the range
  and search again (as for the SVM's C on 3 Nov: best C 0.001 at the lower bound, re-run with C down to 1e-5).
- `check_search_edges.py [db]` (reads only): only the 5-fold MLP search has its best values at an edge: width 512
  (upper bound; 10 of the 10 best trials) and lr 1.6e-5 (lower bound 1e-5; 9 of 10). Logistic regression, SVM and kNN
  are inside their ranges; search 1's L1 at its upper bound was already widened in search 2.
- `check_mlp_edges.py` + `run_edges.slurm` (array of 6): trial 112 beyond each edge, 5 seeds x the same 5 folds, up to
  1500 epochs (none reached). Kept-model AUROC: as found 0.8082; lr / 2, / 4, / 8: 0.8056 / 0.8050 / 0.8045 (worse in
  all 5 folds; kept epoch 118 / 228 / 451); width 1024: 0.8075; width 2048: 0.8103 (+0.002, 4 of 5 folds, inside the
  seed range, 5.0 M parameters). The edges hid no better model, so no new search. Results `check_mlp_edges.csv`,
  figure `figures/overfitting_8_search_edges.png` (`python check_mlp_edges.py summary`).

## Test, run once (Hamilton job 19405953, 8 Nov; `TEST_DONE` now blocks a second run)

- Logistic regression (C 0.00725, refitted on the 1,141 training + validation images) on the 204 test images
  (28 glaucoma): AUROC 0.872 (95% CI 0.810-0.923), AUPRC 0.538 (guessing 0.137). `test_results.json`, `test_scores.csv`,
  `figures/fig5_test.png`.
- 90%-sensitivity threshold from the out-of-fold scores (0.2828): 28 of 28 glaucoma found, 86 of 176 healthy flagged
  (specificity 0.51, PPV 0.25, NPV 1.00). Youden's J threshold (0.4724): 22 of 28 found, 46 of 176 flagged
  (specificity 0.74, PPV 0.32, NPV 0.96).
- The test AUROC is above the 5-fold estimate (0.814) but its CI includes it: with 28 glaucoma images one image moves
  the AUROC by up to about 0.036. Nothing may now be changed on the basis of the test result.
- Open question from before: images with consecutive file numbers look alike (possibly fellow eyes); if fellow eyes sit
  on both sides of the split, the test AUROC is optimistic.

