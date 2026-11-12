# Harvard-FairVLMed: the larger dataset (from 11 Nov 2026)

Status: searches, comparison and per-group results DONE 11 Nov 2026 (training + validation patients only; the 2,000
test patients have not been used). Figures in `figures/`, made by `make_figures.py`.

Rahele (10 Nov) suggested a larger dataset on Hamilton next to Chakshu; on 11 Nov she named it:
https://huggingface.co/datasets/harvardairobotics/FairVLMed (FairCLIP, CVPR 2024; Harvard Ophthalmology AI Lab).

## The dataset

- 10,000 patients, one SLO (scanning laser ophthalmoscopy) fundus image each, 664 x 512 pixels, single channel.
  Not colour fundus photographs like PAPILA and Chakshu, and RETFound-Green was trained on colour photographs:
  expect a domain gap.
- Glaucoma 5,048 / 10,000 (balanced, unlike Chakshu's 14 %). The dataset's own split: training 7,000 (3,535 glaucoma),
  validation 1,000 (490), test 2,000 (1,023).
- Per patient also: age, gender (5,631 F / 4,369 M), race (819 Asian, 1,491 Black, 7,690 White), ethnicity, language,
  marital status. The npz files hold no visual-field values (the dataset card lists md / tds, but they are not in the
  files), so nothing that leaks the label is used; only the image goes into the model.
- One image per patient: no fellow-eye problem.
- Licence CC BY-NC-ND 4.0 (non-commercial, no derivatives): images, the label table, features and per-image scores
  stay on Hamilton; only code and summary results are on GitHub (`.gitignore`).
- Download (11 Nov, approved by Alison): `Dataset/dataset.zip`, 2,668,667,218 bytes, sha256 01bbfba4...7153, to
  `/nobackup/tggq61/data/raw/FairVLMed/`; unzipped to `/nobackup/tggq61/data/FairVLMed/{Training,Validation,Test}/`
  (data_#####.npz + slo_fundus_#####.jpg per patient).
- Images are dark: over 200 training images the median 99th-percentile grey level is 64 of 255 (maximum 12-255).

## Plan (same method as Chakshu, `../after_10_Nov`)

1. `make_labels.py` -> `fairvlmed_labels.csv` (Hamilton only).
2. `extract_features.py` (GPU, `run_extract.slurm`): frozen RETFound-Green, 384 features per image, two inputs:
   `raw` (grey copied to 3 channels, official resize to 392 and mean / std 0.5) and `stretch` (per-image contrast
   stretch, 1st-99th percentile to 0-255, then the same).
3. `choose_input.py`: logistic regression by 5-fold cross-validation on training + validation (8,000) for both inputs;
   the better one is used from then on.
4. `step3_auprc.py` (copy of the Chakshu one): logistic regression, SVM, kNN and the small MLP (2 layers of 32-128,
   300 trials on 16 workers) tuned by 5-fold mean AUPRC (guessing level here 0.5).
5. `step5_compare.py`: out-of-fold comparison, confusion matrices, and AUROC per race and per gender.
6. `step6_fairness_ci.py`: confidence intervals for the per-group results, and the differences between groups.
7. The 2,000 test patients are not used until the model is chosen (and then only once, when Alison decides).

## Log

- 11 Nov: download, unzip, label table (job 19429685 started with the label table and feature extraction).
- Job 19429685 (one H200 slice): 10,000 images per input in 30 s; features (10,000 x 384) for `raw` and `stretch` in
  `/nobackup/tggq61/retfound_green/features/fairvlmed_rfg_{raw,stretch}.npz`.
- `choose_input.py` (8,000 training + validation patients, logistic regression, C chosen by 5-fold CV):
  raw: C 0.0117, 5-fold mean AUPRC 0.796 (folds 0.785 / 0.794 / 0.805 / 0.794 / 0.801), AUROC 0.786;
  stretch: C 0.0189, AUPRC 0.795 (0.773 / 0.789 / 0.810 / 0.799 / 0.803), AUROC 0.786.
  A tie, so the simpler official input (`raw`) is used from here on (`PREPROC=raw`).
  First result: frozen RETFound-Green features on SLO images reach AUROC about 0.79 with logistic regression,
  close to Chakshu (0.81) even though RETFound-Green never saw SLO images. AUPRC is high (0.80) because glaucoma is
  50 % here (guessing = 0.5), so AUPRC is not comparable with Chakshu's 0.41 (guessing 0.14).
- Jobs 19429774 (logistic regression, kNN, SVM searches), 19429775 (small MLP, 16 workers, journal
  `optuna_mlp_small_raw.log`), 19429776 (comparison, starts after both).
- Job 19429775 (16 workers, 11:40-12:55): small-MLP search finished, 315 trials (the workers stop when 300 are
  complete; trials already started are finished and all count).
- Job 19429774 (12:10-20:10): logistic regression (50 trials, 7 min) and kNN (200 settings, 3 min) finished. **The SVM
  search reached the 8-hour limit after 48 of its 100 trials** (36 RBF, 12 linear; a 49th, linear, was still running).
  The time went on the linear kernel, which is very slow at large C on 6,400 training rows: the 12 linear trials took
  427 minutes together (one with C = 99 took 272 minutes), the 36 RBF trials 34 minutes. The best linear trial
  (AUPRC 0.792) is below the best RBF trial (0.802), and the six best RBF trials lie within 0.0006 of each other, so
  the missing trials are unlikely to change the result; the comparison uses the best of the 48.
- Job 19429776 (20:10-20:27): `step5_compare.py`, started when 19429774 ended.
- Job 19433239 (21:24, 2 min): `step6_fairness_ci.py`.
- Copied to the Mac and GitHub: `step5_comparison.csv`, `step5_confusion.csv`, `step5_fairness.csv`,
  `step5_summary.json`, `step6_fairness_ci.csv`, `step6_fairness_diff.csv`, the Optuna studies (`optuna_raw.db`,
  `optuna_mlp_small_raw.log`: hyperparameters and fold scores only) and the job logs. The label table, the features and
  `step5_oof_scores.csv` (one score per patient) stay on Hamilton.

## Results (8,000 training + validation patients, 4,025 glaucoma; the 2,000 test patients are not used)

AUPRC of guessing is 0.50 here (Chakshu: 0.14), so AUPRC values are not comparable between the two datasets; AUROC is.

## Result 1: classifiers tuned by AUPRC (step3_auprc.py; 5-fold mean validation AUPRC; `figures/fig5_searches.png`)

| Classifier | Trials | Best setting | 5-fold mean AUPRC | Folds | 5-fold mean AUROC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 50 | C = 0.0152 | 0.796 | 0.785 / 0.794 / 0.805 / 0.795 / 0.800 | 0.786 |
| SVM | 48 of 100 | RBF, C = 14.4, gamma = 0.00025 | 0.802 | 0.796 / 0.803 / 0.810 / 0.796 / 0.805 | 0.791 |
| kNN | 200 (grid) | k = 63, distance weights, cosine | 0.734 | 0.727 / 0.718 / 0.750 / 0.730 / 0.744 | 0.724 |
| Small MLP | 315 | 128 and 64 neurons, AdamW, lr 0.00004 (constant), dropout 0.67, L2 0.0000014, L1 0.00011, batch 128, weighted cross-entropy x 0.53 | 0.808 | 0.804 / 0.812 / 0.816 / 0.798 / 0.809 | 0.793 |

- The same kinds of setting win as on Chakshu: a strong L2 penalty for logistic regression (C 0.015; Chakshu 0.008),
  an RBF kernel for the SVM, and cosine distance with distance weights and many neighbours for kNN.
- Small MLP (`figures/fig6_small_mlp_search.png`): 57,601 parameters (Chakshu: 14,497). With seven times more
  patients the search prefers the widest first layer: 212 of the 315 trials used 128-128 and the best used 128-64.
  Kept epochs 175 / 264 / 210 / 134 / 163; validation loss minus training loss 0.055, so no overfitting.
- **Several best values lie at or near the edge of the search range**: first layer 128 (largest allowed), dropout 0.67
  (limit 0.7), L2 0.0000014 (limit 0.000001), weight scale 0.53 (limit 0.5). By the rule used on Chakshu (best at the
  edge: widen and search again) a wider search is an open item; the gain so far from the MLP over the SVM is 0.003.
- Loss: weighted cross-entropy in 294 trials, focal 13, plain cross-entropy 8. The best plain trial (0.806) is as good
  as the best weighted one (0.808): with 50 % glaucoma the class weight hardly matters, unlike on Chakshu, where
  leaving it out cost 0.11 AUPRC.

## Result 2: comparison and confusion matrices (step5_compare.py; out-of-fold; `figures/fig1_comparison.png`)

| Classifier | AUPRC (95 % CI) | AUROC (95 % CI) | Best minus this, AUPRC (95 % CI) | Tied with best |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.796 (0.783-0.809) | 0.785 (0.775-0.795) | 0.010 (0.005 to 0.014) | no |
| SVM | 0.802 (0.790-0.814) | 0.791 (0.781-0.801) | 0.003 (-0.0001 to 0.007) | yes |
| kNN | 0.733 (0.719-0.748) | 0.723 (0.713-0.734) | 0.072 (0.062 to 0.082) | no |
| Small MLP (57,601 parameters) | 0.805 (0.793-0.818) | 0.791 (0.782-0.801) | best | - |

Confusion matrices (out-of-fold; `figures/fig2_confusion_sens90.png`, `fig2_confusion_youden.png`):

| Classifier | Threshold | Glaucoma found | Missed | Healthy flagged | Healthy cleared | Sensitivity | Specificity | PPV | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 90 % sensitivity | 3,623 | 402 | 2,429 | 1,546 | 0.900 | 0.389 | 0.599 | 0.719 | 0.337 |
| Logistic regression | Youden's J | 2,556 | 1,469 | 800 | 3,175 | 0.635 | 0.799 | 0.762 | 0.693 | 0.439 |
| SVM | 90 % sensitivity | 3,623 | 402 | 2,348 | 1,627 | 0.900 | 0.409 | 0.607 | 0.725 | 0.356 |
| SVM | Youden's J | 2,855 | 1,170 | 1,061 | 2,914 | 0.709 | 0.733 | 0.729 | 0.719 | 0.442 |
| kNN | 90 % sensitivity | 3,623 | 402 | 2,824 | 1,151 | 0.900 | 0.290 | 0.562 | 0.692 | 0.240 |
| kNN | Youden's J | 2,686 | 1,339 | 1,313 | 2,662 | 0.667 | 0.670 | 0.672 | 0.669 | 0.337 |
| Small MLP | 90 % sensitivity | 3,623 | 402 | 2,379 | 1,596 | 0.900 | 0.402 | 0.604 | 0.723 | 0.348 |
| Small MLP | Youden's J | 2,630 | 1,395 | 855 | 3,120 | 0.653 | 0.785 | 0.755 | 0.700 | 0.442 |

- By the rule fixed in advance (tied if the CI of the AUPRC difference includes 0; simplest tied wins) the choice here
  is the **SVM**: the small MLP is highest, the SVM is tied with it (only just: the CI starts at -0.0001), and logistic
  regression and kNN are not tied.
- Logistic regression is 0.010 AUPRC and 0.006 AUROC below the best. That gap counts as real only because 8,000
  patients make the intervals narrow (about +-0.013; on Chakshu +-0.08); in practice the three models are close, as on
  Chakshu, and kNN is again clearly worse.
- At the screening threshold (90 % sensitivity) the SVM flags the fewest healthy patients (2,348 of 3,975; small MLP
  2,379, logistic regression 2,429), but specificity is only about 0.40: six of ten healthy patients would be referred.
- Frozen RETFound-Green features reach AUROC 0.79 on SLO images, which the model never saw in training, against 0.81
  out-of-fold on Chakshu colour photographs.

## Result 3: race and gender (step5_compare.py, step6_fairness_ci.py; out-of-fold)

Out-of-fold AUROC with 95 % CI (2,000 resamples of the patients of each group; `figures/fig3_fairness_auroc.png`):

| Group | Patients | Glaucoma | Logistic regression | SVM | kNN | Small MLP |
| --- | --- | --- | --- | --- | --- | --- |
| Asian | 661 | 320 (48 %) | 0.819 (0.786-0.850) | 0.821 (0.789-0.851) | 0.746 (0.706-0.782) | 0.820 (0.787-0.851) |
| Black | 1,186 | 772 (65 %) | 0.767 (0.739-0.793) | 0.769 (0.742-0.794) | 0.715 (0.685-0.743) | 0.773 (0.747-0.799) |
| White | 6,153 | 2,933 (48 %) | 0.784 (0.772-0.795) | 0.790 (0.778-0.800) | 0.718 (0.705-0.731) | 0.790 (0.779-0.801) |
| Female | 4,535 | 2,235 (49 %) | 0.783 (0.769-0.796) | 0.788 (0.775-0.801) | 0.722 (0.707-0.736) | 0.789 (0.776-0.802) |
| Male | 3,465 | 1,790 (52 %) | 0.788 (0.773-0.802) | 0.793 (0.779-0.807) | 0.725 (0.708-0.740) | 0.794 (0.779-0.807) |

AUROC differences between groups for the SVM (the other classifiers are in `step6_fairness_diff.csv`):

| Difference | AUROC difference (95 % CI) | CI excludes 0 |
| --- | --- | --- |
| Asian minus White | 0.032 (-0.003 to 0.064) | no |
| Black minus White | -0.021 (-0.050 to 0.006) | no |
| Asian minus Black | 0.053 (0.012 to 0.094) | yes |
| Male minus female | 0.005 (-0.014 to 0.024) | no |

SVM at the one threshold set on all 8,000 patients for 90 % sensitivity (`figures/fig4_fairness_rates.png`):

| Group | Glaucoma found | Sensitivity | Healthy flagged | Specificity |
| --- | --- | --- | --- | --- |
| Asian | 302 of 320 | 0.944 | 207 of 341 | 0.393 |
| Black | 697 of 772 | 0.903 | 286 of 414 | 0.309 |
| White | 2,624 of 2,933 | 0.895 | 1,855 of 3,220 | 0.424 |
| Female | 1,998 of 2,235 | 0.894 | 1,352 of 2,300 | 0.412 |
| Male | 1,625 of 1,790 | 0.908 | 996 of 1,675 | 0.405 |

- **Ranking (AUROC)**: highest for Asian patients (0.82), lowest for Black patients (0.77), White patients between
  (0.79). Only the Asian-Black difference has a CI that excludes 0 (0.053; true for logistic regression, the SVM and
  the small MLP, not for kNN); neither group differs clearly from White patients. Men and women are the same (0.005).
- **One threshold does not give the groups the same error rates.** At the 90 % sensitivity threshold the SVM flags 69 %
  of healthy Black patients against 58 % of healthy White patients (specificity 0.31 against 0.42; difference -0.115,
  CI -0.163 to -0.068), and finds 94 % of the Asian glaucoma patients against 90 % of the White ones (difference
  0.049, CI 0.019 to 0.077). The same pattern holds for logistic regression and the small MLP.
- AUPRC per group is in `step5_fairness.csv` but should not be compared between groups: it rises with the share of
  glaucoma, which is 65 % among Black patients and 48 % among Asian and White patients.
- The intervals are not corrected for the number of comparisons, and the Asian group is small (661 patients).

## Summary for the next meeting

1. FairVLMed is on Hamilton and the Chakshu method runs on it unchanged: 8,000 patients for 5-fold cross-validation,
   2,000 test patients untouched.
2. Frozen RETFound-Green features give AUROC 0.79 on these SLO images (Chakshu 0.81), although the model was trained
   on colour photographs only.
3. Small MLP 0.805 and SVM 0.802 AUPRC are tied; logistic regression 0.796 is just below; kNN 0.733 is clearly worse.
   By the fixed rule the SVM is chosen here (on Chakshu: logistic regression).
4. At 90 % sensitivity the specificity is about 0.40.
5. By group: AUROC 0.82 Asian, 0.79 White, 0.77 Black; no difference between men and women. One common threshold
   flags healthy Black patients more often than healthy White patients (69 % against 58 %).
6. Open: the SVM search stopped at 48 of 100 trials (time limit); several best MLP values are at the edge of the
   search range; when and how to use the 2,000 test patients.

## Still running (submitted 12 Nov, 12:26): the rest of the Chakshu pipeline

- Job array 19438500 (`run_svm_resume.slurm`, 6 workers): finishes the SVM search, from 48 to 100 trials, in the same
  study with the same search space (`RESUME=1 python step3_auprc.py svm`).
- Job 19438499: `check_losses.py` (5 losses x 5 seeds x 5 folds) and `step4_learning_curve.py` (300 epochs on the
  fixed 7,000 / 1,000 split), both copied from `../after_10_Nov`.
- Job 19438501 (after the SVM search): `step5_compare.py` and `step6_fairness_ci.py` again with the final SVM setting.
  The tables above are from the 48-trial SVM search; a copy is kept on Hamilton in `first_run_48_svm_trials/`.
