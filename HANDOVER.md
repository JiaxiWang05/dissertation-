# Handover (state on 15 Nov 2026)

Where everything is, what is finished, what is still running, and the rules to keep. Read this first in a new session.

## Folders (this repo = Mac `~/Desktop/dissertation` = Hamilton `~/glaucoma`)

| Folder | What | State |
| --- | --- | --- |
| `pipeline/` (Hamilton: `retfound_green`) | October PAPILA -> Chakshu work: feature extraction, linear probe, Optuna head / fine-tuning, fairness, crops, ablation, learning curve | finished |
| `3_Nov/` | Chakshu only, one fixed validation split (60/20/20); shown on 3 Nov | finished, superseded |
| `after_3_Nov/` | 70/15/15 split, 7 copies kept, 5-fold CV for all classifiers, MLP overfitting study, **test run once on 8 Nov** (`TEST_DONE` lock: logistic regression, test AUROC 0.872, 95 % CI 0.810-0.923) | finished; test set must not be used again |
| `after_10_Nov/` | tasks from the 10 Nov meeting: AUPRC objective, small MLP (32-64 neurons, 14,497 parameters), loss functions, confusion matrices, left/right eye and fellow eyes | finished; `README.md` has every number |
| `fairvlmed/` | Harvard-FairVLMed, the larger dataset Rahele suggested: the Chakshu pipeline replicated | finished 12 Nov; test set not used |
| `reports/` (own private repo `dissertation-reports`) | Research Proposal & PMR, literature review (LaTeX, Overleaf) | drafts done |

## Main results so far

- Chakshu, frozen RETFound-Green features: logistic regression, SVM and MLP tie at out-of-fold AUROC about 0.81
  (AUPRC about 0.41-0.43, guessing 0.14); kNN worse. Test (once): AUROC 0.872; 28 of 28 glaucoma found at the 90 %
  sensitivity threshold, specificity 0.51.
- MLP: overfitting removed by a smaller network (32 and 64 neurons); AUROC unchanged. Class-weighted loss is essential
  (no weight: AUPRC 0.32 and unstable); weight about 2 x (healthy / glaucoma) best; focal loss worse.
- Strongest single effect found: strong L2 penalty in logistic regression (C about 0.007 vs 10-100: AUROC 0.81 vs 0.74).
- Chakshu has no patient or eye ID. Grouping = only the 30 Bosch file pairs (P#_Image1/2) and the 7 exact copies
  (1,308 groups). Do not write "grouped by eye" or "1,308 eyes".
- Left / right eye: not in the image headers; on Remidio the optic-disc position separates the eyes. Consecutive
  Remidio numbers (IMG_n, IMG_n+1) are opposite eyes in 93 % of pairs = almost certainly fellow eyes (433 pairs;
  121 of the 204 test images have their fellow eye in training/validation). Keeping the pairs together changes the
  cross-validated AUROC by only about 0.004 (`after_10_Nov/laterality_pairs.py`, `pseudo_patients.csv`).
- FairVLMed first result: logistic regression on frozen features, 5-fold CV on 8,000 patients: AUROC 0.786,
  AUPRC 0.796 (guessing 0.5); raw and contrast-stretched inputs tie, raw used.

## FairVLMed (finished 12 Nov)

- The whole Chakshu pipeline of `after_10_Nov` replicated on FairVLMed (jobs 19429774-6, 19433239, 19438499-501);
  every number is in `fairvlmed/README.md`, figures in `fairvlmed/figures/` (`make_figures.py`).
- Out-of-fold on 8,000 patients: small MLP AUPRC 0.805 / AUROC 0.791, SVM 0.803 / 0.791 (tied; chosen by the simplest-
  tied rule), logistic regression 0.796 / 0.785, kNN 0.733 / 0.723. Loss function makes no difference (balanced classes).
  AUROC by race: Asian 0.82, White 0.79, Black 0.77 (only Asian-Black CI excludes 0); men = women.
- The 2,000 FairVLMed test patients are not used until the user decides.

## To-do from the 17 Nov meeting (for the next meeting; RETFound-Green stays the one model)

- [ ] Own small CNN from scratch (no foundation model) as the baseline for every comparison; Optuna chooses layers and
      kernels; on Chakshu and FairVLMed.
- [ ] Class weight: 6.1 looks too strong (healthy images called glaucoma, F1 about 0.39). Optuna tunes the weight,
      including values below 1, objective F1 or AUPRC. (Already searched: scale 0.5-2 with AUPRC, Optuna chose 1.83.)
- [ ] Confusion matrices as percentages so the two datasets compare; Chakshu matrices split by camera.
- [ ] Repeat everything on FairVLMed (no class weight: 50 % glaucoma).
- [ ] If time: unfreeze layers, Optuna chooses 0-3 open layers plus the MLP.
- Notes: absolute loss values are not comparable between models / datasets (only trends); AUPRC only between equally
  balanced sets; FairVLMed (grey SLO) -> Chakshu (colour) is a welcome external domain test; send the Draft Research
  Proposal (due 7 Dec) and the literature review (due 1 Feb) to Rahele before submitting; she will sign support
  letters the student drafts (big data, Hamilton, fine-tuning foundation models; read up on world models).
- Overnight jobs were killed at about 8 h: that is the `--time` limit in the job script; request more or split work.

## Older open items (with Rahele)

- Flip all images to one side (only possible for Remidio, from `after_10_Nov/remidio_disc_side.csv`).
- FairVLMed results (done 12 Nov): discuss; wider MLP search (best values at the edge); when to use its test set.
- Non-linear logistic regression (asked on 3 Nov, not done).
- Later: nested CV, unfreezing RETFound-Green blocks, augmentation, other foundation models, PCA / t-SNE.
- Possibly redo the Chakshu split by fellow-eye pairs (Rahele's decision; the test set has been used once).
- The "grouped by eye" wording is still in `reports/` (lit_review.tex lines 96, 214, 282; proposal_pmr.tex lines 69,
  88, 115, 249) and in `after_3_Nov/make_figures.py` Figure 1.
- A draft email asking Rahele for the dataset name was not needed in the end (she sent the FairVLMed link).

## Notion (workspace of tggq61; the hub is shared with Rahele)

- Hub "Dissertation: glaucoma detection with retinal foundation models" (id 3fc942e9-f0e9-8131-a91a-dfb2f30b25bf):
  plain report style, no emoji. Sections: data, method, results, limitations, decisions per meeting (16/20/27 Oct,
  3 Nov, 10 Nov with tick-box to-dos), next steps, open items, reading, files.
- Subpages: "Results update, 9 November" (only work since 3 Nov) and "Old results" (October to 3 Nov; the five
  original meeting / methods / reading pages sit under it).
- Private pages (never move under the hub): MLP overfitting write-up, meeting scripts, guides.

## Rules the user set

- Push every new change to GitHub (`JiaxiWang05/dissertation-`, PRIVATE since 15 Nov 2026, branch main).
- FairVLMed is CC BY-NC-ND 4.0: never push its images, label table, features or per-image scores (`.gitignore`).
- Real data only; test sets are used once; the user approves downloads and the one-time test run.
- Public Notion pages: clean, professional, no AI traces; no to-do reminders at the end of answers.
