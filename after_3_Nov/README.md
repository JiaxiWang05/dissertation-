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
| Check: the 3 Nov setting at its own lr and at lr / 10, same 5 folds | `check_lr_epoch.py` |

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
