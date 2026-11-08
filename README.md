# Small Data, Large Models: Transfer Learning for Medical Image Classification

BEng Research and Development Project (ENGI 3262, Durham University, 2026-27): glaucoma detection from frozen
RETFound-Green features on the Chákṣu and PAPILA fundus datasets.

| Folder | Contents |
|---|---|
| `pipeline/` | Feature extraction, linear probe, Optuna over the head and over unfrozen layers, fairness and ablations (Oct 2026); its outputs are in `results/` |
| `3_Nov/` | Chákṣu, frozen features: split, Optuna for logistic regression, SVM, kNN and MLP, learning curve, model choice (as shown on 3 Nov) |
| `after_3_Nov/` | 70/15/15 split, all classifiers tuned and compared by 5-fold cross-validation, MLP overfitting checks, and the one-time test (logistic regression, test AUROC 0.872, 95% CI 0.810-0.923) |
| `results/` | Results and figures of `pipeline/` |

Each folder has its own README with the results. Data files (labels, splits, features), results (scores, predictions,
Optuna databases, logs) and figures are included; the fundus images are not. The scripts run on Durham's Hamilton
cluster (`run_*.slurm`).

Licences: code MIT (`LICENSE`); data and results keep the licences of the datasets they come from, Chákṣu CC BY 4.0
and PAPILA GPL-3.0-or-later (`DATA_LICENSES.md`).
