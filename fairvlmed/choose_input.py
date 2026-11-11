"""Which input works better for the SLO images, raw or contrast-stretched? Logistic regression (C chosen by 5-fold
cross-validation over 1e-4..1e2, 30 log-spaced values) on each feature set; out-of-fold AUROC and AUPRC on the
8,000 training + validation patients. The test part is not used.   python choose_input.py"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from common import cv_folds, load_data

for pp in ['raw', 'stretch']:
    lab, X, y = load_data(f'/nobackup/tggq61/retfound_green/features/fairvlmed_rfg_{pp}.npz')
    folds = cv_folds(lab, X, y)
    best = None
    for C in np.logspace(-4, 2, 30):
        ap, auc = [], []
        for Xtr, ytr, Xva, yva in folds:
            s = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
            ap.append(average_precision_score(yva, s)); auc.append(roc_auc_score(yva, s))
        if best is None or np.mean(ap) > best[1]:
            best = (C, np.mean(ap), np.mean(auc), ap)
    print(f'{pp}: best C {best[0]:.4g}; 5-fold mean AUPRC {best[1]:.3f} (folds {np.round(best[3], 3).tolist()}), '
          f'AUROC {best[2]:.3f}', flush=True)
