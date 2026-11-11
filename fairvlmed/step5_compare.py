"""Step 5 (10 Nov): compare the AUPRC-tuned classifiers by 5-fold cross-validation, with confusion matrices.
FairVLMed copy of ../after_10_Nov/step5_compare.py. The test part (2,000 patients) is NOT used; everything is out-of-fold.

Each classifier's best setting (step3_auprc.py) is refitted in each of the 5 folds and scores that fold's validation
images, so all 1,141 training + validation images get one out-of-fold score. The MLP is retrained for the median kept
epoch of the other four folds, so the scored fold's labels never choose its stopping epoch.
Also writes step5_fairness.csv: out-of-fold AUROC and AUPRC per race and per gender.
Main measure: AUPRC with a 95 % CI (2,000 resamples of the image groups); tie rule: a classifier is tied with the best
when the CI of the paired AUPRC difference includes 0; the simplest tied classifier wins (order fixed in advance:
logistic regression, SVM, kNN, MLP).
Writes step5_comparison.csv, step5_oof_scores.csv, step5_confusion.csv."""
import json

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from common import (mlp_small_storage, at_threshold, bootstrap_ci, cv_folds, cv_split, load_data, make_mlp, mlp_scores, n_trainable,
                    paired_difference, threshold_at_recall, train_mlp, youden_threshold)

import os
STORAGE = f'sqlite:///optuna_{os.environ.get("PREPROC", "stretch")}.db'
lab, X, y = load_data()
keep, splits = cv_split(lab)
folds = cv_folds(lab, X, y)
ytv = y[keep]
groups = lab.loc[keep, 'eye'].to_numpy()                  # image groups (Bosch pairs and copies together)
best = {n: optuna.load_study(study_name=f'{n}_auprc', storage=STORAGE).best_params for n in ['logreg', 'svm', 'knn']}
mlp_hp = optuna.load_study(study_name='mlp_small', storage=mlp_small_storage()).best_params
mlp_params = {k: v for k, v in mlp_hp.items() if k not in ('width1', 'width2')}
mlp_params.update({'n_layers': 2, 'widths': [mlp_hp['width1'], mlp_hp['width2']]})
ORDER = ['logreg', 'svm', 'knn', 'mlp_small']   # simplest first


def make(name):
    hp = best[name]
    if name == 'logreg':
        return LogisticRegression(C=hp['C'], class_weight='balanced', max_iter=5000)
    if name == 'svm':
        return SVC(kernel=hp['kernel'], C=hp['C'], gamma=hp.get('gamma', 'scale'), class_weight='balanced')
    return KNeighborsClassifier(n_neighbors=hp['k'], weights=hp['weights'], metric=hp['metric'])


oof = {n: np.zeros(len(ytv)) for n in ORDER}
epochs = []
for (tr, va), (Xtr, ytr, Xva, yva) in zip(splits, folds):
    for n in ORDER[:-1]:
        m = make(n).fit(Xtr, ytr)
        oof[n][va] = m.decision_function(Xva) if n == 'svm' else m.predict_proba(Xva)[:, 1]
    epochs.append(train_mlp(mlp_params, Xtr, ytr, Xva, yva)[2])
for k, ((tr, va), (Xtr, ytr, Xva, yva)) in enumerate(zip(splits, folds)):
    e = int(np.median([x for j, x in enumerate(epochs) if j != k]))
    oof['mlp_small'][va] = mlp_scores(train_mlp(mlp_params, Xtr, ytr, max_epochs=e)[0], Xva)

top = max(ORDER, key=lambda n: average_precision_score(ytv, oof[n]))
rows, conf = [], []
for n in ORDER:
    s = oof[n]
    ap, auc = average_precision_score(ytv, s), roc_auc_score(ytv, s)
    ap_lo, ap_hi = bootstrap_ci(average_precision_score, ytv, s, groups)
    auc_lo, auc_hi = bootstrap_ci(roc_auc_score, ytv, s, groups)
    d, (d_lo, d_hi) = paired_difference(average_precision_score, ytv, oof[top], s, groups)
    rows.append({'classifier': n, 'AUPRC': ap, 'AUPRC CI low': ap_lo, 'AUPRC CI high': ap_hi,
                 'AUROC': auc, 'AUROC CI low': auc_lo, 'AUROC CI high': auc_hi,
                 'best minus this (AUPRC)': d, 'diff CI low': d_lo, 'diff CI high': d_hi,
                 'tied with best': bool(d_lo <= 0 <= d_hi),
                 'fold AUPRC': [round(average_precision_score(ytv[va], s[va]), 3) for _, va in splits]})
    for rule, thr in [('90% sensitivity', threshold_at_recall(ytv, s, 0.9)), ("Youden's J", youden_threshold(ytv, s))]:
        r = at_threshold(ytv, s, thr)
        r.update({'classifier': n, 'threshold rule': rule, 'threshold': thr,
                  'F1': f1_score(ytv, (s >= thr).astype(int))})
        conf.append(r)
comp = pd.DataFrame(rows)
chosen = next(n for n in ORDER if comp.set_index('classifier').loc[n, 'tied with best'])
comp.to_csv('step5_comparison.csv', index=False)
pd.DataFrame(conf).to_csv('step5_confusion.csv', index=False)
pd.DataFrame({'image_id': lab.loc[keep, 'image_id'].to_numpy(), 'label': ytv, **oof}).to_csv(
    'step5_oof_scores.csv', index=False, float_format='%.17g')
info = {'best hyperparameters': {**best, 'mlp_small': mlp_hp},
        'mlp_small trainable parameters': n_trainable(make_mlp(mlp_params['widths'], 0.0, X.shape[1])),
        'mlp_small kept epochs per fold': epochs, 'highest AUPRC': top, 'chosen (simplest tied)': chosen}
json.dump(info, open('step5_summary.json', 'w'), indent=2, default=str)
pd.set_option('display.width', 250)
print(comp.round(3).to_string(index=False))
print(pd.DataFrame(conf)[['classifier', 'threshold rule', 'TP', 'FN', 'FP', 'TN', 'sensitivity', 'specificity',
                          'PPV', 'F1', 'MCC']].round(3).to_string(index=False))
print(json.dumps(info, indent=2, default=str))

# fairness (Rahele's interest; the dataset was built for it): out-of-fold AUROC per group for every classifier
rows = []
for col, names in [('race', {0: 'Asian', 1: 'Black', 2: 'White'}), ('gender', {0: 'female', 1: 'male'})]:
    g = lab.loc[keep, col].to_numpy()
    for v, nm in names.items():
        m = g == v
        for n in ORDER:
            rows.append({'attribute': col, 'group': nm, 'patients': int(m.sum()), 'glaucoma': int(ytv[m].sum()),
                         'classifier': n, 'AUROC': roc_auc_score(ytv[m], oof[n][m]),
                         'AUPRC': average_precision_score(ytv[m], oof[n][m])})
fair = pd.DataFrame(rows)
fair.to_csv('step5_fairness.csv', index=False)
print(fair.round(3).to_string(index=False))
