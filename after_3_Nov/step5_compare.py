"""Step 5: compare the four tuned classifiers by 5-fold cross-validation, choose one, set its thresholds,
refit it on training + validation and save it. The test part is not used.
Each classifier's best setting (step 3) is refitted in each of the 5 folds and scores that fold's validation images,
so every training + validation image gets one out-of-fold score (1,141 in all). The AUROC, its confidence interval,
the tie rule and the thresholds all use these out-of-fold scores."""
import json

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (STORAGE, at_threshold, bootstrap_auroc_ci, cv_folds, cv_split, load_data, mlp_scores, n_trainable,
                    paired_auroc_difference, threshold_at_recall, train_mlp, youden_threshold)

lab, X, y = load_data()
keep, splits = cv_split(lab)
folds = cv_folds(lab, X, y)                      # the same 5 folds as step 3, scaled on each fold's training rows
Xtv, ytv = X[keep], y[keep]                      # training + validation
eyes = lab.loc[keep, 'eye'].to_numpy()
best = {name: optuna.load_study(study_name=name, storage=STORAGE).best_params
        for name in ['logreg', 'knn', 'svm', 'mlp']}


def make_sklearn(name):
    """A fresh classifier with the best hyperparameters from step 3."""
    hp = best[name]
    if name == 'logreg':
        return LogisticRegression(C=hp['C'], class_weight='balanced', max_iter=5000)
    if name == 'knn':
        return KNeighborsClassifier(n_neighbors=hp['k'], weights=hp['weights'], metric=hp['metric'])
    return SVC(kernel=hp['kernel'], C=hp['C'], gamma=hp.get('gamma', 'scale'), class_weight='balanced')


def score(name, model, X):
    return model.decision_function(X) if name == 'svm' else model.predict_proba(X)[:, 1]   # SVM: signed distance


# out-of-fold scores: each image is scored by the model of the fold in which it was a validation image
oof = {name: np.zeros(len(ytv)) for name in best}
mlp_epochs, svm_support = [], []
for (tr, va), (Xtr, ytr, Xva, yva) in zip(splits, folds):
    for name in ['logreg', 'knn', 'svm']:
        model = make_sklearn(name).fit(Xtr, ytr)
        oof[name][va] = score(name, model, Xva)
        if name == 'svm':
            svm_support.append(int(model.n_support_.sum()))
    mlp, _, epoch = train_mlp(best['mlp'], Xtr, ytr, Xva, yva)   # early stopping on this fold's validation images
    mlp_epochs.append(epoch)
# The MLP's epoch above was chosen on the very images it would score. To keep the scored fold's labels out, as for the
# other classifiers, retrain each fold for the median kept epoch of the OTHER four folds (the final-fit rule below).
for k, ((tr, va), (Xtr, ytr, Xva, yva)) in enumerate(zip(splits, folds)):
    epochs = int(np.median([e for j, e in enumerate(mlp_epochs) if j != k]))
    oof['mlp'][va] = mlp_scores(train_mlp(best['mlp'], Xtr, ytr, max_epochs=epochs)[0], Xva)

size = {'logreg': 'trainable parameters: 385 (384 weights + 1 bias)',
        'knn': 'trainable parameters: 0 (stores the training images x 384 features)',
        'svm': ('trainable parameters: 385 (384 weights + 1 bias)' if best['svm']['kernel'] == 'linear'
                else f'support vectors: {int(np.mean(svm_support))} (mean over folds)'),
        'mlp': f'trainable parameters: {n_trainable(mlp)} (kept epochs {mlp_epochs})'}
top = max(oof, key=lambda n: roc_auc_score(ytv, oof[n]))         # the classifier with the highest out-of-fold AUROC

rows = []
for name, s in oof.items():
    fold_auroc = [roc_auc_score(ytv[va], s[va]) for _, va in splits]
    thr = threshold_at_recall(ytv, s, 0.9)
    r = at_threshold(ytv, s, thr)
    lo, hi = bootstrap_auroc_ci(ytv, s, eyes)
    diff, (dlo, dhi) = paired_auroc_difference(ytv, oof[top], s, eyes)
    rows.append({'classifier': name, 'best hyperparameters': best[name], 'size': size[name],
                 'out-of-fold AUROC': roc_auc_score(ytv, s), 'AUROC CI low': lo, 'AUROC CI high': hi,
                 'mean of fold AUROCs': np.mean(fold_auroc), 'fold AUROC sd': np.std(fold_auroc),
                 'best minus this': diff, 'diff CI low': dlo, 'diff CI high': dhi,
                 'tied with best': dlo <= 0,
                 'AUPRC': average_precision_score(ytv, s),
                 'sensitivity @90% recall': r['sensitivity'], 'specificity @90% recall': r['specificity']})
table = pd.DataFrame(rows)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 60)
print(table.drop(columns='best hyperparameters').round(3).to_string(index=False))
table.to_csv('step5_comparison.csv', index=False)
scores = lab.loc[keep, ['image_id', 'camera', 'eye', 'part', 'label_majority']].copy()
for name, s in oof.items():
    scores[name] = s
scores.to_csv('step5_oof_scores.csv', index=False, float_format='%.17g')   # exact; training + validation only

# rule decided before looking: a classifier is tied with the best if the 95% CI of (best minus it) includes 0;
# among the tied ones, take the simplest
simplest_first = ['logreg', 'knn', 'svm', 'mlp']
chosen = next(n for n in simplest_first if table.set_index('classifier').loc[n, 'tied with best'])
s = oof[chosen]
info = {'classifier': chosen, 'hyperparameters': best[chosen],
        'trained on': f'training + validation ({len(ytv)} images)', 'thresholds from': 'out-of-fold scores',
        'threshold_recall90': threshold_at_recall(ytv, s, 0.9), 'threshold_youden': youden_threshold(ytv, s)}
# the final model learns from all training + validation images
if chosen == 'mlp':
    info['epochs'] = int(np.median(mlp_epochs))      # no validation part is left: train for the median kept epoch
    scaler = StandardScaler().fit(Xtv)
    mlp, _, _ = train_mlp(best['mlp'], scaler.transform(Xtv), ytv, max_epochs=info['epochs'])
    torch.save(mlp.state_dict(), 'chosen_mlp.pt')
    joblib.dump(scaler, 'chosen_scaler.joblib')
else:
    joblib.dump(make_pipeline(StandardScaler(), make_sklearn(chosen)).fit(Xtv, ytv), 'chosen_model.joblib')
json.dump(info, open('chosen.json', 'w'), indent=2)
print('\nhighest out-of-fold AUROC:', top, '  chosen:', chosen, '  refitted on', len(ytv), 'images',
      '  thresholds from the out-of-fold scores:', round(info['threshold_recall90'], 4), '(90% recall),',
      round(info['threshold_youden'], 4), "(Youden's J)")
