"""Step 5: compare the four tuned classifiers on validation, choose one, set its threshold, save it."""
import json

import joblib
import optuna
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (STORAGE, at_threshold, bootstrap_auroc_ci, load_data, mlp_scores, n_trainable,
                    paired_auroc_difference, part, threshold_at_recall, train_mlp, youden_threshold)

lab, X, y = load_data()
Xtr, ytr = part(lab, X, y, 'train')
Xva, yva = part(lab, X, y, 'val')
eyes_va = lab.loc[lab['part'] == 'val', 'eye'].to_numpy()
best = {name: optuna.load_study(study_name=name, storage=STORAGE).best_params
        for name in ['logreg', 'knn', 'svm', 'mlp']}

# refit each classifier once with its best hyperparameters (training part) and score the validation part
lr = make_pipeline(StandardScaler(), LogisticRegression(C=best['logreg']['C'], class_weight='balanced', max_iter=5000))
lr.fit(Xtr, ytr)
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best['knn']['k'], weights=best['knn']['weights'],
                                                           metric=best['knn']['metric']))
knn.fit(Xtr, ytr)
svm = make_pipeline(StandardScaler(), SVC(kernel=best['svm']['kernel'], C=best['svm']['C'],
                                          gamma=best['svm'].get('gamma', 'scale'), class_weight='balanced'))
svm.fit(Xtr, ytr)
scaler = StandardScaler().fit(Xtr)
mlp, _, mlp_epoch = train_mlp(best['mlp'], scaler.transform(Xtr), ytr, scaler.transform(Xva), yva)

scores = {'logreg': lr.predict_proba(Xva)[:, 1], 'knn': knn.predict_proba(Xva)[:, 1],
          'svm': svm.decision_function(Xva), 'mlp': mlp_scores(mlp, scaler.transform(Xva))}
size = {'logreg': 'trainable parameters: 385 (384 weights + 1 bias)',
        'knn': f'trainable parameters: 0 (stores all {len(Xtr)} training images x 384 features)',
        'svm': (f"support vectors: {svm[-1].n_support_.sum()}" if best['svm']['kernel'] == 'rbf'
                else 'trainable parameters: 385 (384 weights + 1 bias)'),
        'mlp': f'trainable parameters: {n_trainable(mlp)} (kept epoch {mlp_epoch})'}
top = max(scores, key=lambda n: roc_auc_score(yva, scores[n]))     # the classifier with the highest validation AUROC

rows = []
for name, s in scores.items():
    thr = threshold_at_recall(yva, s, 0.9)
    r = at_threshold(yva, s, thr)
    lo, hi = bootstrap_auroc_ci(yva, s, eyes_va)
    diff, (dlo, dhi) = paired_auroc_difference(yva, scores[top], s, eyes_va)
    rows.append({'classifier': name, 'best hyperparameters': best[name], 'size': size[name],
                 'val AUROC': roc_auc_score(yva, s), 'AUROC CI low': lo, 'AUROC CI high': hi,
                 'best minus this': diff, 'diff CI low': dlo, 'diff CI high': dhi,
                 'tied with best': dlo <= 0,
                 'val AUPRC': average_precision_score(yva, s),
                 'sensitivity @90% recall': r['sensitivity'], 'specificity @90% recall': r['specificity']})
table = pd.DataFrame(rows)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 60)
print(table.drop(columns='best hyperparameters').round(3).to_string(index=False))
table.to_csv('step5_comparison.csv', index=False)

# rule decided before looking: a classifier is tied with the best if the 95% CI of (best minus it) includes 0;
# among the tied ones, take the simplest
simplest_first = ['logreg', 'knn', 'svm', 'mlp']
chosen = next(n for n in simplest_first if table.set_index('classifier').loc[n, 'tied with best'])
s = scores[chosen]
info = {'classifier': chosen, 'hyperparameters': best[chosen],
        'threshold_recall90': threshold_at_recall(yva, s, 0.9), 'threshold_youden': youden_threshold(yva, s)}
if chosen == 'mlp':
    torch.save(mlp.state_dict(), 'chosen_mlp.pt')
    joblib.dump(scaler, 'chosen_scaler.joblib')
else:
    joblib.dump({'logreg': lr, 'knn': knn, 'svm': svm}[chosen], 'chosen_model.joblib')
json.dump(info, open('chosen.json', 'w'), indent=2)
print('\nhighest validation AUROC:', top, '  chosen:', chosen,
      '  thresholds set on validation:', round(info['threshold_recall90'], 4), '(90% recall),',
      round(info['threshold_youden'], 4), "(Youden's J)")
