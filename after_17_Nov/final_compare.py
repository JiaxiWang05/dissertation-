"""17 Nov: compare the image-level models with the frozen-feature models on the same images (out-of-fold; tests unused).
For each finished search (cnn_chaksu, cnn_fairvlmed, unfreeze_chaksu) the best setting is retrained in every fold with
3 seeds; out-of-fold AUPRC / AUROC (seed 0 for the CI and the paired difference, 3-seed mean and range as well) and
the confusion matrix in percent at the 90 % sensitivity threshold. Reference: the frozen-feature classifiers' out-of-fold
scores (../after_10_Nov/step5_oof_scores.csv, ../fairvlmed/step5_oof_scores.csv on Hamilton only).
GPU:  python final_compare.py chaksu | fairvlmed   ->  final_<ds>.csv, final_<ds>_oof.npz (Hamilton only)"""
import os
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from img_common import data, storage

DS = sys.argv[1]
sys.argv = [sys.argv[0], DS]                       # cnn_cv reads the dataset from argv
import cnn_cv  # noqa: E402

sys.path.insert(0, '../after_10_Nov')
from common import bootstrap_ci, paired_difference, threshold_at_recall  # noqa: E402

models = {'cnn': cnn_cv}
if DS == 'chaksu':
    import unfreeze_cv
    models['unfreeze'] = unfreeze_cv
ref_file = '../after_10_Nov/step5_oof_scores.csv' if DS == 'chaksu' else '../fairvlmed/step5_oof_scores.csv'
ref = pd.read_csv(ref_file, float_precision='round_trip')
PSEUDO = os.environ.get('FOLDS') == 'pseudo'
TAG = f'{DS}_pseudo' if PSEUDO else DS
if PSEUDO:
    # frozen-feature reference recomputed on the same fellow-eye-grouped folds: logistic regression (C 0.00725) and
    # SVM (RBF, 10 Nov AUPRC-tuned setting) on the frozen RETFound-Green features
    import os as _os
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from common import load_data as _load
    _cwd = _os.getcwd(); _os.chdir('../after_10_Nov'); _lab, _X, _y = _load(); _os.chdir(_cwd)
    _keep = _lab['part'].isin(['train', 'val']).to_numpy(); _Xs = _X[_keep]
    _, _yy, _splits, _sub = data(DS, 256)
    ref = pd.DataFrame({'image_id': _sub['image_id'], 'label': _yy})
    for nm, mk in [('logreg', lambda: LogisticRegression(C=0.007250175039761133, class_weight='balanced', max_iter=5000)),
                   ('svm', lambda: SVC(kernel='rbf', C=6.055286457910599, gamma=0.0005741621088974824, class_weight='balanced'))]:
        o = np.zeros(len(_yy))
        for tr, va in _splits:
            sc = StandardScaler().fit(_Xs[tr]); m = mk().fit(sc.transform(_Xs[tr]), _yy[tr])
            o[va] = m.decision_function(sc.transform(_Xs[va])) if nm == 'svm' else m.predict_proba(sc.transform(_Xs[va]))[:, 1]
        ref[nm] = o
        print('frozen', nm, 'on fellow-eye folds: AUPRC', round(average_precision_score(_yy, o), 4), 'AUROC', round(roc_auc_score(_yy, o), 4), flush=True)
rows, keep_oof = [], {}
for name, mod in models.items():
    study = optuna.load_study(study_name=f'{name}_{DS}', storage=storage(f'{name}_{DS}'))
    b = study.best_trial
    p = {**b.params}
    if name == 'cnn' and 'w_scale' not in p:
        p['w_scale'] = 1.0
    X, y, splits, lab = data(DS, 256 if name == 'cnn' else 392)
    assert (lab['image_id'].to_numpy() == ref['image_id'].to_numpy()).all()
    seeds, es_scores = [], []
    for seed in range(3):
        # pass 1: early stopping on each fold gives that fold's kept epoch (its scores are optimistic and only reported)
        _, _, epochs, oof_es, n_par = mod.run(p, X, y, splits, seed=seed)
        es_scores.append(oof_es)
        # pass 2: each fold trains for the median kept epoch of the OTHER folds and is scored at its last epoch
        fixed = [max(1, int(np.median([e for j, e in enumerate(epochs) if j != k]))) for k in range(len(splits))]
        _, _, _, oof, _ = mod.run(p, X, y, splits, seed=seed, fixed=fixed)
        seeds.append(oof)
        print(name, 'seed', seed, 'fair AUPRC', round(average_precision_score(y, oof), 4), 'AUROC', round(roc_auc_score(y, oof), 4),
              '| early-stopped AUPRC', round(average_precision_score(y, oof_es), 4), '| epochs', epochs, 'fixed', fixed, flush=True)
    keep_oof[name] = seeds[0]
    groups = (lab['image_id'].map(pd.read_csv('../after_10_Nov/pseudo_patients.csv').set_index('image_id')['pseudo_patient']).to_numpy()
              if PSEUDO else lab['eye'].to_numpy() if 'eye' in lab else lab['image_id'].to_numpy())
    s = seeds[0]
    best_ref = max([c for c in ref.columns if c not in ('image_id', 'label')],
                   key=lambda c: average_precision_score(y, ref[c]))
    d, (dlo, dhi) = paired_difference(average_precision_score, y, s, ref[best_ref].to_numpy(), groups)
    t = threshold_at_recall(y, s, 0.9)
    pred = s >= t
    rows.append({'model': name, 'trials': len(study.trials), 'best search AUPRC': b.value, 'setting': p,
                 'parameters (trainable)': n_par,
                 'AUPRC seed 0': average_precision_score(y, s),
                 'early-stopped AUPRC (optimistic), 3-seed mean': np.mean([average_precision_score(y, o) for o in es_scores]),
                 'AUPRC CI': np.round(bootstrap_ci(average_precision_score, y, s, groups), 3).tolist(),
                 'AUPRC 3 seeds mean': np.mean([average_precision_score(y, o) for o in seeds]),
                 'AUPRC 3 seeds range': [round(min(average_precision_score(y, o) for o in seeds), 3),
                                         round(max(average_precision_score(y, o) for o in seeds), 3)],
                 'AUROC seed 0': roc_auc_score(y, s),
                 'AUROC CI': np.round(bootstrap_ci(roc_auc_score, y, s, groups), 3).tolist(),
                 'best frozen-feature model': best_ref,
                 'frozen AUPRC': average_precision_score(y, ref[best_ref]), 'frozen AUROC': roc_auc_score(y, ref[best_ref]),
                 'this minus frozen AUPRC': d, 'diff CI': [round(dlo, 3), round(dhi, 3)],
                 'healthy flagged % at 90 % sens': 100 * (pred & (y == 0)).sum() / (y == 0).sum()})
out = pd.DataFrame(rows)
out.to_csv(f'final_{TAG}.csv', index=False)
np.savez(f'final_{TAG}_oof.npz', image_id=ref['image_id'].to_numpy().astype(str), **keep_oof)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(out.T.to_string())
