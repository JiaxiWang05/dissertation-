"""22 Nov (borrowed from EndoWAM, which reads its features from an intermediate block, not the last one): do the frozen
RETFound-Green features of earlier blocks classify glaucoma better than the last block's?
For every training + validation image (1,141; test untouched) and every block 1-12: output tokens -> the model's final
layer norm -> mean of the 784 patch tokens = 384 features (block 12 = exactly the features used so far). Also blocks 9-12
concatenated (1,536). Classifiers per feature set, honest out-of-fold on the same 5 folds as before: logistic regression
and RBF SVM, each with its setting chosen by an inner 3-fold grid search on the training part only (AUPRC). Reference
check: the 10 Nov SVM setting on block 12 must give the stored out-of-fold AUPRC 0.427.
GPU: python layer_features.py  ->  layer_features.csv, layer_features_oof.npz, /nobackup/.../chaksu_layers.npz"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pipeline')); sys.path.insert(0, str(HERE.parent / 'after_10_Nov'))
import rfg  # noqa: E402
os.chdir(HERE.parent / 'after_10_Nov')
import common  # noqa: E402
lab, Xold, y_all = common.load_data(); keep, splits = common.cv_split(lab)
os.chdir(HERE)
WEIGHTS = '/nobackup/tggq61/weights/retfoundgreen_statedict.pth'
OUT = Path('/nobackup/tggq61/retfound_green/cache/chaksu_layers.npz')
sub = lab[keep].reset_index(drop=True); y = y_all[keep]; groups = sub['eye'].to_numpy()

if not OUT.exists():
    model = rfg.load_retfound_green(WEIGHTS).eval().cuda()
    npt = model.num_prefix_tokens
    store = {}
    for i, blk in enumerate(model.blocks):
        blk.register_forward_hook(lambda m, inp, out, i=i: store.__setitem__(i, out))
    ds = rfg.FundusImages(sub['path'].tolist(), rfg.eval_transform(), 'resize')
    feats = np.zeros((12, len(sub), 384), np.float32); final = np.zeros((len(sub), 384), np.float32)
    with torch.inference_mode():
        for x, idx in DataLoader(ds, batch_size=32, num_workers=2):
            f = model(x.cuda()).float()
            final[idx.numpy()] = f.cpu().numpy()
            for i in range(12):
                feats[i, idx.numpy()] = model.norm(store[i])[:, npt:].mean(1).float().cpu().numpy()
    print('block 12 vs model output, max abs diff', float(np.abs(feats[11] - final).max()))
    print('block 12 vs stored 10 Nov features, max abs diff', float(np.abs(feats[11] - Xold[keep]).max()), flush=True)
    np.savez(OUT, feats=feats, image_id=sub['image_id'].to_numpy().astype(str))
feats = np.load(OUT)['feats']

ref = pd.read_csv('../after_10_Nov/step5_oof_scores.csv', float_precision='round_trip')
assert (ref['image_id'].to_numpy() == sub['image_id'].to_numpy()).all()
inner = StratifiedGroupKFold(3, shuffle=True, random_state=42)


def oof_scores(X, kind):
    o, chosen = np.zeros(len(y)), []
    for tr, va in splits:
        if kind == 'logreg':
            est = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=5000))
            grid = {'logisticregression__C': np.logspace(-4, 1, 11)}
        else:
            d = X.shape[1]
            est = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced'))
            grid = {'svc__C': [0.3, 1, 3, 10, 30], 'svc__gamma': [0.1 / d, 0.3 / d, 1 / d, 3 / d]}
        gs = GridSearchCV(est, grid, scoring='average_precision', cv=inner, n_jobs=-1)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        o[va] = gs.decision_function(X[va]); chosen.append(gs.best_params_)
    return o, chosen


# reference check: the 10 Nov SVM setting on block 12
chk = np.zeros(len(y))
for tr, va in splits:
    sc = StandardScaler().fit(feats[11][tr])
    m = SVC(kernel='rbf', C=6.055286457910599, gamma=0.0005741621088974824, class_weight='balanced').fit(sc.transform(feats[11][tr]), y[tr])
    chk[va] = m.decision_function(sc.transform(feats[11][va]))
print('check: 10 Nov SVM on block 12, AUPRC', round(average_precision_score(y, chk), 4),
      'stored', round(average_precision_score(y, ref['svm']), 4), flush=True)

sets = {f'block {i + 1}': feats[i] for i in range(12)}
sets['blocks 9-12 joined'] = np.concatenate([feats[i] for i in range(8, 12)], 1)
rows, oofs = [], {}
for name, X in sets.items():
    for kind in ('logreg', 'svm'):
        o, chosen = oof_scores(X, kind)
        oofs[f'{name}|{kind}'] = o
        rows.append({'features': name, 'classifier': kind, 'AUPRC': average_precision_score(y, o), 'AUROC': roc_auc_score(y, o),
                     'AUPRC CI': np.round(common.bootstrap_ci(average_precision_score, y, o, groups), 3).tolist(),
                     'chosen per fold': chosen})
        print(name, kind, 'AUPRC', round(rows[-1]['AUPRC'], 4), 'AUROC', round(rows[-1]['AUROC'], 4), flush=True)
out = pd.DataFrame(rows)
for kind in ('logreg', 'svm'):
    base = oofs[f'block 12|{kind}']
    for i in out.index[out['classifier'] == kind]:
        d, (lo, hi) = common.paired_difference(average_precision_score, y, oofs[f"{out.at[i, 'features']}|{kind}"], base, groups)
        out.at[i, 'minus block 12'] = d; out.at[i, 'diff CI'] = str([round(lo, 3), round(hi, 3)])
out.to_csv('layer_features.csv', index=False)
np.savez('layer_features_oof.npz', image_id=sub['image_id'].to_numpy().astype(str), y=y, **oofs)
print(out.drop(columns='chosen per fold').to_string())
