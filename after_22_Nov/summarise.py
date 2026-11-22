"""22 Nov: statistics for the EndoWAM-inspired runs (g_<cfg>.npz from grounding_cv.py).
Per variant: fair AUPRC (3-seed mean, range), seed 0 AUPRC with grouped-bootstrap CI, AUROC, healthy flagged at 90 %
sensitivity, disc Dice of the mask head, paired AUPRC difference (seed 0, 2,000 grouped resamples) to the LoRA rerun
('base') and to the best frozen model (SVM). Camera hold-out: AUPRC on each held-out camera (3-seed mean) for the image
models and for the frozen logistic regression / SVM (10 Nov settings) trained on the other two cameras, next to the
in-distribution score of the same camera (out-of-fold, base run).
CPU: python summarise.py -> grounding_summary.csv, camera_holdout.csv"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'after_10_Nov'))
os.chdir(HERE.parent / 'after_10_Nov')
import common  # noqa: E402
lab, X, yall = common.load_data(); keep, _ = common.cv_split(lab)
os.chdir(HERE)
sub = lab[keep].reset_index(drop=True); groups = sub['eye'].to_numpy()
ref = pd.read_csv('../after_10_Nov/step5_oof_scores.csv', float_precision='round_trip')
assert (ref['image_id'].to_numpy() == sub['image_id'].to_numpy()).all()
svm = ref['svm'].to_numpy()

base = np.load('g_base.npz'); y = base['y']
assert (base['image_id'] == sub['image_id'].to_numpy().astype(str)).all()
rows = []
for cfg in ('base', 'strong', 'mask025', 'mask1', 'disc'):
    f = HERE / f'g_{cfg}.npz'
    if not f.exists():
        print('missing', f.name); continue
    z = np.load(f); s0 = z['seed0']; aps = [ap(y, z[f'seed{i}']) for i in range(3)]
    t = common.threshold_at_recall(y, s0, 0.9)
    d_b, ci_b = common.paired_difference(ap, y, s0, base['seed0'], groups)
    d_f, ci_f = common.paired_difference(ap, y, s0, svm, groups)
    d_auc, ci_auc = common.paired_difference(auc, y, s0, base['seed0'], groups)
    dice = np.nanmean([np.nanmean(z[f'dice{i}']) for i in range(3)]) if 'mask' in cfg else np.nan
    rows.append({'variant': cfg, 'AUPRC 3-seed mean': np.mean(aps), 'AUPRC range': [round(min(aps), 3), round(max(aps), 3)],
                 'AUPRC seed 0': aps[0], 'AUPRC CI': np.round(common.bootstrap_ci(ap, y, s0, groups), 3).tolist(),
                 'AUROC 3-seed mean': np.mean([auc(y, z[f'seed{i}']) for i in range(3)]),
                 'early-stopped AUPRC (optimistic)': np.mean([ap(y, z[f'es{i}']) for i in range(3)]),
                 'healthy flagged % at 90 % sens': 100 * ((s0 >= t) & (y == 0)).sum() / (y == 0).sum(),
                 'disc Dice (val)': dice, 'fixed epochs': [z[f'fixed{i}'].tolist() for i in range(3)],
                 'minus base AUPRC': d_b, 'CI vs base': [round(v, 3) for v in ci_b],
                 'minus base AUROC': d_auc, 'AUROC CI vs base': [round(v, 3) for v in ci_auc],
                 'minus frozen SVM AUPRC': d_f, 'CI vs frozen': [round(v, 3) for v in ci_f]})
out = pd.DataFrame(rows); out.to_csv('grounding_summary.csv', index=False)
pd.set_option('display.width', 250); print(out.T.to_string())

cams = sub['camera'].to_numpy(); C = ('Remidio', 'Bosch', 'Forus')
crow = []
Xk = X[keep]
for name, mk in [('frozen logistic regression', lambda: LogisticRegression(C=0.007250175039761133, class_weight='balanced', max_iter=5000)),
                 ('frozen SVM', lambda: SVC(kernel='rbf', C=6.055286457910599, gamma=0.0005741621088974824, class_weight='balanced'))]:
    r = {'model': name}
    for c in C:
        tr, va = cams != c, cams == c
        sc = StandardScaler().fit(Xk[tr]); m = mk().fit(sc.transform(Xk[tr]), y[tr])
        s = m.decision_function(sc.transform(Xk[va]))
        r[f'{c} held out'] = ap(y[va], s); r[f'{c} in-distribution'] = ap(y[va], ref['svm' if 'SVM' in name else 'logreg'].to_numpy()[va])
    crow.append(r)
for cfg in ('camera_base', 'camera_strong', 'camera_mask1'):
    f = HERE / f'g_{cfg}.npz'
    if not f.exists():
        print('missing', f.name); continue
    z = np.load(f); r = {'model': 'LoRA ' + cfg.replace('camera_', '')}
    ind = np.load(f"g_{cfg.replace('camera_', '')}.npz") if (HERE / f"g_{cfg.replace('camera_', '')}.npz").exists() else None
    for c in C:
        va = cams == c
        r[f'{c} held out'] = np.mean([ap(y[va], z[f'seed{i}'][va]) for i in range(3)])
        r[f'{c} in-distribution'] = np.mean([ap(y[va], ind[f'seed{i}'][va]) for i in range(3)]) if ind is not None else np.nan
    crow.append(r)
co = pd.DataFrame(crow)
for c in C:
    co[f'{c} glaucoma / images'] = f'{int(y[cams == c].sum())} / {int((cams == c).sum())}'
co.to_csv('camera_holdout.csv', index=False); print(co.T.to_string())
