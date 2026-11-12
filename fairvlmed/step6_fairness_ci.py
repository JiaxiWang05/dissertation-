"""Step 6 (11 Nov): confidence intervals for the per-group results of step5_compare.py. The test part is NOT used.

Reads the out-of-fold scores of the 8,000 training + validation patients (step5_oof_scores.csv, Hamilton only) and
the label table, and writes two summary tables (no per-patient rows):

step6_fairness_ci.csv    per attribute group and classifier: patients, glaucoma, AUROC with a 95 % CI (2,000 resamples
                         of the patients of that group), and the confusion counts, sensitivity and specificity of the
                         group at the two thresholds set on all 8,000 patients (90 % sensitivity, Youden's J).
step6_fairness_diff.csv  AUROC difference between two groups with a 95 % CI (the groups are resampled independently),
                         and the difference in sensitivity and specificity at the two thresholds.
A difference counts as real only when its CI excludes 0."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import SEED, at_threshold, cv_split, load_data, threshold_at_recall, youden_threshold

N_BOOT = 2000
ORDER = ['logreg', 'svm', 'knn', 'mlp_small']
ATTRS = [('race', {0: 'Asian', 1: 'Black', 2: 'White'}), ('gender', {0: 'female', 1: 'male'})]
PAIRS = {'race': [('Asian', 'White'), ('Black', 'White'), ('Asian', 'Black')], 'gender': [('male', 'female')]}

lab, _, _ = load_data()
keep, _ = cv_split(lab)
sub = lab[keep].reset_index(drop=True)
oof = pd.read_csv('step5_oof_scores.csv', float_precision='round_trip')
assert (oof['image_id'].to_numpy() == sub['image_id'].to_numpy()).all() and (oof['label'] == sub['label']).all()
y = oof['label'].to_numpy()
thresholds = {n: {'90% sensitivity': threshold_at_recall(y, oof[n].to_numpy(), 0.9),
                  "Youden's J": youden_threshold(y, oof[n].to_numpy())} for n in ORDER}

rows, diffs = [], []
for col, names in ATTRS:
    g = sub[col].to_numpy()
    boot, point, rates = {}, {}, {}
    for v, nm in names.items():
        idx = np.flatnonzero(g == v)
        rng = np.random.default_rng(SEED)                 # the same resamples for every classifier of this group
        picks = [rng.choice(idx, len(idx)) for _ in range(N_BOOT)]
        for n in ORDER:
            s = oof[n].to_numpy()
            point[nm, n] = roc_auc_score(y[idx], s[idx])
            boot[nm, n] = np.array([roc_auc_score(y[p], s[p]) for p in picks])
            lo, hi = np.percentile(boot[nm, n], [2.5, 97.5])
            row = {'attribute': col, 'group': nm, 'patients': len(idx), 'glaucoma': int(y[idx].sum()),
                   'classifier': n, 'AUROC': point[nm, n], 'AUROC CI low': lo, 'AUROC CI high': hi}
            for rule, tag in [('90% sensitivity', 'sens90'), ("Youden's J", 'youden')]:
                thr = thresholds[n][rule]
                r = at_threshold(y[idx], s[idx], thr)
                sens_b = np.array([((s[p] >= thr) & (y[p] == 1)).sum() / (y[p] == 1).sum() for p in picks])
                spec_b = np.array([((s[p] < thr) & (y[p] == 0)).sum() / (y[p] == 0).sum() for p in picks])
                rates[nm, n, tag] = (r['sensitivity'], r['specificity'], sens_b, spec_b)
                row.update({f'{tag} TP': r['TP'], f'{tag} FN': r['FN'], f'{tag} FP': r['FP'], f'{tag} TN': r['TN'],
                            f'{tag} sensitivity': r['sensitivity'], f'{tag} specificity': r['specificity']})
            rows.append(row)
    for a, b in PAIRS[col]:
        for n in ORDER:
            d = boot[a, n] - boot[b, n]                   # independent groups: independent resamples
            lo, hi = np.percentile(d, [2.5, 97.5])
            row = {'attribute': col, 'difference': f'{a} minus {b}', 'classifier': n,
                   'AUROC difference': point[a, n] - point[b, n], 'CI low': lo, 'CI high': hi,
                   'CI excludes 0': bool(lo > 0 or hi < 0)}
            for tag in ['sens90', 'youden']:
                sa, pa, sab, pab = rates[a, n, tag]
                sb, pb, sbb, pbb = rates[b, n, tag]
                for name, pt, bt in [('sensitivity', sa - sb, sab - sbb), ('specificity', pa - pb, pab - pbb)]:
                    lo, hi = np.percentile(bt, [2.5, 97.5])
                    row.update({f'{tag} {name} difference': pt, f'{tag} {name} CI low': lo,
                                f'{tag} {name} CI high': hi})
            diffs.append(row)

fair = pd.DataFrame(rows)
diff = pd.DataFrame(diffs)
fair.to_csv('step6_fairness_ci.csv', index=False)
diff.to_csv('step6_fairness_diff.csv', index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
print(fair.round(3).to_string(index=False))
print(diff.round(3).to_string(index=False))
