"""Numbers asked for by the literature-review markers (28 Nov), all from saved out-of-fold scores, nothing retrained
except the frozen SVM on the camera hold-out (seconds).
(a) healthy images flagged at 90 % sensitivity, with 95 % CIs, and paired differences against the LoRA rerun
    (threshold re-chosen inside every bootstrap resample, so its uncertainty is included)
(b) the same comparisons on seed-averaged scores (mean of the 3 seeds' out-of-fold scores) for AUPRC and AUROC
(c) leave-one-camera-out AUPRC with CIs for the frozen SVM and LoRA (seed-averaged scores)
Bootstrap: 2,000 resamples of image groups (`eye`), seed 42, as everywhere else.  ->  report_stats.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

HERE = Path(__file__).resolve().parent; R = HERE.parent
os.chdir(R / 'after_10_Nov'); sys.path.insert(0, str(R / 'after_10_Nov'))
from common import cv_split, load_data  # noqa: E402
lab_all, Xall, yall = load_data(); keep, splits = cv_split(lab_all)
lab = lab_all.loc[keep].reset_index(drop=True); X = Xall[keep]
ids = lab['image_id'].astype(str).to_numpy(); y = lab['label_majority'].to_numpy(); groups = lab['eye'].to_numpy()
cams = lab['camera'].to_numpy()
ug = np.unique(groups); gidx = {g: np.where(groups == g)[0] for g in ug}
rng = np.random.default_rng(42); BOOT = [np.concatenate([gidx[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]


def load(f, key):
    z = np.load(f); assert (z['image_id'].astype(str) == ids).all(), f; return z[key]


def seedmean(f):
    z = np.load(f); assert (z['image_id'].astype(str) == ids).all(), f
    return np.mean([z[f'seed{i}'] for i in range(3)], axis=0)


def fpr90(yy, s):
    p = np.sort(s[yy == 1])[::-1]; t = p[int(np.ceil(0.9 * len(p))) - 1]
    return 100 * ((s >= t) & (yy == 0)).sum() / (yy == 0).sum()


def ci(f, *ss):
    vals = []
    for b in BOOT:
        if y[b].sum() == 0 or y[b].sum() == len(b):
            continue
        vals.append(f(y[b], *[s[b] for s in ss]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


A22 = R / 'after_22_Nov'
models = {'LoRA rerun (22 Nov)': A22 / 'g_base.npz', 'LoRA + mask head 0.25': A22 / 'g_mask025.npz',
          'LoRA + mask head 1': A22 / 'g_mask1.npz', 'LoRA + stronger augmentation': A22 / 'g_strong.npz',
          'LoRA + 2 open blocks': HERE / 'g_lora_blocks.npz'}
out = {'a_fpr90_seed0': {}, 'b_seedmean': {}, 'c_camera_holdout': {}}
base0 = load(models['LoRA rerun (22 Nov)'], 'seed0'); basem = seedmean(models['LoRA rerun (22 Nov)'])
for n, f in models.items():
    s0 = load(f, 'seed0'); sm = seedmean(f)
    out['a_fpr90_seed0'][n] = {'fpr': round(fpr90(y, s0), 1), 'ci': ci(fpr90, s0)}
    out['b_seedmean'][n] = {'auprc': round(ap(y, sm), 3), 'auprc_ci': ci(ap, sm), 'auroc': round(auc(y, sm), 3), 'auroc_ci': ci(auc, sm),
                            'fpr90': round(fpr90(y, sm), 1), 'fpr90_ci': ci(fpr90, sm)}
    if n != 'LoRA rerun (22 Nov)':
        out['a_fpr90_seed0'][n]['minus_rerun'] = round(fpr90(y, s0) - fpr90(y, base0), 1)
        out['a_fpr90_seed0'][n]['minus_rerun_ci'] = ci(lambda yy, a, b: fpr90(yy, a) - fpr90(yy, b), s0, base0)
        for met, fn in (('auprc', ap), ('auroc', auc), ('fpr90', fpr90)):
            out['b_seedmean'][n][f'{met}_minus_rerun'] = round(fn(y, sm) - fn(y, basem), 3)
            out['b_seedmean'][n][f'{met}_minus_rerun_ci'] = ci(lambda yy, a, b: fn(yy, a) - fn(yy, b), sm, basem)
# 17 Nov models (seed 0 only saved for those) and the frozen SVM
ref = pd.read_csv(R / 'after_10_Nov' / 'step5_oof_scores.csv', float_precision='round_trip')
assert (ref['image_id'].astype(str).to_numpy() == ids).all()
svm = ref['svm'].to_numpy(); lr = ref['logreg'].to_numpy()
out['b_seedmean']['frozen SVM'] = {'auprc': round(ap(y, svm), 3), 'auroc': round(auc(y, svm), 3), 'fpr90': round(fpr90(y, svm), 1),
                                    'fpr90_ci': ci(fpr90, svm)}
for n in ('LoRA + mask head 0.25', 'LoRA + 2 open blocks', 'LoRA rerun (22 Nov)'):
    sm = seedmean(models[n])
    out['b_seedmean'][n]['auprc_minus_svm'] = round(ap(y, sm) - ap(y, svm), 3)
    out['b_seedmean'][n]['auprc_minus_svm_ci'] = ci(lambda yy, a, b: ap(yy, a) - ap(yy, b), sm, svm)
# (c) camera hold-out: LoRA rerun trained on the other two cameras (g_camera_base.npz) vs frozen SVM refitted the same way
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
cb = np.load(A22 / 'g_camera_base.npz'); assert (cb['image_id'].astype(str) == ids).all()
lora_c = np.mean([cb[f'seed{i}'] for i in range(3)], axis=0)
for c in ('Remidio', 'Bosch', 'Forus'):
    va = np.where(cams == c)[0]; tr = np.where(cams != c)[0]
    sc = StandardScaler().fit(X[tr])
    s = SVC(kernel='rbf', C=6.055286457910599, gamma=0.0005741621088974824, class_weight='balanced').fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[va]))
    out['c_camera_holdout'][c] = {'images': int(len(va)), 'glaucoma': int(y[va].sum()), 'chance': round(float(y[va].mean()), 3),
                                  'frozen_svm': round(ap(y[va], s), 3), 'lora': round(ap(y[va], lora_c[va]), 3)}
    # CIs on the held-out camera: resample that camera's image groups
    gv = groups[va]; ugv = np.unique(gv); gi = {g: np.where(gv == g)[0] for g in ugv}; r2 = np.random.default_rng(42)
    bs = [np.concatenate([gi[g] for g in r2.choice(ugv, len(ugv))]) for _ in range(2000)]
    yy = y[va]; lv = lora_c[va]
    def cib(fn, *ss):
        v = [fn(yy[b], *[q[b] for q in ss]) for b in bs if 0 < yy[b].sum() < len(b)]
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    out['c_camera_holdout'][c].update({'frozen_svm_ci': cib(ap, s), 'lora_ci': cib(ap, lv),
                                       'lora_minus_svm': round(ap(yy, lv) - ap(yy, s), 3), 'lora_minus_svm_ci': cib(lambda t, a, b: ap(t, a) - ap(t, b), lv, s)})
json.dump(out, open(HERE / "report_stats.json", "w"), indent=1)

# (d) seeds pooled properly: metric averaged over the 3 seeds (each seed scored on its own), paired against the rerun,
#     bootstrap over image groups with the same resample for every seed
def seeds(f):
    z = np.load(f); assert (z['image_id'].astype(str) == ids).all(), f; return [z[f'seed{i}'] for i in range(3)]


def mean_metric(fn, ss, b=None):
    return float(np.mean([fn(y if b is None else y[b], s if b is None else s[b]) for s in ss]))


out['d_seed_averaged_metric'] = {}
bs = seeds(models['LoRA rerun (22 Nov)'])
for n, f in models.items():
    ss = seeds(f); row = {}
    for met, fn in (('auprc', ap), ('auroc', auc), ('fpr90', fpr90)):
        row[met] = round(mean_metric(fn, ss), 3); row[met + '_per_seed'] = [round(fn(y, s), 3) for s in ss]
        if n != 'LoRA rerun (22 Nov)':
            d = [mean_metric(fn, ss, b) - mean_metric(fn, bs, b) for b in BOOT]
            row[met + '_minus_rerun'] = round(mean_metric(fn, ss) - mean_metric(fn, bs), 3)
            row[met + '_minus_rerun_ci'] = [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]
    out['d_seed_averaged_metric'][n] = row
json.dump(out, open(HERE / 'report_stats.json', 'w'), indent=1)
print(json.dumps(out['d_seed_averaged_metric'], indent=0))

# (e) frozen SVM against LoRA models with the metric averaged over seeds (same rule as d), and the camera hold-out with
#     the metric averaged over seeds; 2,000 resamples of image groups (per camera for the hold-out)
out['e_vs_svm_seed_averaged'] = {}
for n in ('LoRA rerun (22 Nov)', 'LoRA + mask head 0.25', 'LoRA + 2 open blocks'):
    ss = seeds(models[n]); row = {}
    for met, fn in (('auprc', ap), ('auroc', auc)):
        d = [mean_metric(fn, ss, b) - fn(y[b], svm[b]) for b in BOOT]
        row[met + '_minus_svm'] = round(mean_metric(fn, ss) - fn(y, svm), 3)
        row[met + '_minus_svm_ci'] = [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]
    out['e_vs_svm_seed_averaged'][n] = row
out['e_camera_holdout_seed_averaged'] = {}
S = [cb[f'seed{i}'] for i in range(3)]
for c in ('Remidio', 'Bosch', 'Forus'):
    va = np.where(cams == c)[0]; tr = np.where(cams != c)[0]; sc = StandardScaler().fit(X[tr])
    s = SVC(kernel='rbf', C=6.055286457910599, gamma=0.0005741621088974824, class_weight='balanced').fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[va]))
    yy = y[va]; gv = groups[va]; ugv = np.unique(gv); gi = {g: np.where(gv == g)[0] for g in ugv}; r2 = np.random.default_rng(42)
    bs = [np.concatenate([gi[g] for g in r2.choice(ugv, len(ugv))]) for _ in range(2000)]
    m = lambda b: float(np.mean([ap(yy[b], q[va][b]) for q in S]))
    d = [m(b) - ap(yy[b], s[b]) for b in bs if 0 < yy[b].sum() < len(b)]
    out['e_camera_holdout_seed_averaged'][c] = {'lora': round(m(np.arange(len(va))), 3), 'svm': round(ap(yy, s), 3),
        'diff': round(m(np.arange(len(va))) - ap(yy, s), 3), 'diff_ci': [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]}
json.dump(out, open(HERE / 'report_stats.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('e_vs_svm_seed_averaged', 'e_camera_holdout_seed_averaged')}, indent=0))
print(json.dumps({k: {m: v[m] for m in v if 'minus_rerun' in m} for k, v in out['d_seed_averaged_metric'].items()}, indent=0))
