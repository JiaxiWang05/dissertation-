"""Fairness check with Fairlearn: does performance change between groups?

Chakshu: groups = camera (Remidio / Bosch / Forus), model trained on PAPILA only, threshold fixed from PAPILA.
PAPILA: groups = sex and age (below / at or above the median), out-of-fold predictions with each fold's threshold.
For each group: sensitivity (true positive rate), specificity, false positive rate, precision, selection rate, AUROC.
Per group: 95% cluster-bootstrap CIs. Fairlearn's difference = largest minus smallest value across groups.

Usage:
    python fairness.py --result results/frozen_lr_tuned_resize --papila-labels papila_labels.csv --out results/fairness
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_negative_rate, true_positive_rate
from sklearn.metrics import precision_score

import rfg

METRICS = {'sensitivity': true_positive_rate, 'specificity': true_negative_rate, 'false_positive_rate': false_positive_rate,
           'precision': lambda y, p: precision_score(y, p, zero_division=0), 'selection_rate': selection_rate}
GAPS = ['sensitivity', 'specificity', 'false_positive_rate']


def frame(y, yhat, groups):
    return MetricFrame(metrics=METRICS, y_true=y, y_pred=yhat, sensitive_features=groups)


def analyse(name, y, p, thr, groups, clusters):
    """Per-group metrics with 95% cluster-bootstrap CIs (rfg.evaluate), and Fairlearn's between-group differences.
    The difference (largest minus smallest) is descriptive: compare the per-group intervals to judge it."""
    y, p, groups, clusters = (np.asarray(y).astype(int), np.asarray(p, dtype=float), np.asarray(groups).astype(str),
                              np.asarray(clusters))
    thr = np.broadcast_to(np.asarray(thr, dtype=float), y.shape)
    yhat = (p >= thr).astype(int)
    mf = frame(y, yhat, groups)
    rows = []
    for g in sorted(np.unique(groups)):
        m = groups == g
        e = rfg.evaluate(y[m], p[m], clusters[m], thr=thr[m])
        rows.append({'group': g, 'n': e['n'], 'n_glaucoma': e['n_pos'], **{k: e[k] for k in ['auroc', 'sensitivity', 'specificity', 'precision']},
                     'ci95': {k: e['ci95'][k] for k in ['auroc', 'sensitivity', 'specificity', 'precision']},
                     'false_positive_rate': float(mf.by_group.loc[g, 'false_positive_rate'])})
    diff = mf.difference()
    return {'name': name, 'by_group': rows, 'difference': {k: float(diff[k]) for k in GAPS}}


def md(res):
    c = lambda r, k: f"{r[k]:.2f} ({r['ci95'][k][0]:.2f}–{r['ci95'][k][1]:.2f})"  # noqa: E731
    L = [f"### {res['name']}\n\n| Group | Images (glaucoma) | AUROC | Sensitivity | Specificity | Precision |\n"
         '|---|---|---|---|---|---|\n']
    for r in res['by_group']:
        L.append(f"| {r['group']} | {r['n']} ({r['n_glaucoma']}) | {c(r, 'auroc')} | {c(r, 'sensitivity')} | "
                 f"{c(r, 'specificity')} | {c(r, 'precision')} |\n")
    L.append('\nFairlearn difference (largest minus smallest): ' + '; '.join(
        f"{k.replace('_', ' ')} {v:.2f}" for k, v in res['difference'].items()) + '\n\n')
    return ''.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='a run_linear_probe.py output folder')
    ap.add_argument('--papila-labels', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    folder, out = Path(args.result), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((folder / 'metrics.json').read_text())

    cha = pd.read_csv(folder / 'chaksu_predictions.csv')
    groups_c = rfg.chaksu_external(Path(args.papila_labels).with_name('chaksu_labels.csv')).set_index('image_id')['group']
    results = [analyse('Chákṣu by camera (trained on PAPILA, threshold fixed from PAPILA)', cha['y'], cha['p_glaucoma'],
                       res['papila_full_fit_threshold'], cha['camera'], groups_c.loc[cha['image_id']].to_numpy())]

    pap = pd.read_csv(folder / 'papila_oof_predictions.csv')
    lab = pd.read_csv(args.papila_labels).set_index('image_id').loc[pap['image_id']]
    sex = np.where(lab['gender'].to_numpy() == 0, 'sex code 0', 'sex code 1')
    med = float(np.median(lab['age']))
    age = np.where(lab['age'].to_numpy() < med, f'age < {med:.0f}', f'age ≥ {med:.0f}')
    for name, g in [('PAPILA by sex (out-of-fold)', sex), ('PAPILA by age (out-of-fold)', age)]:
        results.append(analyse(name, pap['y'], pap['p_image'], pap['thr_image'], g, pap['patient_id']))

    (out / 'fairness.json').write_text(json.dumps(results, indent=2, default=float))
    text = '# Fairness (Fairlearn MetricFrame)\n\n' + ''.join(md(r) for r in results)
    (out / 'fairness.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
