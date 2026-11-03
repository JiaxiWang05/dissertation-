"""Fairness mitigation on Chakshu by post-processing: decision thresholds set on a few labelled images per camera.

The PAPILA-trained model's scores stay fixed; only the threshold changes. Each random split puts a fraction of each
camera's eyes (stratified by label, at least 2 glaucoma eyes; paired Bosch images stay together) into a labelled
calibration set, and every rule is scored on the same remaining test images:
  papila_*    the threshold chosen on PAPILA (Youden's J or 90% sensitivity), no Chakshu labels: the current result
  pooled      one threshold for all cameras: 90% sensitivity on the pooled calibration images
  per_camera  one threshold per camera: at least 90% sensitivity on that camera's calibration images; with fewer
              than 10 calibration glaucoma images (Bosch, Forus) this keeps all of them (lowest glaucoma score)
  fairlearn   Fairlearn ThresholdOptimizer: equal sensitivity (true positive rate) across cameras on the
              calibration images, best balanced accuracy (so a lower sensitivity than 90%); it randomises
              predictions between two thresholds for some cameras
Spread = 2.5-97.5 percentiles over the random splits. Gap = largest minus smallest camera mean over the splits
(a single split's gap is mostly noise from the 13-14 glaucoma test images of Bosch and Forus).

Usage:
    python threshold_calibration.py --runs R/frozen_lr_tuned_resize R/frozen_lr_tuned_resize_sens90 \
        --chaksu-labels chaksu_labels.csv --out R/threshold_calibration_resize
(--runs: run_linear_probe.py folders with the same scores and different PAPILA threshold rules)
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

import rfg

CAMERAS = ['Remidio', 'Bosch', 'Forus']  # colour order of plot_curves.py (largest camera first)
PAPILA_RULES = {'youden': "PAPILA threshold, Youden's J", 'sens90': 'PAPILA threshold, 90% sensitivity'}
CALIBRATED = {'pooled': 'One Chákṣu threshold, 90% sensitivity',
              'per_camera': 'Threshold per camera, at least 90% sensitivity',
              'fairlearn': 'Fairlearn, equal sensitivity on calibration images'}
SENS90 = {'papila_sens90', 'pooled', 'per_camera'}
TICKS = {'papila_youden': "PAPILA\nYouden's J", 'papila_sens90': 'PAPILA\n90% sens.', 'pooled': 'Chákṣu, one\n90% sens.',
         'per_camera': 'Chákṣu, per\ncamera\n≥90% sens.', 'fairlearn': 'Fairlearn\nequal sens.\n(calibration)'}
SURFACE, INK, INK2, GRID, GUESS = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']


class Scores(ClassifierMixin, BaseEstimator):
    """Pass-through classifier for Fairlearn: column 0 of X is the fixed model score."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([1 - s, s])


def calibration_mask(cha, frac, rng):
    """Calibration images: frac of each camera's eyes, stratified by label (at least 2 glaucoma eyes, 1 healthy)."""
    eyes = cha.groupby('group').agg(camera=('camera', 'first'), y=('y', 'max'))
    cal = []
    for (_, lab), e in eyes.groupby(['camera', 'y']):
        k = min(len(e), max(2 if lab == 1 else 1, int(round(frac * len(e)))))
        cal += list(rng.choice(e.index.to_numpy(), k, replace=False))
    return cha['group'].isin(cal).to_numpy()


def predict(rule, s, y, cam, cal, te, thr_papila, seed):
    """Test-image predictions (0/1) under one rule, and the thresholds it chose."""
    if rule.startswith('papila_'):
        return (s[te] >= thr_papila[rule]).astype(int), {'all': thr_papila[rule]}
    if rule == 'pooled':
        t = rfg.sensitivity_threshold(y[cal], s[cal], 0.9)
        return (s[te] >= t).astype(int), {'all': t}
    if rule == 'per_camera':
        t = {c: rfg.sensitivity_threshold(y[cal & (cam == c)], s[cal & (cam == c)], 0.9) for c in CAMERAS}
        return (s[te] >= np.array([t[c] for c in cam[te]])).astype(int), t
    to = ThresholdOptimizer(estimator=Scores().fit(s[cal, None], y[cal]), constraints='true_positive_rate_parity',
                            objective='balanced_accuracy_score', prefit=True, predict_method='predict_proba')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        to.fit(s[cal, None], y[cal], sensitive_features=cam[cal])
        return to.predict(s[te, None], sensitive_features=cam[te], random_state=seed).astype(int), {}


def rates(y, yhat):
    tp, fn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 1)).sum())
    fp, tn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 0)).sum())
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    return {'sensitivity': div(tp, tp + fn), 'specificity': div(tn, tn + fp), 'ppv': div(tp, tp + fp)}


def spread(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {'mean': float(x.mean()), 'lo': float(np.percentile(x, 2.5)), 'hi': float(np.percentile(x, 97.5))}


def plot(df, frac, rules, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = df[df['fraction'] == frac]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE, sharey=True)
    for ax, metric, title in [(axes[0], 'sensitivity', 'Glaucoma caught (sensitivity)'),
                              (axes[1], 'specificity', 'Healthy eyes cleared (specificity)')]:
        ax.set_facecolor(SURFACE)
        for i, rule in enumerate(rules):
            for j, (c, color) in enumerate(zip(CAMERAS, SERIES)):
                v = spread(d.loc[d['rule'] == rule, f'{c}_{metric}'])
                x = i + (j - 1) * 0.22
                ax.plot([x, x], [v['lo'], v['hi']], color=color, lw=2, solid_capstyle='round')
                ax.plot(x, v['mean'], 'o', ms=8, color=color, mec=SURFACE, mew=2, zorder=3,
                        label=c if i == 0 else None)
        if metric == 'sensitivity':
            xs = [i for i, r in enumerate(rules) if r in SENS90]
            for i in xs:
                ax.hlines(0.9, i - 0.42, i + 0.42, ls='--', lw=1.2, color=GUESS)
            if xs:
                ax.text(xs[0] - 0.42, 0.91, 'target 0.90', fontsize=7.5, color=GUESS, ha='left', va='bottom')
        n_pap = sum(r.startswith('papila_') for r in rules)
        ax.axvline(n_pap - 0.5, color=GRID, lw=1.2)
        ax.text((n_pap - 1) / 2, 1.06, 'no Chákṣu labels', fontsize=7.5, color=INK2, ha='center')
        ax.text((n_pap + len(rules) - 1) / 2, 1.06, f'thresholds set on {frac:.0%} of eyes', fontsize=7.5, color=INK2,
                ha='center')
        ax.set_xticks(range(len(rules)), [TICKS[r] for r in rules], fontsize=7.5, color=INK)
        ax.set_ylim(-0.02, 1.1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='bold', pad=14)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc='lower left', labelcolor=INK)
    n = d['split'].nunique()
    fig.text(0.01, 0.01, f'Chákṣu, model trained on PAPILA only. Thresholds set on {frac:.0%} of each camera\'s eyes, '
             f'tested on the rest; dot = mean, line = 2.5–97.5 percentile over {n} random splits.\n'
             'Per camera: with under 10 calibration glaucoma images (Bosch, Forus) the rule keeps all of them. '
             'Fairlearn: equal sensitivity on the calibration images only, at the best balanced accuracy.',
             fontsize=7, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=220, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='run_linear_probe.py output folders: same Chakshu scores, different PAPILA threshold rules')
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--fractions', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument('--main-fraction', type=float, default=0.3)
    ap.add_argument('--splits', type=int, default=200)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if not np.isclose(args.fractions, args.main_fraction).any():
        ap.error('--main-fraction must be one of --fractions')
    runs, out = [Path(r) for r in args.runs], Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = runs[0]
    cha = pd.read_csv(run / 'chaksu_predictions.csv')
    thr_papila = {}
    for r in runs:
        res = json.loads((r / 'metrics.json').read_text())
        other = pd.read_csv(r / 'chaksu_predictions.csv').set_index('image_id').loc[cha['image_id'], 'p_glaucoma']
        assert np.allclose(other.to_numpy(), cha['p_glaucoma'].to_numpy()), f'{r.name}: different scores'
        thr_papila[f"papila_{res['config'].get('threshold', 'youden')}"] = float(res['papila_full_fit_threshold'])
    RULES = {**{k: PAPILA_RULES[k.split('_', 1)[1]] for k in thr_papila}, **CALIBRATED}
    cha['group'] = rfg.chaksu_external(args.chaksu_labels).set_index('image_id').loc[cha['image_id'], 'group'].to_numpy()
    s, y, cam = cha['p_glaucoma'].to_numpy(), cha['y'].to_numpy(), cha['camera'].to_numpy()

    rows = []
    for frac in args.fractions:
        for r in range(args.splits):
            cal = calibration_mask(cha, frac, np.random.default_rng([round(frac * 1000), r]))
            te = ~cal
            for rule in RULES:
                yhat, thr = predict(rule, s, y, cam, cal, te, thr_papila, seed=r)
                row = {'fraction': frac, 'split': r, 'rule': rule, 'n_test': int(te.sum()),
                       **{f'{c}_cal_glaucoma': int(y[cal & (cam == c)].sum()) for c in CAMERAS},
                       **{f'thr_{k}': v for k, v in thr.items()}}
                for c in CAMERAS + ['All']:
                    m = np.ones(te.sum(), bool) if c == 'All' else cam[te] == c
                    row.update({f'{c}_{k}': v for k, v in rates(y[te][m], yhat[m]).items()})
                for k in ['sensitivity', 'specificity']:
                    v = [row[f'{c}_{k}'] for c in CAMERAS]
                    row[f'{k}_gap_per_split'] = max(v) - min(v)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'splits.csv', index=False)
    cols = [f'{c}_{k}' for c in CAMERAS + ['All'] for k in ['sensitivity', 'specificity', 'ppv']]
    cols += ['sensitivity_gap_per_split', 'specificity_gap_per_split'] + [f'{c}_cal_glaucoma' for c in CAMERAS]
    summ = {f'{f:g}': {rule: {k: spread(g[k]) for k in cols} for rule, g in d.groupby('rule')}
            for f, d in df.groupby('fraction')}
    for by_rule in summ.values():
        for s_ in by_rule.values():
            for k in ['sensitivity', 'specificity']:
                means = [s_[f'{c}_{k}']['mean'] for c in CAMERAS]
                s_[f'{k}_gap'] = max(means) - min(means)
    (out / 'calibration.json').write_text(json.dumps({'runs': [r.name for r in runs], 'papila_thresholds': thr_papila,
                                                       'splits': args.splits, 'by_fraction': summ}, indent=2))
    plot(df, args.main_fraction, RULES, out / 'threshold_calibration.png')

    m = summ[f'{args.main_fraction:g}']
    cell = lambda rule, k: f"{m[rule][k]['mean']:.2f} ({m[rule][k]['lo']:.2f}–{m[rule][k]['hi']:.2f})"  # noqa: E731
    cal_n = ', '.join(f"{c} {m['per_camera'][f'{c}_cal_glaucoma']['mean']:.0f}" for c in CAMERAS)
    L = [f'# Chákṣu thresholds set on labelled images per camera (`{run.name}` scores)\n\n',
         f'Calibration on {args.main_fraction:.0%} of each camera\'s eyes (glaucoma images in it: {cal_n}); '
         f'tested on the rest. Mean (2.5–97.5 percentile) over {args.splits} random splits. Gap = largest minus '
         'smallest camera mean. Per camera: under 10 calibration glaucoma images means all are kept. Fairlearn: '
         'equal sensitivity on the calibration images, at the best balanced accuracy. PAPILA thresholds: '
         + ', '.join(f'{RULES[k]} = {v:.3f}' for k, v in thr_papila.items()) + '.\n\n',
         '| Rule | ' + ' | '.join(f'{c} sensitivity' for c in CAMERAS) + ' | Gap between camera means |\n',
         '|---|' + '---|' * (len(CAMERAS) + 1) + '\n']
    for rule, label in RULES.items():
        L.append(f'| {label} | ' + ' | '.join(cell(rule, f'{c}_sensitivity') for c in CAMERAS)
                 + f" | {m[rule]['sensitivity_gap']:.2f} |\n")
    L += ['\n| Rule | ' + ' | '.join(f'{c} specificity' for c in CAMERAS) + ' | All: sensitivity | All: specificity '
          '| All: PPV |\n', '|---|' + '---|' * (len(CAMERAS) + 3) + '\n']
    for rule, label in RULES.items():
        L.append(f'| {label} | ' + ' | '.join(cell(rule, f'{c}_specificity') for c in CAMERAS)
                 + f" | {cell(rule, 'All_sensitivity')} | {cell(rule, 'All_specificity')} | {cell(rule, 'All_ppv')} |\n")
    L += ['\n## Size of the calibration set (threshold per camera)\n\n',
          '| Calibration share | Glaucoma images per camera (R / B / F) | Gap between camera means | Bosch sensitivity '
          '| Forus sensitivity | All: specificity |\n', '|---|---|---|---|---|---|\n']
    for f in args.fractions:
        p = summ[f'{f:g}']['per_camera']
        c2 = lambda k: f"{p[k]['mean']:.2f} ({p[k]['lo']:.2f}–{p[k]['hi']:.2f})"  # noqa: E731
        L.append(f"| {f:.0%} | {' / '.join(str(round(p[f'{c}_cal_glaucoma']['mean'])) for c in CAMERAS)} | "
                 f"{p['sensitivity_gap']:.2f} | {c2('Bosch_sensitivity')} | {c2('Forus_sensitivity')} | "
                 f"{c2('All_specificity')} |\n")
    (out / 'summary.md').write_text(''.join(L))
    print(''.join(L))


if __name__ == '__main__':
    main()
