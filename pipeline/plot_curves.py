"""ROC and precision-recall curves for one step-1 result folder (run_linear_probe.py output).

Top row: PAPILA out-of-fold (image features vs the age + sex baseline). Bottom row: Chakshu by camera.
Dots mark the operating point actually used (each fold's training-set threshold; for Chakshu the threshold
frozen from PAPILA). Dashed lines show what random guessing scores.

Usage:
    python plot_curves.py results/frozen_lr_tuned_resize
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rfg  # noqa: E402

SURFACE, INK, INK2, GRID, GUESS = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']  # categorical slots 1-3 (validated all-pairs)


def roc_pr(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    fpr, tpr = rfg._roc_points(y, p)
    order = np.argsort(-p, kind='mergesort')
    ys, ps = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp = np.cumsum(ys)[last]
    prec, rec = tp / (last + 1), tp / ys.sum()
    return fpr, tpr, np.r_[0, rec], np.r_[prec[0], prec]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect('equal')


def panel(ax_roc, ax_pr, series, title):
    ax_roc.plot([0, 1], [0, 1], ls='--', lw=1.2, color=GUESS, label='Guessing (0.50)')
    for i, (name, y, p, m) in enumerate(series):
        fpr, tpr, rec, prec = roc_pr(y, p)
        shown = rfg.metrics(y, p)  # areas of the curves as drawn
        c = SERIES[i]
        ax_roc.plot(fpr, tpr, lw=2, color=c, label=f"{name} (AUROC {shown['auroc']:.2f})")
        ax_roc.plot(1 - m['specificity'], m['sensitivity'], 'o', ms=8, color=c, mec=SURFACE, mew=2)
        ax_pr.plot(rec, prec, lw=2, color=c, label=f"{name} (AUPRC {shown['auprc']:.2f}; guessing {m['prevalence']:.2f})")
        ax_pr.axhline(m['prevalence'], ls='--', lw=1.2, color=c, alpha=0.6)
        ax_pr.plot(m['sensitivity'], m['precision'], 'o', ms=8, color=c, mec=SURFACE, mew=2)
    style(ax_roc, f'{title}: ROC', 'False alarms (1 − specificity)', 'Glaucoma caught (sensitivity)')
    style(ax_pr, f'{title}: precision-recall', 'Glaucoma caught (recall)', 'Alarms that are right (precision)')
    for ax in (ax_roc, ax_pr):
        ax.legend(fontsize=7.5, frameon=False, loc='lower right' if ax is ax_roc else 'upper right',
                  labelcolor=INK)


def main(folder):
    folder = Path(folder)
    res = json.loads((folder / 'metrics.json').read_text())
    pap = pd.read_csv(folder / 'papila_oof_predictions.csv')
    cha = pd.read_csv(folder / 'chaksu_predictions.csv')
    fold = pap['fold'].to_numpy()
    # pooled curves use scores ranked within each fold (the folds' models differ in scale)
    pap_series = [(lab, pap['y'], rfg.rank_within_fold(pap[f'p_{key}'], fold), res['papila'][key])
                  for lab, key in [('Image features', 'image'), ('Age + sex only', 'clinical (age, sex)')]]
    cam_series = [(cam, d['y'], d['p_glaucoma'], res['chaksu'][cam])
                  for cam, d in sorted(cha.groupby('camera'), key=lambda t: -len(t[1]))]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.4), facecolor=SURFACE)
    panel(axes[0, 0], axes[0, 1], pap_series, 'PAPILA')
    panel(axes[1, 0], axes[1, 1], cam_series, 'Chákṣu')
    proto = res['config'].get('protocol', 'tuned')
    fig.suptitle(f'RETFound-Green, frozen + logistic regression ({proto})', x=0.02, ha='left',
                 fontsize=11, color=INK, fontweight='bold')
    fig.text(0.02, 0.006, 'PAPILA: 5 folds pooled (scores ranked within each fold). Chákṣu: by camera, no retraining.\n'
             'Dots: operating point used. Dashed: random guessing (on PR, the share of glaucoma images).',
             fontsize=7.5, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = folder / 'curves.png'
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == '__main__':
    main(sys.argv[1])
