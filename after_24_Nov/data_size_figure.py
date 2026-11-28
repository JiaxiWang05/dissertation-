"""Figures for data_size.py: figures/data_size.png (validation AUPRC / AUROC against the number of training images, and
training vs validation AUPRC per model) and figures/data_size_protocol.png (how the subsets are made); data_size.csv.
Input: data_size/<model>_<frac>.json.   python data_size_figure.py   (DEMO=1 draws made-up numbers to check the layout)"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent; IN = HERE / 'data_size'; OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
TEXT, MUTED = '#1a1a1a', '#666666'
MODELS = [('svm', 'Frozen features + SVM', '#7a7a7a'), ('cnn', 'Small CNN from scratch', '#b08d2a'),
          ('blocks2', '2 open blocks', '#2f6b9a'), ('lora', 'LoRA', '#c0583a'), ('mask025', 'LoRA + disc-mask head', '#2e8b57'),
          ('lora_blocks', 'LoRA + 2 open blocks', '#7b4fa0')]
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def load():
    rows = []
    for m, _, _ in MODELS:
        for f in FRACS:
            p = IN / f'{m}_{f}.json'
            if os.environ.get('DEMO'):
                top = {'svm': .43, 'cnn': .16, 'blocks2': .50, 'lora': .53, 'mask025': .58, 'lora_blocks': .57}[m]
                d = {'repeats': [{'n_train': [int(913 * f)] * 5, 'val': {'auprc': top - (0.25 if m != 'cnn' else .01) * (1 - f) ** 1.5 + .01 * r,
                                  'auroc': .5 + (top - .1) * (0.6 + 0.4 * f)}, 'train': {'auprc': 1 - .3 * f * (m != 'svm'), 'auroc': .99}}
                                 for r in range(3)], 'prevalence_val': 0.14}
            elif p.exists():
                d = json.load(open(p))
            else:
                continue
            for r in d['repeats']:
                rows.append({'model': m, 'frac': f, 'repeat': r['repeat'] if 'repeat' in r else 0, 'n_train': np.mean(r['n_train']),
                             'val_auprc': r['val']['auprc'], 'val_auroc': r['val']['auroc'],
                             'train_auprc': r['train']['auprc'], 'train_auroc': r['train']['auroc'], 'prev': d['prevalence_val'],
                             'valf_auprc': r.get('val_fold_mean', r['val'])['auprc'], 'seconds': r.get('seconds', np.nan)})
    return pd.DataFrame(rows)


def curves(df):
    g = df.groupby(['model', 'frac'])
    s = g.agg(n=('n_train', 'mean'), val=('val_auprc', 'mean'), vlo=('val_auprc', 'min'), vhi=('val_auprc', 'max'),
              roc=('val_auroc', 'mean'), rlo=('val_auroc', 'min'), rhi=('val_auroc', 'max'),
              tr=('train_auprc', 'mean'), tlo=('train_auprc', 'min'), thi=('train_auprc', 'max'),
              vf=('valf_auprc', 'mean'), vflo=('valf_auprc', 'min'), vfhi=('valf_auprc', 'max'), sec=('seconds', 'mean')).reset_index()
    s.to_csv(HERE / ('data_size_demo.csv' if os.environ.get('DEMO') else 'data_size.csv'), index=False)
    return s


def main_figure(s, prev):
    have = [mm for mm in MODELS if (s['model'] == mm[0]).any()]
    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1], hspace=0.42, wspace=0.55)
    axA = fig.add_subplot(gs[0, :4]); axB = fig.add_subplot(gs[0, 4:])
    for ax, key, lo, hi, ylab, base in ((axA, 'val', 'vlo', 'vhi', 'validation AUPRC (pooled out-of-fold)', prev),
                                         (axB, 'roc', 'rlo', 'rhi', 'validation AUROC', 0.5)):
        for m, name, c in have:
            d = s[s['model'] == m].sort_values('n')
            ax.fill_between(d['n'], d[lo], d[hi], color=c, alpha=0.15, lw=0)
            ax.plot(d['n'], d[key], '-o', color=c, lw=2.2, ms=5.5, mec='white', mew=0.8)
        if ax is axA:                                        # end labels, pushed apart so they never overlap
            ends = []
            for m, name, c in have:
                d = s[s['model'] == m].sort_values('n'); last = d.iloc[-1]
                dl = f" ({last[key] - d.iloc[-2][key]:+.2f})" if len(d) > 1 else ''
                ends.append([last[key], last[key], f"{name}  {last[key]:.2f}{dl}", c, last['n']])
            ends.sort(key=lambda e: e[0])
            for i in range(1, len(ends)):
                ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.035)
            for v, yl, txt, c, n in ends:
                ax.annotate(txt, xy=(n, v), xytext=(n * 1.08, yl), fontsize=9, color=c, fontweight='bold', va='center',
                            arrowprops=dict(arrowstyle='-', color=c, lw=0.6))
        ax.axhline(base, color='#999999', ls=':', lw=1.2)
        ax.text(ax.get_xlim()[0] if False else 95, base + 0.012, 'guessing', color=MUTED, fontsize=8.5)
        ax.set_xscale('log'); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([90, 230, 460, 680, 910]); ax.set_xticklabels(['90', '230', '460', '680', '910'])
        ax.set_xlim(80, 960 if ax is axB else 3600); ax.set_ylim(0 if ax is axA else 0.4, 0.8 if ax is axA else 1.0)
        ax.set_xlabel('training images per fold (log scale; 10 / 25 / 50 / 75 / 100 % of the fold)'); ax.set_ylabel(ylab)
        ax.grid(axis='y', color='#eeeeee', lw=0.8)
    axA.spines['bottom'].set_bounds(80, 920)
    axA.set_title('A   More training images: how much better does each model get?', loc='left', fontsize=12, fontweight='bold')
    axA.text(0.0, 0.985, 'end label: AUPRC with all training images (change from 75 % to 100 %)', transform=axA.transAxes, fontsize=8.5, color=MUTED, va='top')
    axB.set_title('B   Same for AUROC', loc='left', fontsize=12, fontweight='bold')
    for j, (m, name, c) in enumerate(have):
        ax = fig.add_subplot(gs[1, j]); d = s[s['model'] == m].sort_values('n')
        ax.fill_between(d['n'], d['vf'], d['tr'], color=c, alpha=0.12, lw=0)
        ax.fill_between(d['n'], d['tlo'], d['thi'], color='#2f6b9a', alpha=0.15, lw=0)
        ax.plot(d['n'], d['tr'], '-o', color='#2f6b9a', lw=1.8, ms=4, label='training images')
        ax.fill_between(d['n'], d['vflo'], d['vfhi'], color=c, alpha=0.2, lw=0)
        ax.plot(d['n'], d['vf'], '-o', color=c, lw=1.8, ms=4, label='validation images')
        last = d.iloc[-1]
        ax.annotate('', xy=(last['n'] * 1.12, last['vf']), xytext=(last['n'] * 1.12, last['tr']),
                    arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.9))
        ax.text(last['n'] * 1.2, (last['vf'] + last['tr']) / 2, f"gap\n{last['tr'] - last['vf']:.2f}", fontsize=8, color=MUTED, va='center')
        ax.axhline(prev, color='#999999', ls=':', lw=1)
        ax.set_xscale('log'); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([90, 460, 910]); ax.set_xticklabels(['90', '460', '910']); ax.set_xlim(75, 1400)
        ax.set_ylim(0, 1.05); ax.set_title(name, fontsize=10, color=c, fontweight='bold'); ax.set_xlabel('training images')
        if j == 0:
            ax.set_ylabel('AUPRC'); ax.legend(frameon=False, fontsize=8, loc='lower left')
            ax.text(0, 1.22, 'C   Training vs validation AUPRC: the shaded gap is overfitting; it should shrink as images are added',
                    transform=ax.transAxes, fontsize=12, fontweight='bold')
    fig.text(0.5, 0.005, 'Each point: 5 folds x 3 repeats (different random subsets and seeds); line = mean, band = lowest to highest repeat. '
             'The validation fold is never reduced, so all points of a model are scored on the same images. Epochs chosen by the fair rule '
             '(median best epoch of the other folds), with the same number of optimiser steps at every size. Panel C averages training and validation AUPRC '
             'per fold in the same way. Settings were tuned at 100 %, so that point is slightly optimistic.', ha='center', fontsize=8.5, color=MUTED, wrap=True)
    name = 'data_size_demo.png' if os.environ.get('DEMO') else 'data_size.png'
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight'); plt.close(fig); print('saved', name)


def protocol_figure():
    fig, ax = plt.subplots(figsize=(12, 4.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis('off')
    # one fold: training part + validation part
    ax.text(0.1, 4.25, 'One of the 5 folds', fontsize=11, fontweight='bold')
    ax.add_patch(Rectangle((0.1, 3.2), 4.0, 0.7, fc='#dce8f2', ec='#2f6b9a')); ax.text(2.1, 3.55, 'training part (about 913 images)', ha='center', va='center')
    ax.add_patch(Rectangle((4.2, 3.2), 1.0, 0.7, fc='#f6dccf', ec='#c0583a')); ax.text(4.7, 3.55, 'validation\n(228)', ha='center', va='center', fontsize=8.5)
    ax.text(0.1, 2.75, 'Training part reduced, the validation part never:', fontsize=9.5, color=MUTED)
    for i, f in enumerate(FRACS):
        yy = 2.25 - i * 0.48
        ax.add_patch(Rectangle((0.1, yy), 4.0, 0.36, fc='white', ec='#cccccc', lw=0.6))
        ax.add_patch(Rectangle((0.1, yy), 4.0 * f, 0.36, fc='#2f6b9a', alpha=0.25 + 0.6 * f, ec='none'))
        ax.text(4.25, yy + 0.18, f'{int(f * 100)} %  ~{int(round(913 * f))} images', va='center', fontsize=9)
        ax.add_patch(Rectangle((6.0, yy), 0.5, 0.36, fc='#f6dccf', ec='#c0583a', lw=0.6))
    ax.text(6.25, 2.75, 'same', ha='center', fontsize=8.5, color='#c0583a')
    rules = ['Whole image groups only (a Bosch pair or a copy is never split)',
             'Drawn within each label x camera stratum: glaucoma share and camera mix stay the same',
             'Nested: the 10 % subset sits inside the 25 % subset, and so on',
             '3 repeats: a new random group order and a new training seed each time',
             'Each model keeps its own setting; epochs by the fair rule (median best epoch of the other folds)',
             'Measured: validation AUPRC / AUROC, and the same models scored on their own training images']
    ax.text(7.0, 4.25, 'Rules', fontsize=11, fontweight='bold')
    for i, r in enumerate(rules):
        ax.text(7.0, 3.75 - i * 0.6, f'{i + 1}', fontsize=10, fontweight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.25', fc='#2f6b9a', ec='none'))
        ax.text(7.3, 3.75 - i * 0.6, r, fontsize=9, va='center', wrap=True)
    fig.savefig(OUT / 'data_size_protocol.png', dpi=170, bbox_inches='tight'); plt.close(fig); print('saved protocol')


if __name__ == '__main__':
    df = load()
    if len(df):
        main_figure(curves(df), float(df['prev'].iloc[0]))
    protocol_figure()
