"""Figures for the meeting report, drawn from the saved real-data results (no model is trained or scored here).
   py make_figures.py          -> figures/fig1..fig4 (block diagram, Optuna searches, learning curve, comparison)
   py make_figures.py test     -> also figures/fig5 (test ROC and confusion matrices, from test_scores.csv)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve

from common import STORAGE

optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = Path('figures')
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
NAMES = {'logreg': 'Logistic regression', 'knn': 'kNN', 'svm': 'SVM', 'mlp': 'MLP'}


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=color, ec='#444', lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))


def fig1_block_diagram():
    fig, ax = plt.subplots(figsize=(16, 5.6))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 5.6)
    ax.axis('off')
    steps = [('Fundus photo\nChákṣu\n1,338 images\n3 cameras', '#e8eef7'),
             ('Resize to\n392 × 392 pixels\n(the size the model\nwas trained on)', '#e8eef7'),
             ('RETFound-Green\nViT-Small\n22 M parameters\nFROZEN', '#dfe9df'),
             ('384 features\nper image', '#dfe9df'),
             ('Scale features\n(fitted on the\ntraining part only)', '#f4ecdc'),
             ('Classifier\nlogistic regression,\nkNN, SVM or MLP', '#f4ecdc'),
             ('Score\n→ threshold\n(set on validation)', '#f4ecdc'),
             ('Glaucoma\nor healthy', '#f6e0e0')]
    w, h, gap, y = 1.75, 1.6, 0.28, 3.6
    for i, (t, c) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        box(ax, x, y, w, h, t, c)
        if i:
            arrow(ax, x - gap, y + h / 2, x, y + h / 2)
    ax.text(0.1, 3.15, 'Fixed: never trained', fontsize=9, color='#2f6b2f')
    ax.text(0.1 + 4 * (w + gap), 3.15, 'Learned from the training images; hyperparameters chosen by Optuna on validation',
            fontsize=9, color='#8a5a00')
    # the split
    parts = [('Data split, made once and saved\nstratified: 14% glaucoma in each part\n'
              'grouped by eye: no eye in two parts\n(chaksu_split.csv, 28 Oct)', '#eeeeee'),
             ('Training 60%\n802 images (111 glaucoma)\nfits the classifier', '#f4ecdc'),
             ('Validation 20%\n268 images (38 glaucoma)\nchooses hyperparameters,\nclassifier and threshold', '#f4ecdc'),
             ('Test 20%\n268 images (38 glaucoma)\nused ONCE, at the end', '#f6e0e0')]
    w, gap = 3.55, 0.55
    for i, (t, c) in enumerate(parts):
        x = 0.1 + i * (w + gap)
        box(ax, x, 0.5, w, 1.9, t, c)
        if i:
            arrow(ax, x - gap, 1.45, x, 1.45)
    fig.savefig(OUT / 'fig1_block_diagram.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def trials(name):
    return optuna.load_study(study_name=name, storage=STORAGE).trials_dataframe()


def fig2_optuna():
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    d = trials('logreg')
    ax = axs[0, 0]
    ax.scatter(d.params_C, d.value, s=18, color='tab:blue')
    b = d.loc[d.value.idxmax()]
    ax.scatter([b.params_C], [b.value], s=90, facecolors='none', edgecolors='k', lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('C (smaller = stronger penalty)')
    ax.set_ylabel('validation AUROC')
    ax.set_title(f'(a) Logistic regression, {len(d)} trials: best C = {b.params_C:.4f}, AUROC {b.value:.3f}', fontsize=9.5)

    d = trials('svm')
    ax = axs[0, 1]
    for k, c in [('linear', 'tab:blue'), ('rbf', 'tab:orange')]:
        e = d[d.params_kernel == k]
        ax.scatter(e.params_C, e.value, s=18, color=c, label=f'{k} ({len(e)} trials)')
    b = d.loc[d.value.idxmax()]
    ax.scatter([b.params_C], [b.value], s=90, facecolors='none', edgecolors='k', lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('C (smaller = wider margin, more mistakes allowed)')
    ax.set_ylabel('validation AUROC')
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f'(b) SVM, {len(d)} trials: best {b.params_kernel}, C = {b.params_C:.2g}, AUROC {b.value:.3f}',
                 fontsize=9.5)

    d = trials('knn')
    ax = axs[1, 0]
    for (wt, met), e in d.groupby(['params_weights', 'params_metric']):
        e = e.sort_values('params_k')
        ax.plot(e.params_k, e.value, marker='.', ms=4, lw=1, label=f'{wt}, {met}')
    b = d.loc[d.value.idxmax()]
    ax.scatter([b.params_k], [b.value], s=90, facecolors='none', edgecolors='k', lw=1.5)
    ax.set_xlabel('k (number of neighbours that vote)')
    ax.set_ylabel('validation AUROC')
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f'(c) kNN, all {len(d)} settings: best k = {int(b.params_k)}, {b.params_weights}, '
                 f'{b.params_metric}, AUROC {b.value:.3f}', fontsize=9.5)

    d = trials('mlp')
    ax = axs[1, 1]
    ax.scatter(d.number + 1, d.value, s=14, color='tab:green', label='one trial')
    ax.plot(d.number + 1, d.value.cummax(), color='k', lw=1.2, label='best so far')
    b = d.loc[d.value.idxmax()]
    ax.set_xlabel('trial number (Optuna chooses each trial from the earlier ones)')
    ax.set_ylabel('validation AUROC')
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.set_title(f'(d) MLP, {len(d)} trials: best {int(b.params_n_layers)} × {int(b.params_width)} neurons, '
                 f'AUROC {b.value:.3f}', fontsize=9.5)
    fig.suptitle('Optuna searches: every trial trains on the 802 training images and is scored on the 268 validation images',
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_optuna.png', dpi=200)
    plt.close(fig)


def fig3_learning_curve():
    h = pd.read_csv('mlp_learning_curve.csv')
    best = int(h.val_loss.idxmin()) + 1
    p = optuna.load_study(study_name='mlp', storage=STORAGE).best_params
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(h.epoch, h.train_loss, label='training loss')
    a1.plot(h.epoch, h.val_loss, label='validation loss')
    a1.axvline(best, color='grey', ls='--', label=f'lowest validation loss (epoch {best})')
    a1.set_xlabel('epoch')
    a1.set_ylabel('loss (binary cross-entropy)')
    a1.legend(frameon=False)
    a2.plot(h.epoch, h.val_auroc, color='tab:orange')
    a2.axvline(best, color='grey', ls='--')
    a2.set_xlabel('epoch')
    a2.set_ylabel('validation AUROC')
    fig.suptitle(f"MLP learning curves, best Optuna setting: {p['n_layers']} hidden layers × {p['width']} neurons, "
                 f"dropout {p['dropout']:.2f}, learning rate {p['lr']:.1e}, weight decay {p['weight_decay']:.1e}; "
                 f"trained all 200 epochs to show the whole curve", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_learning_curve.png', dpi=200)
    plt.close(fig)


def fig4_comparison():
    t = pd.read_csv('step5_comparison.csv')
    chosen = json.load(open('chosen.json'))['classifier']
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, r in t.iterrows():
        c = 'tab:red' if r.classifier == chosen else 'tab:blue'
        ax.errorbar(r['val AUROC'], i, xerr=[[r['val AUROC'] - r['AUROC CI low']], [r['AUROC CI high'] - r['val AUROC']]],
                    fmt='o', color=c, capsize=4)
        tie = 'best' if r['best minus this'] == 0 else ('tied with best' if r['tied with best'] else 'below best')
        ax.text(0.935, i, f"{r['val AUROC']:.3f}  ({r['AUROC CI low']:.3f}–{r['AUROC CI high']:.3f})   {tie}",
                va='center', fontsize=8.5)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([NAMES[n] + (' (chosen)' if n == chosen else '') for n in t.classifier])
    ax.set_xlim(0.68, 1.17)
    ax.set_xticks(np.arange(0.70, 0.95, 0.05))
    ax.set_xlabel('validation AUROC with 95% CI (2,000 bootstrap resamples of eyes)')
    ax.set_title('Classifier comparison on the 268 validation images (38 glaucoma)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_comparison.png', dpi=200)
    plt.close(fig)


def fig5_test():
    s = pd.read_csv('test_scores.csv')
    res = json.load(open('test_results.json'))
    info = json.load(open('chosen.json'))
    y, p = s.label_majority.to_numpy(), s.score.to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={'width_ratios': [1.3, 1, 1]})
    ax = axs[0]
    ax.plot(fpr, tpr, color='tab:red', label=f"test AUROC {res['AUROC']:.3f} "
            f"(95% CI {res['AUROC 95% CI'][0]:.3f}–{res['AUROC 95% CI'][1]:.3f})")
    ax.plot([0, 1], [0, 1], color='grey', ls=':', label='guessing (0.5)')
    for key, lab, m in [('at 90% recall threshold (from validation)', '90%-recall threshold', 'o'),
                        ("at Youden's J threshold (from validation)", "Youden's J threshold", 's')]:
        r = res[key]
        ax.plot(1 - r['specificity'], r['sensitivity'], m, color='k', ms=7, mfc='white', label=lab)
    ax.set_xlabel('1 − specificity (false-alarm rate)')
    ax.set_ylabel('sensitivity (share of glaucoma found)')
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.set_title(f"ROC curve, {NAMES[info['classifier']]}, 268 test images", fontsize=10)
    for ax, key, title in [(axs[1], 'at 90% recall threshold (from validation)', '90%-recall threshold'),
                           (axs[2], "at Youden's J threshold (from validation)", "Youden's J threshold")]:
        r = res[key]
        m = np.array([[r['TP'], r['FN']], [r['FP'], r['TN']]])
        ax.imshow(m, cmap='Blues', vmin=0, vmax=m.max() * 1.6)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, str(v), ha='center', va='center', fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['called glaucoma', 'called healthy'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['glaucoma (38)', 'healthy (230)'])
        ax.set_title(f"{title}\nsensitivity {r['sensitivity']:.2f}, specificity {r['specificity']:.2f}", fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_test.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    fig1_block_diagram()
    fig2_optuna()
    fig3_learning_curve()
    fig4_comparison()
    if 'test' in sys.argv[1:]:
        fig5_test()
    print('saved', sorted(p.name for p in OUT.glob('*.png')))
