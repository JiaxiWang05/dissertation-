"""Figures for the 10 Nov work, drawn from the saved results (nothing is trained here). The test part is not used.
python make_figures.py   ->  figures/*.png   (a figure whose input file is missing is skipped)"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path('figures')
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
NAMES = {'logreg (8 Nov, AUROC-tuned)': 'Logistic regression\n(8 Nov, AUROC-tuned)', 'logreg': 'Logistic regression',
         'svm': 'SVM', 'knn': 'kNN', 'mlp_small': 'Small MLP'}


def disc_side():
    lab = pd.read_csv('../pipeline/chaksu_labels.csv')
    box = pd.read_csv('disc_boxes.csv')
    box = box[box['dataset'] == 'Chaksu']
    d = lab[['image_id', 'camera']].merge(box[['image_id', 'cx', 'width']], on='image_id')
    fig, axs = plt.subplots(1, 3, figsize=(11, 3), sharey=False)
    for ax, cam in zip(axs, ['Remidio', 'Bosch', 'Forus']):
        x = (d.loc[d['camera'] == cam, 'cx'] / d.loc[d['camera'] == cam, 'width'])
        ax.hist(x, bins=40, range=(0, 1), color='tab:blue')
        ax.axvspan(0.45, 0.55, color='0.9', zorder=0)
        ax.set_title(f'{cam} ({len(x)} images)', fontsize=10)
        ax.set_xlabel('optic-disc centre / image width')
    axs[0].set_ylabel('images')
    fig.suptitle('Where the optic disc sits: two groups (left / right eye) on Remidio, one central group on Bosch and Forus',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_disc_position.png', dpi=200)
    plt.close(fig)


def learning_curve():
    h = pd.read_csv('mlp_small_learning_curve.csv')
    best = int(h.loc[h['val_loss'].idxmin(), 'epoch'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    a.plot(h['epoch'], h['train_loss'], label='training loss')
    a.plot(h['epoch'], h['val_loss'], label='validation loss')
    a.axvline(best, ls='--', c='gray', label=f'lowest validation loss (epoch {best})')
    a.set_xlabel('epoch'); a.set_ylabel('loss'); a.legend(frameon=False)
    b.plot(h['epoch'], h['val_auprc'], c='tab:red', label='validation AUPRC')
    b.plot(h['epoch'], h['val_auroc'], c='tab:green', label='validation AUROC')
    b.axhline(0.14, ls=':', c='gray', label='AUPRC of guessing (0.14)')
    b.axvline(best, ls='--', c='gray')
    b.set_xlabel('epoch'); b.legend(frameon=False, fontsize=8)
    s = json.load(open('step5_summary.json'))
    hp = s['best hyperparameters']['mlp_small']
    fig.suptitle(f'Small MLP ({hp["width1"]} and {hp["width2"]} neurons, {s["mlp_small trainable parameters"]:,} parameters): '
                 'all 300 epochs on the fixed training (939) / validation (202) split', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_small_mlp_learning_curve.png', dpi=200)
    plt.close(fig)


def comparison():
    c = pd.read_csv('step5_comparison.csv')
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ys = np.arange(len(c))[::-1]
    for yv, (_, r) in zip(ys, c.iterrows()):
        col = 'tab:red' if r['tied with best'] else 'tab:blue'
        ax.errorbar(r['AUPRC'], yv, xerr=[[r['AUPRC'] - r['AUPRC CI low']], [r['AUPRC CI high'] - r['AUPRC']]],
                    fmt='o', color=col, capsize=4)
        ax.text(0.62, yv, f'{r["AUPRC"]:.3f} ({r["AUPRC CI low"]:.3f}–{r["AUPRC CI high"]:.3f})   AUROC {r["AUROC"]:.3f}'
                + ('   tied with best' if r['tied with best'] else ''), va='center', fontsize=8.5)
    ax.axvline(0.14, ls=':', c='gray')
    ax.text(0.141, ys[0] + 0.45, 'guessing 0.14', fontsize=8, color='gray')
    ax.set_yticks(ys, [NAMES[n] for n in c['classifier']])
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel('out-of-fold AUPRC with 95 % CI (2,000 resamples of the image groups); red = tied with the best')
    ax.set_title('Classifier comparison by AUPRC, 5-fold cross-validation (1,141 images, 160 glaucoma)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_comparison_auprc.png', dpi=200)
    plt.close(fig)


def confusion():
    c = pd.read_csv('step5_confusion.csv')
    for rule, tag in [('90% sensitivity', 'sens90'), ("Youden's J", 'youden')]:
        d = c[c['threshold rule'] == rule]
        fig, axs = plt.subplots(1, len(d), figsize=(3.1 * len(d), 3.2))
        for ax, (_, r) in zip(axs, d.iterrows()):
            m = np.array([[r['TP'], r['FN']], [r['FP'], r['TN']]])
            ax.imshow(m / m.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(m):
                ax.text(j, i, f'{v}', ha='center', va='center', fontsize=12)
            ax.set_xticks([0, 1], ['called\nglaucoma', 'called\nhealthy'])
            ax.set_yticks([0, 1], ['glaucoma\n(160)', 'healthy\n(981)'])
            ax.set_title(f'{NAMES[r["classifier"]]}\nsens {r["sensitivity"]:.2f}, spec {r["specificity"]:.2f}, F1 {r["F1"]:.2f}',
                         fontsize=8.5)
        fig.suptitle(f'Out-of-fold confusion matrices at the {rule} threshold (1,141 training + validation images)',
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f'fig4_confusion_{tag}.png', dpi=200)
        plt.close(fig)


def mlp_search():
    import optuna
    from common import mlp_small_storage
    d = optuna.load_study(study_name='mlp_small', storage=mlp_small_storage()).trials_dataframe()
    d = d[d['state'] == 'COMPLETE']
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    combos = sorted(d.groupby(['params_width1', 'params_width2']).groups)
    a.boxplot([d.loc[(d['params_width1'] == w1) & (d['params_width2'] == w2), 'value'] for w1, w2 in combos])
    a.set_xticks(range(1, len(combos) + 1), [f'{int(w1)}-{int(w2)}' for w1, w2 in combos], rotation=60, fontsize=8)
    a.set_xlabel('neurons in layer 1 - layer 2'); a.set_ylabel('5-fold mean validation AUPRC')
    a.set_title(f'Small MLP search: {len(d)} trials', fontsize=10)
    losses = ['bce', 'weighted_bce', 'focal']
    b.boxplot([d.loc[d['params_loss'] == l, 'value'] for l in losses])
    b.set_xticks([1, 2, 3], [f'{n}\n({(d["params_loss"] == l).sum()} trials)' for n, l in
                             zip(['plain cross-entropy', 'weighted cross-entropy', 'focal (weighted)'], losses)])
    b.set_title('By loss function', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_small_mlp_search.png', dpi=200)
    plt.close(fig)


def losses():
    df = pd.read_csv('check_losses.csv')
    per_seed = df.groupby(['loss', 'seed'])['auprc'].mean().reset_index()
    order = per_seed.groupby('loss')['auprc'].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, l in enumerate(order):
        v = per_seed.loc[per_seed['loss'] == l, 'auprc']
        ax.plot([v.min(), v.max()], [i, i], c='tab:blue')
        ax.plot(v.mean(), i, 'o', c='tab:blue')
        ax.text(v.max() + 0.003, i, f'{v.mean():.3f}', va='center', fontsize=8.5)
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    ax.set_xlabel('kept-model validation AUPRC, mean of 5 folds (dot = mean of 5 seeds, line = range over seeds)')
    ax.set_title('Best small MLP trained with different loss functions', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_loss_functions.png', dpi=200)
    plt.close(fig)


for f in [disc_side, learning_curve, comparison, confusion, mlp_search, losses]:
    try:
        f()
        print('made', f.__name__)
    except (FileNotFoundError, KeyError) as e:
        print('skipped', f.__name__, '(input not ready:', e, ')')
