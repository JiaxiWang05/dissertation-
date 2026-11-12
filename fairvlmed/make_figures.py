"""Figures for the FairVLMed results, drawn from the saved summary tables (nothing is trained here, and no per-patient
data is needed). The 2,000 test patients are not used.
PREPROC=raw python make_figures.py   ->  figures/*.png   (a figure whose input file is missing is skipped)"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path('figures')
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
NAMES = {'logreg': 'Logistic regression', 'svm': 'SVM', 'knn': 'kNN', 'mlp_small': 'Small MLP'}
COLOURS = {'logreg': 'tab:blue', 'svm': 'tab:orange', 'knn': 'tab:gray', 'mlp_small': 'tab:red'}
PREPROC = os.environ.get('PREPROC', 'raw')


def comparison():
    c = pd.read_csv('step5_comparison.csv')
    conf = pd.read_csv('step5_confusion.csv')
    n, pos = int(conf.loc[0, ['TP', 'FN', 'FP', 'TN']].sum()), int(conf.loc[0, ['TP', 'FN']].sum())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    ys = np.arange(len(c))[::-1]
    for ax, m, guess in [(axs[0], 'AUPRC', pos / n), (axs[1], 'AUROC', 0.5)]:
        for yv, (_, r) in zip(ys, c.iterrows()):
            col = 'tab:red' if r['tied with best'] else 'tab:blue'
            ax.errorbar(r[m], yv, xerr=[[r[m] - r[f'{m} CI low']], [r[f'{m} CI high'] - r[m]]], fmt='o', color=col,
                        capsize=4)
            ax.text(r[f'{m} CI high'] + 0.004, yv, f'{r[m]:.3f} ({r[f"{m} CI low"]:.3f}–{r[f"{m} CI high"]:.3f})',
                    va='center', fontsize=8.5)
        ax.set_xlim(0.70, 0.87)
        ax.set_xlabel(f'out-of-fold {m} with 95 % CI (guessing = {guess:.2f}, off the axis)')
    axs[0].set_yticks(ys, [NAMES[k] for k in c['classifier']])
    fig.suptitle(f'FairVLMed: classifier comparison, 5-fold cross-validation ({n:,} patients, {pos:,} glaucoma); '
                 'red = tied with the best by AUPRC', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_comparison.png', dpi=200)
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
                ax.text(j, i, f'{v:,}', ha='center', va='center', fontsize=12,
                        color='white' if v / m[i].sum() > 0.6 else 'black')
            ax.set_xticks([0, 1], ['called\nglaucoma', 'called\nhealthy'])
            ax.set_yticks([0, 1], [f'glaucoma\n({m[0].sum():,})', f'healthy\n({m[1].sum():,})'])
            ax.set_title(f'{NAMES[r["classifier"]]}\nsens {r["sensitivity"]:.2f}, spec {r["specificity"]:.2f}, '
                         f'F1 {r["F1"]:.2f}', fontsize=8.5)
        fig.suptitle(f'FairVLMed: out-of-fold confusion matrices at the {rule} threshold '
                     f'({int(m.sum()):,} training + validation patients)', fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f'fig2_confusion_{tag}.png', dpi=200)
        plt.close(fig)


def fairness_auroc():
    f = pd.read_csv('step6_fairness_ci.csv')
    groups = list(dict.fromkeys(f['group']))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    clfs = list(NAMES)
    for k, n in enumerate(clfs):
        d = f[f['classifier'] == n].set_index('group').loc[groups]
        x = np.arange(len(groups)) + (k - 1.5) * 0.18
        ax.errorbar(x, d['AUROC'], yerr=[d['AUROC'] - d['AUROC CI low'], d['AUROC CI high'] - d['AUROC']], fmt='o',
                    color=COLOURS[n], capsize=3, label=NAMES[n])
    d = f[f['classifier'] == clfs[0]].set_index('group').loc[groups]
    ax.set_xticks(range(len(groups)), [f'{g}\n{int(r["patients"]):,} patients\n{r["glaucoma"] / r["patients"]:.0%} glaucoma'
                                       for g, r in d.iterrows()])
    ax.axvline(2.5, c='0.8', lw=1)
    ax.set_ylabel('out-of-fold AUROC, 95 % CI')
    ax.legend(frameon=False, ncol=4, fontsize=8.5, loc='lower center')
    ax.set_ylim(0.62, 0.87)
    ax.set_title('FairVLMed: AUROC by race and by gender (2,000 resamples of the patients of each group)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_fairness_auroc.png', dpi=200)
    plt.close(fig)


def fairness_rates():
    f = pd.read_csv('step6_fairness_ci.csv')
    chosen = json.load(open('step5_summary.json'))['chosen (simplest tied)']
    d = f[f['classifier'] == chosen].reset_index(drop=True)
    conf = pd.read_csv('step5_confusion.csv')
    conf = conf[conf['classifier'] == chosen].set_index('threshold rule')
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    x = np.arange(len(d))
    for ax, (rule, tag) in zip(axs, [('90% sensitivity', 'sens90'), ("Youden's J", 'youden')]):
        for off, m, col in [(-0.2, 'sensitivity', 'tab:red'), (0.2, 'specificity', 'tab:blue')]:
            pos = d[f'{tag} TP'] + d[f'{tag} FN'] if m == 'sensitivity' else d[f'{tag} FP'] + d[f'{tag} TN']
            p = d[f'{tag} {m}']
            z = 1.96                                                    # Wilson 95 % interval from the counts
            centre = (p + z * z / (2 * pos)) / (1 + z * z / pos)
            half = z * np.sqrt(p * (1 - p) / pos + z * z / (4 * pos * pos)) / (1 + z * z / pos)
            ax.bar(x + off, p, 0.38, color=col, alpha=0.75, label=m)
            ax.errorbar(x + off, p, yerr=[p - (centre - half), (centre + half) - p], fmt='none', ecolor='k', capsize=3)
            ax.axhline(conf.loc[rule, m], color=col, ls=':', lw=1)
            for xi, v in zip(x + off, p):
                ax.text(xi, 0.02, f'{v:.2f}', ha='center', fontsize=8, color='white')
        ax.set_xticks(x, d['group'])
        ax.axvline(2.5, c='0.8', lw=1)
        ax.set_title(f'{rule} threshold (dotted = all patients)', fontsize=10)
        ax.set_ylim(0, 1)
    axs[0].legend(frameon=False, fontsize=8.5, loc='upper right', ncol=2)
    fig.suptitle(f'FairVLMed, {NAMES[chosen]}: sensitivity and specificity of each group at the one threshold set on '
                 'all patients (Wilson 95 % intervals)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_fairness_rates.png', dpi=200)
    plt.close(fig)


def searches():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = f'sqlite:///optuna_{PREPROC}.db'
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    d = optuna.load_study(study_name='logreg_auprc', storage=db).trials_dataframe().sort_values('params_C')
    axs[0].semilogx(d['params_C'], d['value'], 'o-', ms=3, c='tab:blue')
    b = d.loc[d['value'].idxmax()]
    axs[0].axvline(b['params_C'], ls='--', c='gray')
    axs[0].set_xlabel('C (smaller = stronger L2 penalty)'); axs[0].set_ylabel('5-fold mean validation AUPRC')
    axs[0].set_title(f'Logistic regression: {len(d)} trials, best C = {b["params_C"]:.4f}', fontsize=10)
    d = optuna.load_study(study_name='svm_auprc', storage=db).trials_dataframe()
    d = d[d['state'] == 'COMPLETE']
    r, lin = d[d['params_kernel'] == 'rbf'], d[d['params_kernel'] == 'linear']
    sc = axs[1].scatter(r['params_C'], r['params_gamma'], c=r['value'], cmap='viridis', s=30, vmin=0.70,
                        vmax=d['value'].max())
    b = d.loc[d['value'].idxmax()]
    axs[1].scatter([b['params_C']], [b['params_gamma']], marker='*', s=160, facecolor='none', edgecolor='red')
    axs[1].set_xscale('log'); axs[1].set_yscale('log')
    axs[1].set_xlabel('C'); axs[1].set_ylabel('gamma (RBF kernel)')
    axs[1].set_title(f'SVM: {len(d)} finished trials ({len(r)} RBF, {len(lin)} linear;\nbest linear '
                     f'{lin["value"].max():.3f}, best RBF {r["value"].max():.3f})', fontsize=10)
    fig.colorbar(sc, ax=axs[1], label='AUPRC (below 0.70 = darkest)')
    d = optuna.load_study(study_name='knn_auprc', storage=db).trials_dataframe()
    for (w, m), g in d.groupby(['params_weights', 'params_metric']):
        g = g.sort_values('params_k')
        axs[2].plot(g['params_k'], g['value'], label=f'{w} weights, {m}')
    axs[2].set_xlabel('k (number of neighbours)'); axs[2].legend(frameon=False, fontsize=8)
    axs[2].set_title(f'kNN: all {len(d)} grid settings', fontsize=10)
    fig.suptitle('FairVLMed: hyperparameter searches (objective = 5-fold mean validation AUPRC)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_searches.png', dpi=200)
    plt.close(fig)


def mlp_search():
    import optuna
    from optuna.storages.journal import JournalFileBackend
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.JournalStorage(JournalFileBackend(f'optuna_mlp_small_{PREPROC}.log'))
    d = optuna.load_study(study_name='mlp_small', storage=storage).trials_dataframe()
    d = d[d['state'] == 'COMPLETE']
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={'width_ratios': [2, 1]})
    combos = sorted(d.groupby(['params_width1', 'params_width2']).groups)
    a.boxplot([d.loc[(d['params_width1'] == w1) & (d['params_width2'] == w2), 'value'] for w1, w2 in combos])
    a.set_xticks(range(1, len(combos) + 1),
                 [f'{int(w1)}-{int(w2)}\n({((d["params_width1"] == w1) & (d["params_width2"] == w2)).sum()})'
                  for w1, w2 in combos], fontsize=7.5)
    a.set_xlabel('neurons in layer 1 - layer 2 (number of trials)'); a.set_ylabel('5-fold mean validation AUPRC')
    a.set_ylim(0.77, 0.81)
    a.set_title(f'Small MLP search: {len(d)} trials ({(d["value"] < 0.77).sum()} below 0.77 not shown)', fontsize=10)
    losses = ['bce', 'weighted_bce', 'focal']
    b.boxplot([d.loc[d['params_loss'] == l, 'value'] for l in losses])
    b.set_xticks([1, 2, 3], [f'{n}\n({(d["params_loss"] == l).sum()} trials)' for n, l in
                             zip(['plain\ncross-entropy', 'weighted\ncross-entropy', 'focal\n(weighted)'], losses)],
                 fontsize=8.5)
    b.set_ylim(0.77, 0.81)
    b.set_title('By loss function', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_small_mlp_search.png', dpi=200)
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
    b.axhline(0.49, ls=':', c='gray', label='AUPRC of guessing (0.49 in the validation part)')
    b.axvline(best, ls='--', c='gray')
    b.set_xlabel('epoch'); b.legend(frameon=False, fontsize=8)
    s = json.load(open('step5_summary.json'))
    hp = s['best hyperparameters']['mlp_small']
    fig.suptitle(f'FairVLMed, small MLP ({hp["width1"]} and {hp["width2"]} neurons, '
                 f'{s["mlp_small trainable parameters"]:,} parameters): all 300 epochs on the fixed training (7,000) / '
                 'validation (1,000) split', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig7_small_mlp_learning_curve.png', dpi=200)
    plt.close(fig)


def losses():
    df = pd.read_csv('check_losses.csv')
    per_seed = df.groupby(['loss', 'seed'])['auprc'].mean().reset_index()
    order = per_seed.groupby('loss')['auprc'].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, l in enumerate(order):
        v = per_seed.loc[per_seed['loss'] == l, 'auprc']
        ax.plot([v.min(), v.max()], [i, i], c='tab:blue')
        ax.plot(v.mean(), i, 'o', c='tab:blue')
        ax.text(v.max() + 0.0003, i, f'{v.mean():.3f}', va='center', fontsize=8.5)
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    ax.set_xlabel('kept-model validation AUPRC, mean of 5 folds (dot = mean of 5 seeds, line = range over seeds)')
    ax.set_title('FairVLMed: best small MLP trained with different loss functions', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig8_loss_functions.png', dpi=200)
    plt.close(fig)


for f in [comparison, confusion, fairness_auroc, fairness_rates, searches, mlp_search, learning_curve, losses]:
    try:
        f()
        print('made', f.__name__)
    except (FileNotFoundError, KeyError) as e:
        print('skipped', f.__name__, '(input not ready:', e, ')')
