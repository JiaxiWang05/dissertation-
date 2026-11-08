"""Check: search 2's best MLP sits at two edges of its search range (check_search_edges.py): learning rate at the lower
bound 0.00001 and width at the upper bound 512. Does the score keep improving beyond them?
Trial 112's setting, one change at a time: learning rate / 2, / 4, / 8 (width 512), and width 1024, 2048 (its own
learning rate). 5 seeds x the same 5 folds (training + validation; the test part is not used). Up to 1500 epochs, so
the small learning rates can reach their lowest validation loss (patience 30 as before).
Per run: the search's own score (validation AUROC averaged over the kept epoch +- 5), AUROC of the kept model,
kept epoch, validation minus training loss.
Run: CELL=0 python check_mlp_edges.py   (one cell per Slurm array task; writes check_mlp_edges/cell_0.csv)
     python check_mlp_edges.py summary   (after all cells: prints the table, writes check_mlp_edges.csv)"""
import glob
import os
import sys

import pandas as pd

from common import cv_folds, load_data, train_mlp

SEEDS = [0, 1, 2, 3, 4]                          # as in check_mlp_seeds.py, so 'as found' repeats its trial 112 runs
WINDOW, MAX_EPOCHS, PATIENCE = 5, 1500, 30
TRIAL_112 = {'n_layers': 2, 'width': 512, 'dropout': 0.14945119263016082, 'optimizer': 'adam',
             'lr': 1.5668028813030473e-05, 'lr_schedule': 'constant', 'weight_decay': 4.893450570717874e-05,
             'l1': 2.57827285430375e-08, 'batch_size': 64, 'loss': 'weighted_bce'}   # optuna.db, study 'mlp'
LR = TRIAL_112['lr']
CELLS = [('as found (width 512, lr 0.0000157)', {}),
         ('lr / 2', {'lr': LR / 2}), ('lr / 4', {'lr': LR / 4}), ('lr / 8', {'lr': LR / 8}),
         ('width 1024', {'width': 1024}), ('width 2048', {'width': 2048})]

if sys.argv[1:] == ['summary']:
    r = pd.concat(pd.read_csv(f) for f in sorted(glob.glob('check_mlp_edges/cell_*.csv')))
    r.to_csv('check_mlp_edges.csv', index=False)
    order = [name for name, _ in CELLS if name in set(r['cell'])]
    out = {}
    for score, label in [('window_score', 'search score (val AUROC, kept epoch +- 5)'), ('auroc', 'kept-model AUROC')]:
        per_seed = r.groupby(['cell', 'seed'])[score].mean().groupby(level=0)
        fold = r.groupby(['cell', 'fold'])[score].mean()           # 5-seed mean per fold, paired with 'as found'
        base = fold.xs(order[0], level='cell')
        diff = {c: fold.xs(c, level='cell') - base for c in order}
        out[f'{label}, mean of 5 seeds'] = per_seed.mean()
        out[f'{label}, seed range'] = per_seed.min().round(3).astype(str) + '-' + per_seed.max().round(3).astype(str)
        out[f'{label}, minus as found'] = pd.Series({c: d.mean() for c, d in diff.items()})
        out[f'{label}, folds better (of 5)'] = pd.Series({c: int((d > 0).sum()) for c, d in diff.items()})
    by = r.groupby('cell')
    out['kept epoch (median)'] = by['kept_epoch'].median()
    out['val minus train loss'] = by['loss_gap'].mean()
    out['runs stopped by the epoch limit'] = by['capped'].sum()
    pd.set_option('display.width', 300)
    print(pd.DataFrame(out).loc[order].round(4).T.to_string())

    # figure: kept-model AUROC beyond each edge (dot = mean of 5 seeds, bar = range over seeds)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e5e1', 'axes.axisbelow': True})
    seed_auroc = r.groupby(['cell', 'seed'])['auroc'].mean()
    epochs = r.groupby('cell')['kept_epoch'].median()
    LOGREG = 0.8135                              # logistic regression, C = 0.00537, the same 5 folds (plot_overfitting.py)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, names, xs, edge, outside, xlabel in [
            (axes[0], [order[3], order[2], order[1], order[0]], [LR / 8, LR / 4, LR / 2, LR], 1e-5, 'left',
             'learning rate (log scale), width 512'),
            (axes[1], [order[0], order[4], order[5]], [512, 1024, 2048], 512, 'right', 'width (neurons per layer), lr 0.0000157')]:
        v = [seed_auroc[n] for n in names]
        ax.axvspan(*((xs[0] / 1.5, edge) if outside == 'left' else (edge, xs[-1] * 1.5)), color='#f0efec', lw=0)
        ax.axvline(edge, color='#8a8984', ls=':', lw=1.5)
        ax.text(edge, 0.8168, "  outside search 2's range" if outside == 'right' else "outside search 2's range  ",
                fontsize=8, color='#52514e', ha='left' if outside == 'right' else 'right', va='center')
        ax.errorbar(xs, [x.mean() for x in v], yerr=[[x.mean() - x.min() for x in v], [x.max() - x.mean() for x in v]],
                    color='#1baf7a', marker='o', ms=8, mec='white', mew=1.5, lw=2, capsize=4)
        for x, n, val in zip(xs, names, v):
            ax.annotate(f'{val.mean():.3f}\nepoch {epochs[n]:.0f}', (x, val.max()), textcoords='offset points',
                        xytext=(0, 6), ha='center', fontsize=8, color='#0b0b0b')
        ax.axhline(LOGREG, color='#8a8984', ls='--', lw=1.5)
        ax.set_xscale('log', base=2 if outside == 'right' else 10)
        ax.set_xlim(xs[0] / 1.5, xs[-1] * 1.5)
        ax.set_ylim(0.800, 0.8178)
        ax.set_xlabel(xlabel)
        ax.minorticks_off()
        ax.set_xticks(xs, [str(x) for x in xs] if outside == 'right' else [f'{x:.1e}' for x in xs])
    axes[0].set_ylabel('kept-model AUROC (5 folds; bar = seed range)')
    axes[0].set_title('Learning rate below the search range: slightly worse')
    axes[1].set_title('Width above the search range: +0.002 (within the seed range)')
    axes[0].text(LR / 8 / 1.4, LOGREG + 0.0003, ' logistic regression', fontsize=8, color='#52514e', ha='left', va='bottom')
    fig.suptitle("Trial 112 beyond the edges of search 2's range: 5 seeds x the same 5 folds", fontsize=10)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/overfitting_8_search_edges.png', dpi=200, facecolor='white')
    print('saved figures/overfitting_8_search_edges.png')
    sys.exit()

lab, X, y = load_data()
folds = cv_folds(lab, X, y)
cells = [int(os.environ['CELL'])] if 'CELL' in os.environ else range(len(CELLS))
os.makedirs('check_mlp_edges', exist_ok=True)
for c in cells:
    name, change = CELLS[c]
    params, rows = {**TRIAL_112, **change}, []
    for seed in SEEDS:
        for k, (Xtr, ytr, Xva, yva) in enumerate(folds):
            model, history, best_epoch = train_mlp(params, Xtr, ytr, Xva, yva, max_epochs=MAX_EPOCHS,
                                                   patience=PATIENCE, seed=seed)
            near = history[(history['epoch'] - best_epoch).abs() <= WINDOW]
            kept = history.loc[best_epoch - 1]
            rows.append({'cell': name, 'seed': seed, 'fold': k, 'window_score': near['val_auroc'].mean(),
                         'auroc': kept['val_auroc'], 'kept_epoch': best_epoch,
                         'loss_gap': kept['val_loss'] - kept['train_loss'],
                         'capped': best_epoch > MAX_EPOCHS - PATIENCE})   # still improving when the limit stopped it
        print(f'{name}, seed {seed}: done', flush=True)
    pd.DataFrame(rows).to_csv(f'check_mlp_edges/cell_{c}.csv', index=False)
