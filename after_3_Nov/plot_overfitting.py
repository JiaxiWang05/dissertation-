"""Figures for the MLP-overfitting write-up, drawn from saved results (the only model fitted here is the
logistic-regression reference line: C = 0.00537, the same 5 folds).
Inputs: check_noise.csv, check_noise_curves.csv, check_lr_epoch_curves.csv, check_mlp_seeds.csv, optuna.db (study 'mlp').
Run: python plot_overfitting.py   (writes figures/overfitting_*.png)"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import STORAGE, cv_folds, load_data

INK, INK2, GRID, GREY = '#0b0b0b', '#52514e', '#e6e5e1', '#8a8984'
COLOR = {'3 Nov': '#2a78d6', 'trial 72': '#eb6834', 'trial 112': '#1baf7a'}   # one colour per setting, every figure
MARKER = {'3 Nov': 'o', 'trial 72': 's', 'trial 112': '^'}
RAMP = {0.0: '#86b6ef', 0.3: '#3987e5', 1.0: '#1c5cab', 2.0: '#0d366b'}        # noise levels in the curve figure
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.edgecolor': GREY, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 10,
                     'legend.frameon': False, 'axes.axisbelow': True, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
os.makedirs('figures', exist_ok=True)


LABEL = {'3 Nov': '3 Nov setting', 'trial 72': 'trial 72 (search 1)', 'trial 112': 'trial 112 (search 2)'}


def short(name):                                   # 'trial 72 (RMSprop, ...)' -> 'trial 72'
    return '3 Nov' if name.startswith('3 Nov') else ' '.join(name.split()[:2])


def save(fig, name, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(f'figures/{name}.png', dpi=200)
    plt.close(fig)
    print('saved figures/' + name + '.png')


# logistic regression on the same 5 folds: the reference every MLP is compared with
lab, X, y = load_data()
LOGREG = np.mean([roc_auc_score(yva, LogisticRegression(C=0.00537, class_weight='balanced', max_iter=5000)
                                .fit(Xtr, ytr).decision_function(Xva)) for Xtr, ytr, Xva, yva in cv_folds(lab, X, y)])
print('logistic regression, 5-fold mean AUROC:', round(LOGREG, 4))

cv = pd.read_csv('check_noise.csv').assign(s=lambda d: d['setting'].map(short))
curves = pd.read_csv('check_noise_curves.csv').assign(s=lambda d: d['setting'].map(short))
NOISE = sorted(cv['noise'].unique())
pos = {n: i for i, n in enumerate(NOISE)}          # noise levels at even spacing on the x axis

# ---- 1. what overfitting looks like: the 3 Nov setting, fixed split, no early stop, no noise
h = curves[(curves['s'] == '3 Nov') & (curves['noise'] == 0)]
best = int(h['lowest_val_loss_epoch'].iloc[0])
kept = h[h['epoch'] == best].iloc[0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
a1.axvspan(best, h['epoch'].max(), color='#fbe3d9', alpha=0.6, lw=0)
a1.plot(h['epoch'], h['train_loss'], color=COLOR['3 Nov'], ls='--', lw=2, label='training loss')
a1.plot(h['epoch'], h['val_loss'], color=COLOR['3 Nov'], lw=2, label='validation loss')
a1.axvline(best, color=GREY, ls=':', lw=1.5)
a1.plot([best, best], [kept['train_loss'], kept['val_loss']], color=INK, lw=2.5)       # the gap at the kept epoch
a1.annotate(f"kept model (epoch {best}):\ngap = validation - training loss\n= {kept['val_loss']:.2f} - {kept['train_loss']:.2f}"
            f" = {kept['val_loss'] - kept['train_loss']:.2f}", xy=(best, (kept['train_loss'] + kept['val_loss']) / 2),
            xytext=(h['epoch'].max() * 0.3, h['val_loss'].max() * 0.3), fontsize=9, color=INK,
            bbox={'boxstyle': 'round,pad=0.4', 'fc': 'white', 'ec': GREY}, arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 1})
a1.text(h['epoch'].max() * 0.55, h['val_loss'].max() * 0.85, 'after it, overfitting:\nvalidation loss rises,\ntraining loss falls',
        fontsize=9, color=INK2)
a1.set(xlabel='epoch', ylabel='loss (weighted binary cross-entropy)', title='Loss: training vs validation')
a1.legend(loc='upper left')
a2.axvspan(best, h['epoch'].max(), color='#fbe3d9', alpha=0.6, lw=0)
a2.plot(h['epoch'], h['val_auroc'], color=COLOR['3 Nov'], lw=1.5)
a2.axvline(best, color=GREY, ls=':', lw=1.5)
a2.set(xlabel='epoch', ylabel='validation AUROC', title='Validation AUROC: the ranking gets only a little worse')
fig.suptitle('3 Nov MLP setting, 300 epochs without early stopping (fixed training / validation split)', fontsize=10)
save(fig, 'overfitting_1_what_it_looks_like')

# ---- 2. a smaller learning rate only stretches the same curve
lc = pd.read_csv('check_lr_epoch_curves.csv')
lrs = sorted(lc['lr'].unique(), reverse=True)
shade = {lrs[0]: '#6da7ec', lrs[1]: '#104281'}
lab_lr = {lrs[0]: f'lr {lrs[0]:.2g} (3 Nov)', lrs[1]: f'lr {lrs[1]:.2g} (÷ 10)'}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for lr in lrs:
    for k, g in lc[lc['lr'] == lr].groupby('fold'):
        ke = int(g['kept_epoch'].iloc[0])
        for ax, x in [(a1, g['epoch']), (a2, g['epoch'] * lr / lrs[0])]:
            ax.plot(x, g['val_loss'], color=shade[lr], lw=1.3, label=lab_lr[lr] if k == 0 else None)
            ax.plot(x[g['epoch'] == ke], g.loc[g['epoch'] == ke, 'val_loss'], 'o', color=shade[lr], ms=6,
                    mec='white', mew=1.5)
a1.set(xlabel='epoch', ylabel='validation loss', title='Per epoch: the smaller lr is kept ~10x later (dots)')
a2.set(xlabel='epoch x (lr / 3 Nov lr): epochs counted in 3 Nov-lr steps',
       title='Per amount of learning: the same descent and the same minimum', xlim=(0, 20))
a1.legend()
fig.suptitle('3 Nov MLP setting at its own learning rate and at lr / 10, 5 folds (dots = kept epoch)', fontsize=10)
save(fig, 'overfitting_2_learning_rate')

# ---- 3. the 5-fold Optuna search: a flat surface, pushed to the lr lower bound
study = optuna.load_study(study_name='mlp', storage=STORAGE)
t = pd.DataFrame([{'lr': tr.params['lr'], 'opt': tr.params['optimizer'], 'width': tr.params['width'],
                   'score': tr.value, 'number': tr.number} for tr in study.trials if tr.value is not None])
LOW = 0.78                                         # zoom on the top: the few failed settings sit far below
n_low = int((t['score'] < LOW).sum())
fig, ax = plt.subplots(figsize=(10, 4))
for is_adam, g in t.groupby(t['opt'] == 'adam'):
    shown, below = int((g['score'] >= LOW).sum()), int((g['score'] < LOW).sum())
    ax.scatter(g['lr'], g['score'], s=22, color=INK2 if is_adam else '#6250d6', alpha=0.55 if is_adam else 0.9, lw=0,
               zorder=2 if is_adam else 3,
               label=f"{'Adam' if is_adam else 'AdamW / SGD / RMSprop'} ({shown} shown, {below} below {LOW})")
b = t.loc[t['score'].idxmax()]
ax.scatter([b['lr']], [b['score']], s=120, facecolor='none', edgecolor=COLOR['trial 112'], lw=2.5,
           label=f"best: trial {int(b['number'])} ({b['score']:.3f})")
ax.axhline(LOGREG, color=GREY, ls='--', lw=1.5)
ax.text(t['lr'].max(), LOGREG + 0.002, f'logistic regression {LOGREG:.3f}', ha='right', fontsize=9, color=INK2)
ax.axvline(1e-5, color=GREY, ls=':', lw=1)
ax.text(1.1e-5, LOW + 0.001, 'search lower bound', fontsize=8, color=INK2)
ax.set_xscale('log')
ax.set_ylim(LOW, t['score'].max() + 0.006)
ax.set(xlabel='learning rate (log scale)', ylabel='5-fold score (val AUROC, kept epoch ± 5)',
       title=f'Optuna 5-fold MLP search, {len(t)} trials ({n_low} below {LOW} not shown): '
             f'median {t["score"].median():.3f}, best {t["score"].max():.3f}')
ax.legend(loc='lower right')
save(fig, 'overfitting_3_optuna_search')

# ---- 4. seed check: four settings, 5 new seeds x the same 5 folds
sc = pd.read_csv('check_mlp_seeds.csv')
order = list(dict.fromkeys(sc['setting']))
nice = {order[0]: '3 Nov setting', order[1]: 'trial 174 (search 1 best)', order[2]: 'trial 72 (search 1)',
        order[3]: 'trial 112 (search 2 best)'}
col = {order[0]: COLOR['3 Nov'], order[1]: GREY, order[2]: COLOR['trial 72'], order[3]: COLOR['trial 112']}
per_seed = sc.groupby(['setting', 'seed'])['auroc'].mean()
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for i, s in enumerate(order):
    v = per_seed[s]
    a1.plot([v.min(), v.max()], [i, i], color=col[s], lw=3, solid_capstyle='round')
    a1.plot(v.mean(), i, 'o', color=col[s], ms=9, mec='white', mew=2)
    a1.text(v.max() + 0.002, i, f'{v.mean():.3f}', va='center', fontsize=9, color=INK)
    gap = sc.loc[sc['setting'] == s, 'loss_gap'].mean()
    a2.barh(i, gap, color=col[s], height=0.55)
    a2.text(gap + 0.005, i, f'{gap:.2f}', va='center', fontsize=9, color=INK)
    ep = sc.loc[sc['setting'] == s, 'kept_epoch'].median()
    a3.barh(i, ep, color=col[s], height=0.55)
    a3.text(ep + 1, i, f'{ep:.0f}', va='center', fontsize=9, color=INK)
a1.axvline(LOGREG, color=GREY, ls='--', lw=1.5)
a1.text(LOGREG + 0.0005, len(order) - 0.6, 'logistic regression', fontsize=8, color=INK2, ha='left', va='bottom')
a1.set_yticks(range(len(order)), [nice[s] for s in order])
a1.invert_yaxis()
a1.set(xlabel='AUROC (dot = mean of 5 seeds, bar = seed range)', title='Kept-model AUROC')
a2.set(xlabel='val minus train loss at the kept epoch', title='Overfitting (smaller = less)')
a3.set(xlabel='kept epoch (median)', title='Kept epoch')
fig.suptitle('Seed check: 5 new seeds x the same 5 folds (training + validation only)', fontsize=10)
save(fig, 'overfitting_4_seed_check')

# ---- 5. noise: AUROC, overfitting gap, kept epoch, lowest validation loss vs noise level
per_seed = cv.groupby(['s', 'noise', 'seed'])['auroc'].mean()
agg = cv.groupby(['s', 'noise']).agg(gap=('loss_gap', 'mean'), epoch=('kept_epoch', 'median'),
                                     vloss=('lowest_val_loss', 'mean'))
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for s in COLOR:
    x = [pos[n] for n in NOISE]
    a = per_seed[s].groupby(level=0)
    m, lo, hi = a.mean().loc[NOISE], a.min().loc[NOISE], a.max().loc[NOISE]
    axes[0, 0].fill_between(x, lo, hi, color=COLOR[s], alpha=0.15, lw=0)
    for ax, v in [(axes[0, 0], m), (axes[0, 1], agg.loc[s, 'gap'].loc[NOISE]),
                  (axes[1, 0], agg.loc[s, 'epoch'].loc[NOISE]), (axes[1, 1], agg.loc[s, 'vloss'].loc[NOISE])]:
        ax.plot(x, v, color=COLOR[s], lw=2, marker=MARKER[s], ms=7, mec='white', mew=1.5, label=LABEL[s])
capped = cv[cv['kept_epoch'] > 300 - 30].groupby(['s', 'noise']).size()   # early stopping cannot end these runs
for (s, n), k in capped.items():
    axes[1, 0].annotate(f'{k} of 25 runs stopped by the\n300-epoch limit (not converged)', xy=(pos[n], agg.loc[(s, n), 'epoch']),
                        xytext=(pos[n] - 3.6, agg.loc[(s, n), 'epoch'] * 0.7), fontsize=8, color=INK2,
                        arrowprops={'arrowstyle': '->', 'color': INK2, 'lw': 0.8})
axes[0, 0].axhline(LOGREG, color=GREY, ls='--', lw=1.5, label='logistic regression')
axes[0, 0].set(ylabel='AUROC (mean of 5 seeds; band = seed range)', title='Kept-model AUROC (higher = better)')
axes[0, 1].set(ylabel='val minus train loss', title='Overfitting gap at the kept epoch (smaller = less)')
axes[1, 0].set(ylabel='kept epoch (median)', title='Kept epoch')
axes[1, 1].set(ylabel='lowest validation loss', title='Lowest validation loss (lower = better)')
for ax in axes[1]:
    ax.set_xticks(range(len(NOISE)), [f'{n:g}' for n in NOISE])
    ax.set_xlabel('noise added to the training features (standard deviations)')
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=4)
fig.suptitle('Gaussian noise on the training features: 3 settings x 7 noise levels x 5 seeds x the same 5 folds', fontsize=10)
save(fig, 'overfitting_5_noise_summary', rect=(0, 0.04, 1, 1))

# ---- 6. learning curves with and without noise (fixed split, 300 epochs, no early stop)
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
ymax = curves.loc[curves['noise'].isin(RAMP), 'val_loss'].max()
for j, s in enumerate(COLOR):
    for n, c in RAMP.items():
        g = curves[(curves['s'] == s) & (curves['noise'] == n)]
        if g.empty:
            continue
        be = int(g['lowest_val_loss_epoch'].iloc[0])
        axes[0, j].plot(g['epoch'], g['train_loss'], color=c, ls='--', lw=1.2)
        axes[0, j].plot(g['epoch'], g['val_loss'], color=c, lw=2, label=f'noise {n:g}')
        axes[0, j].plot(be, g.loc[g['epoch'] == be, 'val_loss'], 'o', color=c, ms=7, mec='white', mew=1.5, zorder=5)
        axes[1, j].plot(g['epoch'], g['val_auroc'].rolling(9, min_periods=1, center=True).mean(), color=c, lw=1.6)
    axes[0, j].set(title=LABEL[s], ylim=(0, ymax * 1.05))
    axes[1, j].set_xlabel('epoch')
axes[0, 0].set_ylabel('loss (solid = validation, dashed = training)')
axes[1, 0].set_ylabel('validation AUROC (9-epoch moving average)')
axes[0, 0].legend(loc='upper left', fontsize=9)
fig.suptitle('Learning curves, fixed training / validation split, 300 epochs (dot = lowest validation loss)', fontsize=10)
save(fig, 'overfitting_6_noise_curves')

# ---- 7. per fold: AUROC with noise minus without (5-seed mean per fold)
fm = cv.groupby(['s', 'noise', 'fold'])['auroc'].mean()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for j, s in enumerate(COLOR):
    d = fm[s] - fm[s].xs(0.0, level='noise').reindex(fm[s].index.get_level_values('fold')).to_numpy()
    for n in NOISE[1:]:
        v = d.xs(n, level='noise')
        axes[j].scatter(pos[n] + np.linspace(-0.15, 0.15, len(v)), v, s=28, color=COLOR[s], alpha=0.75, lw=0)
        axes[j].plot([pos[n] - 0.25, pos[n] + 0.25], [v.mean()] * 2, color=INK, lw=2)
    axes[j].axhline(0, color=GREY, lw=1)
    axes[j].set_xticks([pos[n] for n in NOISE[1:]], [f'{n:g}' for n in NOISE[1:]])
    axes[j].set(title=LABEL[s], xlabel='noise (standard deviations)')
axes[0].set_ylabel('AUROC minus noise 0\n(dot = one fold, line = mean)')
fig.suptitle('Does noise help in every fold? Paired with noise 0: same setting, seeds and fold', fontsize=10)
save(fig, 'overfitting_7_noise_per_fold')
