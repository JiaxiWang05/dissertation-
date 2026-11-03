"""Learning curve: AUROC of the tuned logistic regression on frozen features vs the number of training images.

For n = 25 / 50 / 100 / 200 and 10 random draws each, n training images are drawn (stratified: the glaucoma share of
the training pool, at least 3 glaucoma), C is tuned by patient-grouped inner CV on those images only (ties in inner
AUROC, common with few glaucoma patients, are broken by inner log-loss), and the head is tested:
- PAPILA: draws from each official fold's training patients, AUROC inside that fold's test patients, averaged over
  the 5 folds (the main table's pooled out-of-fold AUROC is lower: it mixes the 5 fold models' score scales);
- Chakshu: draws from all PAPILA, AUROC on all of Chakshu.
'all' = every training image, one fit (per fold), as in run_linear_probe.py --protocol tuned.
Spread = SD over the draws. Draws overlap more as n nears the pool size, so the SD then understates the spread
of a new training set of that size.

Usage:
    python learning_curve.py --papila-labels papila_labels.csv --papila-feats F/papila_rfg_resize.npz \
        --chaksu-labels chaksu_labels.csv --chaksu-feats F/chaksu_rfg_resize.npz --out R/learning_curve_resize
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rfg
from run_linear_probe import C_GRID

SIZES = [25, 50, 100, 200]
INPUTS = {'resize': 'full image', 'fov_square': 'retina crop + pad', 'disc_crop': 'optic-disc crop'}
SURFACE, INK, INK2, GRID, GUESS = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
SERIES = {'PAPILA': '#2a78d6', 'Chákṣu': '#eb6834'}  # categorical slots 1-2, as in plot_curves.py


def draw(y, n, rng):
    """Indices of n images, stratified by label: the pool's glaucoma share, at least 3 of each class."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = int(np.clip(round(n * len(pos) / len(y)), 3, n - 3))
    return np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)])


def best_auc_then_logloss(cv):
    """Index of the C with the best inner AUROC; ties broken by the best inner log-loss."""
    auc, nll = np.asarray(cv['mean_test_auc'], float), np.asarray(cv['mean_test_nll'], float)
    if not np.isfinite(auc).any():
        return 0
    tied = np.flatnonzero(np.isclose(auc, np.nanmax(auc)))
    return int(tied[np.nanargmax(nll[tied])])


def fit_tuned(X, y, groups, seed, tries=5):
    """As run_linear_probe.fit_head (protocol tuned), with fewer inner folds when few patients have glaucoma.
    An inner split without both classes gives NaN scores; then the split is reshuffled."""
    n_splits = int(min(5, len(np.unique(groups[y == 1])), len(np.unique(groups[y == 0]))))
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight='balanced', max_iter=5000, random_state=seed))
    for t in range(tries):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1000 * t)
        search = GridSearchCV(base, {'logisticregression__C': C_GRID}, scoring={'auc': 'roc_auc', 'nll': 'neg_log_loss'},
                              refit=best_auc_then_logloss, cv=cv)
        search.fit(X, y, groups=groups)
        if np.isfinite(search.cv_results_['mean_test_auc']).any():
            C = float(search.best_params_['logisticregression__C'])
            n_tied = int(np.isclose(search.cv_results_['mean_test_auc'], np.nanmax(search.cv_results_['mean_test_auc'])).sum())
            return clone(base).set_params(logisticregression__C=C).fit(X, y), C, n_splits, n_tied
    raise RuntimeError(f'no inner split with both classes after {tries} tries')


def one_fit(test, fold_no, n, d, X, y, g, tr, X_te, y_te):
    idx = tr if n == 'all' else tr[draw(y[tr], n, np.random.default_rng([fold_no, n, d]))]
    head, C, k, n_tied = fit_tuned(X[idx], y[idx], g[idx], seed=d)
    m = rfg.metrics(y_te, head.predict_proba(X_te)[:, 1])
    return {'test': test, 'fold': fold_no, 'size': str(n), 'n_train': len(idx), 'n_train_glaucoma': int(y[idx].sum()),
            'draw': d, 'C': C, 'inner_folds': k, 'C_tied': n_tied, 'auroc': m['auroc'], 'auprc': m['auprc']}


def tasks(test, fold_no, X, y, g, tr, X_te, y_te, draws):
    return [delayed(one_fit)(test, fold_no, n, d, X, y, g, tr, X_te, y_te)
            for n in SIZES + ['all'] for d in range(1 if n == 'all' else draws)]


def summarise(df):
    """One value per draw (PAPILA: mean of the within-fold values over the folds), then mean and SD over draws."""
    per_draw = df.groupby(['test', 'size', 'draw'], sort=False).agg(
        n_train=('n_train', 'mean'), auroc=('auroc', 'mean'), auprc=('auprc', 'mean'))
    out = []
    for (test, size), s in per_draw.groupby(level=[0, 1], sort=False):
        fold_sd = df[(df['test'] == test) & (df['size'] == size)].groupby('draw')['auroc'].std().mean()
        out.append({'test': test, 'size': size, 'n_train': float(s['n_train'].mean()), 'draws': len(s),
                    'auroc_fold_sd': float(fold_sd),
                    **{f'{m}_{f}': float(getattr(s[m], f)()) for m in ['auroc', 'auprc'] for f in ['mean', 'std']}})
    return pd.DataFrame(out)


def plot(summ, prevalence, draws, label, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for test, color in SERIES.items():
        s = summ[summ['test'] == test]
        x, m, sd = s['n_train'].to_numpy(), s['auroc_mean'].to_numpy(), s['auroc_std'].to_numpy()
        ax.fill_between(x[:-1], (m - sd)[:-1], (m + sd)[:-1], color=color, alpha=0.12, lw=0)
        ax.plot(x, m, lw=2, color=color, label=test)
        ax.plot(x[:-1], m[:-1], 'o', ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
        ax.plot(x[-1], m[-1], 'D', ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f'{m[-1]:.2f}', (x[-1], m[-1]), xytext=(8, 0), textcoords='offset points', va='center',
                    fontsize=8, color=INK)
    ax.axhline(0.5, ls='--', lw=1.2, color=GUESS)
    ax.text(26, 0.505, 'guessing', fontsize=7.5, color=GUESS, va='bottom')
    ax.set_xscale('log', base=2)
    ticks = sorted(set(SIZES) | set(summ.loc[summ['size'] == 'all', 'n_train'].round().astype(int)))
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(21, 520)
    ax.set_ylim(0.45, 0.95)
    ax.set_xlabel('Training images (diamond = all, one fit)', fontsize=9, color=INK2)
    ax.set_ylabel('AUROC', fontsize=9, color=INK2)
    ax.set_title(f'RETFound-Green, {label}: frozen features + tuned logistic regression', loc='left', fontsize=9.5,
                 color=INK, fontweight='bold')
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='upper left', labelcolor=INK)
    fig.text(0.02, 0.012, f'PAPILA: AUROC inside each official test fold, averaged over the 5 folds. Chákṣu: trained on '
             f'PAPILA only.\nBand = ±1 SD over {draws} random draws of the training images (glaucoma share '
             f'{prevalence:.0%} in each); draws overlap more as n grows.', fontsize=7, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=220, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papila-labels', required=True)
    ap.add_argument('--papila-feats', required=True)
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--chaksu-feats', required=True)
    ap.add_argument('--draws', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=1, help='parallel fits')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = Path(args.papila_feats).stem.split('_rfg_')[-1]
    label = INPUTS.get(pre, pre)

    pap = rfg.papila_binary(args.papila_labels, 'exclude')
    y, g, fold = pap['y'].to_numpy(), pap['patient_id'].to_numpy(), rfg.papila_folds(pap, 'official')
    X = rfg.load_features(pap, args.papila_feats)
    cha = rfg.chaksu_external(args.chaksu_labels)
    Xc, yc = rfg.load_features(cha, args.chaksu_feats), cha['y'].to_numpy()

    todo = []
    for k in np.unique(fold):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        todo += tasks('PAPILA', int(k) + 1, X, y, g, tr, X[te], y[te], args.draws)
    todo += tasks('Chákṣu', 0, X, y, g, np.arange(len(y)), Xc, yc, args.draws)
    df = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=5)(todo))
    df.to_csv(out / 'fits.csv', index=False)
    summ = summarise(df)
    (out / 'learning_curve.json').write_text(json.dumps({'input': pre, 'rows': summ.to_dict('records')}, indent=2))
    plot(summ, y.mean(), args.draws, label, out / 'learning_curve.png')

    piv = {t: summ[summ['test'] == t].set_index('size') for t in SERIES}

    def cell(t, s, m):
        sd = piv[t].loc[s, f'{m}_std']
        return f"{piv[t].loc[s, f'{m}_mean']:.3f}" + (f' ± {sd:.3f}' if np.isfinite(sd) else '')

    L = [f'# Learning curve: RETFound-Green, {label}, tuned logistic regression on frozen features\n\n',
         f'Mean ± SD over {args.draws} random draws of the training images. PAPILA: AUROC inside each official test '
         "fold, averaged over the 5 folds (not the pooled out-of-fold AUROC of the main table). Chákṣu: trained on "
         f"PAPILA only. 'all' = every training image, one fit. AUPRC of guessing: PAPILA {y.mean():.2f}, "
         f'Chákṣu {yc.mean():.2f}.\n\n',
         '| Training images | PAPILA AUROC | Chákṣu AUROC | PAPILA AUPRC | Chákṣu AUPRC |\n', '|---|---|---|---|---|\n']
    for s in [str(n) for n in SIZES] + ['all']:
        name = s if s != 'all' else (f"all ({piv['PAPILA'].loc['all', 'n_train']:.0f} / "
                                     f"{piv['Chákṣu'].loc['all', 'n_train']:.0f})")
        L.append(f"| {name} | {cell('PAPILA', s, 'auroc')} | {cell('Chákṣu', s, 'auroc')} | "
                 f"{cell('PAPILA', s, 'auprc')} | {cell('Chákṣu', s, 'auprc')} |\n")
    (out / 'summary.md').write_text(''.join(L))
    print(''.join(L))


if __name__ == '__main__':
    main()
