"""Step 2a: Optuna search over the head on FROZEN RETFound-Green features (0 unfrozen layers).

Nested, so the PAPILA test folds are never used for tuning:
  outer = PAPILA's official patient-level 5 folds (out-of-fold predictions, as in step 1)
  inner = 3-fold patient-grouped CV on the outer-training patients only; Optuna maximises mean inner AUROC.
The best head is refit on the whole outer-training set (mean of 3 seeds) and scored on the outer test fold.
For Chaksu, one more study is run on all of PAPILA and the refit head is applied without retraining.

Search space: hidden layers 0-3. 0 = scikit-learn logistic regression with C searched (the same model family
as step 1); 1-3 = MLP with width, dropout, learning rate, L2 penalty and epochs searched.
Runs on CPU in a few minutes. Unfreezing backbone layers needs a GPU: see optuna_finetune.py.

Usage:
    python optuna_head.py --papila-labels papila_labels.csv --papila-feats F/papila_rfg_resize.npz \
        --chaksu-labels chaksu_labels.csv --chaksu-feats F/chaksu_rfg_resize.npz --out results/optuna_head \
        [--compare-to results/frozen_lr_resize] [--n-trials 40]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

import rfg

FINAL_SEEDS = (0, 1, 2)


def make_mlp(d_in, n_layers, width, dropout):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def fit_predict(Xtr, ytr, Xte, p, seed):
    """Returns P(glaucoma) for Xte. n_layers = 0: scikit-learn logistic regression (balanced classes) with
    the suggested C, i.e. exactly step 1's model family. n_layers > 0: an MLP trained full-batch with Adam on
    class-balanced BCE plus an explicit L2 penalty on the weights (strength l2), on standardised features."""
    sc = StandardScaler().fit(Xtr)
    if p['n_layers'] == 0:
        lr = LogisticRegression(C=p['C'], class_weight='balanced', max_iter=5000, random_state=seed)
        return lr.fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
    torch.manual_seed(seed)
    xtr = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    xte = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    net = make_mlp(xtr.shape[1], p['n_layers'], p['width'], p['dropout'])
    weights = [m.weight for m in net if isinstance(m, nn.Linear)]
    opt = torch.optim.Adam(net.parameters(), lr=p['lr'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1 - ytr).sum() / max(ytr.sum(), 1))))
    net.train()
    for _ in range(p['epochs']):
        opt.zero_grad()
        loss = loss_fn(net(xtr).squeeze(1), y) + p['l2'] * sum((w ** 2).sum() for w in weights)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(xte).squeeze(1)).numpy()


def suggest(trial):
    p = {'n_layers': trial.suggest_int('n_layers', 0, 3)}
    if p['n_layers'] == 0:
        p['C'] = trial.suggest_float('C', 1e-4, 1e2, log=True)
    else:
        p['width'] = trial.suggest_categorical('width', [64, 128, 256, 512])
        p['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        p['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        p['l2'] = trial.suggest_float('l2', 1e-5, 10.0, log=True)
        p['epochs'] = trial.suggest_int('epochs', 50, 400, step=50)
    return p


def run_study(X, y, groups, name, n_trials, seed, storage):
    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(X, y, groups))

    def objective(trial):
        p = suggest(trial)
        scores = []
        for i, (tr, va) in enumerate(inner):
            scores.append(rfg.metrics(y[va], fit_predict(X[tr], y[tr], X[va], p, seed))['auroc'])
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(study_name=name, direction='maximize', storage=storage, load_if_exists=False,
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    study.optimize(objective, n_trials=n_trials)
    best = suggest(optuna.trial.FixedTrial(study.best_params))
    inner_p = np.full(len(y), np.nan)
    for tr, va in inner:
        inner_p[va] = fit_predict(X[tr], y[tr], X[va], best, seed)
    return study, best, rfg.youden_threshold(y, inner_p)


def ensemble(Xtr, ytr, Xte, p):
    return np.mean([fit_predict(Xtr, ytr, Xte, p, s) for s in FINAL_SEEDS], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papila-labels', required=True)
    ap.add_argument('--papila-feats', required=True)
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--chaksu-feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--compare-to', default=None, help='step-1 result folder, for a paired bootstrap')
    ap.add_argument('--n-trials', type=int, default=40)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / 'optuna_head.db'
    db.unlink(missing_ok=True)
    storage = f'sqlite:///{db}'

    pap = rfg.papila_binary(args.papila_labels, 'exclude')
    X, y, g = rfg.load_features(pap, args.papila_feats), pap['y'].to_numpy(), pap['patient_id'].to_numpy()
    fold = rfg.papila_folds(pap, 'official')

    res = {'config': vars(args), 'folds': [], 'papila': {}, 'chaksu': {}}
    oof, thr = np.full(len(y), np.nan), np.full(len(y), np.nan)
    trials = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        study, best, t = run_study(X[tr], y[tr], g[tr], f'fold{k + 1}', args.n_trials, args.seed, storage)
        oof[te], thr[te] = ensemble(X[tr], y[tr], X[te], best), t
        res['folds'].append({'fold': int(k) + 1, 'inner_auroc': study.best_value, 'best': best, 'threshold': t,
                             'test_auroc': rfg.metrics(y[te], oof[te])['auroc']})
        trials.append(study.trials_dataframe().assign(study=f'fold{k + 1}'))
        print(f"fold {k + 1}: inner AUROC {study.best_value:.3f}, test AUROC {res['folds'][-1]['test_auroc']:.3f}, "
              f"best {best}", flush=True)
    res['papila']['Optuna head (frozen)'] = rfg.evaluate(y, oof, g, thr=thr) | {
        'per_fold_auroc': rfg.per_fold_auroc(y, oof, fold)}
    pd.DataFrame({'image_id': pap['image_id'], 'y': y, 'fold': fold + 1, 'p_optuna_head': oof}).to_csv(
        out / 'papila_oof_predictions.csv', index=False)

    if args.compare_to:
        lr = pd.read_csv(Path(args.compare_to) / 'papila_oof_predictions.csv').set_index('image_id')
        p_lr = lr.loc[pap['image_id'], 'p_image'].to_numpy()
        t_lr = lr.loc[pap['image_id'], 'thr_image'].to_numpy()
        res['papila']['Logistic regression (step 1)'] = rfg.evaluate(y, p_lr, g, thr=t_lr) | {
            'per_fold_auroc': rfg.per_fold_auroc(y, p_lr, fold)}
        res['paired'] = rfg.paired_bootstrap_diff(y, rfg.rank_within_fold(oof, fold), rfg.rank_within_fold(p_lr, fold), g)

    # Chaksu: tune on all PAPILA, refit, apply without retraining
    study, best, t = run_study(X, y, g, 'all_papila', args.n_trials, args.seed, storage)
    trials.append(study.trials_dataframe().assign(study='all_papila'))
    res['all_papila_best'] = best | {'inner_auroc': study.best_value, 'threshold': t}
    cha = rfg.chaksu_external(args.chaksu_labels)
    cha['p_glaucoma'] = ensemble(X, y, rfg.load_features(cha, args.chaksu_feats), best)
    res['chaksu']['All cameras'] = rfg.evaluate(cha['y'], cha['p_glaucoma'], cha['group'], thr=t)
    for cam, d in cha.groupby('camera'):
        res['chaksu'][cam] = rfg.evaluate(d['y'], d['p_glaucoma'], d['group'], thr=t)
    res['chaksu_failures'] = rfg.failure_split(cha['y'], cha['p_glaucoma'], cha['n_glaucoma_votes'], t)
    cha[['image_id', 'camera', 'n_glaucoma_votes', 'y', 'p_glaucoma']].to_csv(out / 'chaksu_predictions.csv',
                                                                            index=False)
    pd.concat(trials).to_csv(out / 'optuna_trials.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(res, indent=2, default=float))

    L = ['# RETFound-Green: frozen backbone + Optuna-tuned head\n\n',
         f'Nested: PAPILA official 5 folds outside, 3-fold patient-grouped CV inside; {args.n_trials} trials per '
         f"study; final head = mean of {len(FINAL_SEEDS)} seeds. Threshold: Youden's J on inner predictions "
         'of the training data; frozen from all PAPILA for Chákṣu. Inner AUROC is the selection score '
         '(best of many trials), not a performance estimate.\n\n',
         '## PAPILA (out-of-fold)\n\n', rfg.TABLE_HEADER]
    L += [rfg.fmt_row(k, m) for k, m in res['papila'].items()]
    L += ['\n', rfg.CM_HEADER] + [rfg.fmt_cm(k, m) for k, m in res['papila'].items()]
    L.append('\nAUROC inside each fold (mean ± SD of 5): ' + '; '.join(
        f"{k} {m['per_fold_auroc']['mean']:.3f} ± {m['per_fold_auroc']['sd']:.3f}" for k, m in res['papila'].items()) + '\n')
    if 'paired' in res:
        d = res['paired']
        L.append(f"\nOptuna head minus logistic regression, AUROC on within-fold ranks: {d['diff']:+.3f} "
                 f"(95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}): {d['verdict']}\n")
    L.append('\n| Fold | Inner AUROC | Test AUROC | Best head |\n|---|---|---|---|\n')
    for f in res['folds']:
        L.append(f"| {f['fold']} | {f['inner_auroc']:.3f} | {f['test_auroc']:.3f} | {fmt_params(f['best'])} |\n")
    L += [f"\n## Chákṣu external test (head tuned on all PAPILA: {fmt_params(best)})\n\n", rfg.TABLE_HEADER]
    L += [rfg.fmt_row(k, m) for k, m in res['chaksu'].items()]
    L += ['\n', rfg.CM_HEADER] + [rfg.fmt_cm(k, m) for k, m in res['chaksu'].items()] + ['\n', rfg.CM_NOTE]
    f = res['chaksu_failures']
    L.append(f"\nErrors: {f['n_wrong']} of {f['n']}; dangerous (experts unanimous) {f['dangerous']}, "
             f"safe (experts split) {f['safe']}.\n")
    (out / 'summary.md').write_text(''.join(L))
    print(''.join(L))


def fmt_params(p):
    if p['n_layers'] == 0:
        return f"logistic regression, C {p['C']:.2g}"
    return (f"MLP {p['n_layers']}×{p['width']}, dropout {p['dropout']:.2f}, lr {p['lr']:.1e}, "
            f"L2 {p['l2']:.1e}, {p['epochs']} epochs")


if __name__ == '__main__':
    main()
