"""Step 1: frozen RETFound-Green features + logistic regression.

--protocol paper  replicates the RETFound-Green paper's probe: StandardScaler + LogisticRegression with
                  sklearn defaults (C = 1, no class weights, max_iter 20000), also reporting the paper's
                  metric (median AUROC over 100 bootstrap resamples, RandomState(42)).
--protocol tuned  (default) C tuned by patient-grouped inner CV, balanced class weights.

PAPILA (internal): binary glaucoma vs healthy, PAPILA's official patient-level 5 folds (out-of-fold
predictions); C is tuned by patient-grouped inner CV inside each training fold.
Chaksu (external): train on all of PAPILA, test on every Chaksu image without retraining; overall, per
camera and per expert agreement (5-0 / 4-1 / 3-2), plus safe vs dangerous failures.
95% CIs: 2000x cluster bootstrap (patients for PAPILA, eyes for Chaksu).

Usage:
    python run_linear_probe.py --papila-labels papila_labels.csv --papila-feats F/papila_rfg_resize.npz \
        --chaksu-labels chaksu_labels.csv --chaksu-feats F/chaksu_rfg_resize.npz --out results/frozen_lr_resize
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rfg

C_GRID = np.logspace(-4, 2, 13)


def fit_head(X, y, groups, seed, protocol='tuned', thr_rule='youden'):
    """StandardScaler + logistic regression, fitted on (X, y) only.
    tuned: C chosen by 5-fold patient-grouped inner CV on AUROC. paper: sklearn defaults.
    The decision threshold is also chosen on the training data only: Youden's J on inner
    cross-validated predictions. Returns (fitted head, C, threshold)."""
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    if protocol == 'paper':
        head, C = make_pipeline(StandardScaler(), LogisticRegression(max_iter=20_000, random_state=42)), 1.0
    else:
        base = make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight='balanced', max_iter=5000, random_state=seed))
        search = GridSearchCV(base, {'logisticregression__C': C_GRID}, scoring='roc_auc', cv=inner)
        search.fit(X, y, groups=groups)
        C = float(search.best_params_['logisticregression__C'])
        head = clone(base).set_params(logisticregression__C=C)
    inner_p = cross_val_predict(clone(head), X, y, groups=groups, cv=inner, method='predict_proba')[:, 1]
    return head.fit(X, y), C, rfg.choose_threshold(y, inner_p, thr_rule)


def paper_median_auroc(y, p, n_boot=100):
    """The paper's metric: median AUROC over 100 image-level bootstrap resamples (RandomState(42))."""
    rs, aucs = np.random.RandomState(42), []
    for _ in range(n_boot):
        idx = rs.randint(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(rfg.metrics(y[idx], p[idx])['auroc'])
    return float(np.median(aucs))


def cross_validate(X, y, groups, fold, seed, protocol, thr_rule):
    """Out-of-fold probabilities and, per image, the threshold chosen inside its training fold."""
    oof, thr = np.full(len(y), np.nan), np.full(len(y), np.nan)
    per_fold = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        assert not set(groups[tr]) & set(groups[te]), 'patient in both train and test'
        head, C, t = fit_head(X[tr], y[tr], groups[tr], seed, protocol, thr_rule)
        oof[te], thr[te] = head.predict_proba(X[te])[:, 1], t
        per_fold.append({'fold': int(k) + 1, 'n_test': int(te.sum()), 'n_pos_test': int(y[te].sum()), 'C': C,
                         'threshold': t, 'auroc': rfg.metrics(y[te], oof[te])['auroc']})
    return oof, thr, per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papila-labels', required=True)
    ap.add_argument('--papila-feats', required=True)
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--chaksu-feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--suspects', choices=['exclude', 'negative', 'positive'], default='exclude')
    ap.add_argument('--folds', choices=['official', 'random'], default='official')
    ap.add_argument('--protocol', choices=['tuned', 'paper'], default='tuned')
    ap.add_argument('--threshold', choices=['youden', 'sens90'], default='youden',
                    help="youden = max sensitivity + specificity; sens90 = 90%% sensitivity (recall) on the training data")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {'config': vars(args), 'papila': {}, 'papila_paired': {}, 'chaksu': {}}

    # ---------------- PAPILA, internal
    pap = rfg.papila_binary(args.papila_labels, args.suspects)
    Xp = rfg.load_features(pap, args.papila_feats)
    yp, gp = pap['y'].to_numpy(), pap['patient_id'].to_numpy()
    fold = rfg.papila_folds(pap, args.folds, seed=args.seed)
    clin = pap[rfg.CLINICAL_COLS].to_numpy(float)
    inputs = {'image': Xp, 'clinical (age, sex)': clin, 'image + clinical': np.hstack([Xp, clin])}
    preds = pap[['image_id', 'patient_id', 'eye', 'diagnosis', 'y']].copy()
    preds['fold'] = fold + 1
    oofs = {}
    for name, X in inputs.items():
        oofs[name], thr, per_fold = cross_validate(X, yp, gp, fold, args.seed, args.protocol, args.threshold)
        res['papila'][name] = (rfg.evaluate(yp, oofs[name], gp, args.seed, thr) | {'folds': per_fold}
                               | {'per_fold_auroc': rfg.per_fold_auroc(yp, oofs[name], fold)}
                               | {'paper_median_auroc': paper_median_auroc(yp, oofs[name])})
        preds[f'p_{name}'], preds[f'thr_{name}'] = oofs[name], thr
    preds.to_csv(out / 'papila_oof_predictions.csv', index=False)
    # compare on within-fold ranks so that folds' different probability scales don't bias the difference
    ranks = {k: rfg.rank_within_fold(v, fold) for k, v in oofs.items()}
    for a, b in [('image', 'clinical (age, sex)'), ('image + clinical', 'image')]:
        res['papila_paired'][f'{a} minus {b}'] = rfg.paired_bootstrap_diff(yp, ranks[a], ranks[b], gp, seed=args.seed)

    # ---------------- Chaksu, external (image features only)
    head, C, t = fit_head(Xp, yp, gp, args.seed, args.protocol, args.threshold)
    res['papila_full_fit_C'], res['papila_full_fit_threshold'] = C, t
    cha = rfg.chaksu_external(args.chaksu_labels)
    cha['p_glaucoma'] = head.predict_proba(rfg.load_features(cha, args.chaksu_feats))[:, 1]
    cha['pred'] = (cha['p_glaucoma'] >= t).astype(int)  # threshold frozen from PAPILA

    def ev(d):
        y, p = d['y'].to_numpy(), d['p_glaucoma'].to_numpy()
        return rfg.evaluate(y, p, d['group'], args.seed, t) | {'paper_median_auroc': paper_median_auroc(y, p)}

    res['chaksu']['All cameras'] = ev(cha)
    for cam, d in cha.groupby('camera'):
        res['chaksu'][cam] = ev(d)
    res['chaksu_by_agreement'] = {}
    for scope, d0 in [('All cameras', cha), ('Remidio', cha[cha['camera'] == 'Remidio'])]:
        for agr in ['5-0', '4-1', '3-2']:
            d = d0[d0['agreement'] == agr]
            res['chaksu_by_agreement'][f'{scope}, experts {agr}'] = ev(d)
    res['chaksu_failures'] = {'All cameras': rfg.failure_split(cha['y'], cha['p_glaucoma'], cha['n_glaucoma_votes'], t)}
    for cam, d in cha.groupby('camera'):
        res['chaksu_failures'][cam] = rfg.failure_split(d['y'], d['p_glaucoma'], d['n_glaucoma_votes'], t)
    cha[['image_id', 'camera', 'split', 'n_glaucoma_votes', 'agreement', 'y', 'p_glaucoma', 'pred']].to_csv(
        out / 'chaksu_predictions.csv', index=False)

    (out / 'metrics.json').write_text(json.dumps(res, indent=2))
    write_summary(res, args, out / 'summary.md')
    print((out / 'summary.md').read_text())


def write_summary(res, args, path):
    head = ('sklearn defaults as in the RETFound-Green paper (C = 1, no class weights)' if args.protocol == 'paper'
            else 'C tuned by inner patient-grouped CV, balanced class weights')
    L = ['# RETFound-Green: frozen backbone + logistic regression\n\n',
         f"Binary glaucoma vs healthy. PAPILA suspects: {args.suspects}. Folds: {args.folds}. Head: {head}. "
         f"Threshold: {'90% sensitivity' if args.threshold == 'sens90' else 'Youden J'} on inner cross-validated predictions of the training data (never the test fold); "
         'for Chákṣu, the threshold from all of PAPILA is frozen. 95% CIs: 2000x cluster bootstrap (patients / eyes). '
         "AUPRC [guessing] = the score of random guessing, i.e. the share of glaucoma images. "
         "Sens. at 90% spec. is read from the ROC curve. "
         "'Paper AUROC' = median of 100 image-level bootstrap resamples, the paper's metric.\n\n",
         '## PAPILA, 5-fold patient-level CV (out-of-fold, pooled)\n\n', rfg.TABLE_HEADER_PAPER]
    L += [rfg.fmt_row(k, m) for k, m in res['papila'].items()]
    L += ['\n', rfg.CM_HEADER] + [rfg.fmt_cm(k, m) for k, m in res['papila'].items()]
    L.append('\nAUROC inside each fold (mean ± SD of 5): ' + '; '.join(
        f"{k} {m['per_fold_auroc']['mean']:.3f} ± {m['per_fold_auroc']['sd']:.3f}" for k, m in res['papila'].items()))
    L.append('\n\nPaired bootstrap of the AUROC difference (scores ranked within each fold):\n\n')
    for k, d in res['papila_paired'].items():
        L.append(f"- {k}: {d['diff']:+.3f} (95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}): {d['verdict']}\n")
    L += [f"\n## Chákṣu external test (trained on all PAPILA, C = {res['papila_full_fit_C']:.3g}, "
          f"threshold {res['papila_full_fit_threshold']:.3f}; no retraining)\n\n",
          rfg.TABLE_HEADER_PAPER]
    L += [rfg.fmt_row(k, m) for k, m in res['chaksu'].items()]
    L += ['\n', rfg.CM_HEADER] + [rfg.fmt_cm(k, m) for k, m in res['chaksu'].items()] + ['\n', rfg.CM_NOTE]
    L += ['\n### By expert agreement\n\n', rfg.TABLE_HEADER_PAPER]
    L += [rfg.fmt_row(k, m) for k, m in res['chaksu_by_agreement'].items()]
    L.append('\n### Errors: safe (experts split) vs dangerous (experts unanimous)\n\n'
             '| Camera | Images | Wrong | Dangerous | Safe | Wrong, by glaucoma votes (0-5) |\n|---|---|---|---|---|---|\n')
    for k, f in res['chaksu_failures'].items():
        L.append(f"| {k} | {f['n']} | {f['n_wrong']} | {f['dangerous']} | {f['safe']} | {f['wrong_by_votes']} |\n")
    path.write_text(''.join(L))


if __name__ == '__main__':
    main()
