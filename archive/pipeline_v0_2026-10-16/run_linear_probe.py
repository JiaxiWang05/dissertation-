"""Frozen-feature logistic regression: PAPILA (internal, patient-grouped CV) -> Chaksu (external, per camera).

Task: binary glaucoma (1) vs healthy (0). PAPILA suspects are excluded by default.
Backbone is fully frozen; the head is StandardScaler + logistic regression, with C chosen by
patient-grouped inner CV. 95% CIs come from a 1000x cluster bootstrap (resampling patients / eyes).

Usage:
    python run_linear_probe.py --papila-labels papila_labels.csv --papila-feats features/papila_rfg.npz \
        --chaksu-labels chaksu_labels.csv --chaksu-feats features/chaksu_rfg.npz --out results/rfg_frozen_lr
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = np.logspace(-4, 2, 13)
CLINICAL_COLS = ['age', 'gender']  # non-diagnostic only; IOP / VF / CDR would leak the label


def load(labels_csv, feats_npz):
    df = pd.read_csv(labels_csv)
    z = np.load(feats_npz)
    order = pd.Series(np.arange(len(z['image_id'])), index=z['image_id'])
    X = z['features'][order.loc[df['image_id']].to_numpy()]
    return df.reset_index(drop=True), X


def make_head(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=5000,
                                                              random_state=seed))


def fit_head(X, y, groups, seed):
    """Choose C by 5-fold patient-grouped inner CV on AUROC, then refit on all of X."""
    search = GridSearchCV(make_head(seed), {'logisticregression__C': C_GRID}, scoring='roc_auc',
                          cv=GroupKFold(n_splits=5))
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_['logisticregression__C']


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {'n': int(len(y)), 'n_pos': int(y.sum()),
            'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p),
            'balanced_acc': balanced_accuracy_score(y, yhat),
            'sensitivity': recall_score(y, yhat, pos_label=1),
            'specificity': recall_score(y, yhat, pos_label=0),
            'f1': f1_score(y, yhat)}


def bootstrap_ci(y, p, groups, n_boot=1000, seed=0):
    """Cluster bootstrap: resample whole groups (patients/eyes) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metrics(y[idx], p[idx]))
    out = {}
    for k in ['auroc', 'auprc', 'balanced_acc', 'sensitivity', 'specificity', 'f1']:
        v = np.array([s[k] for s in stats])
        out[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def evaluate(y, p, groups, seed):
    m = metrics(y, p)
    ci = bootstrap_ci(y, p, groups, seed=seed)
    return {k: (float(v) if isinstance(v, float) else v) for k, v in m.items()} | {'ci95': ci}


def papila_cv(X, y, groups, n_folds, seed):
    """Outer patient-grouped stratified CV. Returns out-of-fold probabilities and per-fold info."""
    oof = np.full(len(y), np.nan)
    folds = []
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        assert not set(groups[tr]) & set(groups[te]), 'patient leakage between folds'
        head, C = fit_head(X[tr], y[tr], groups[tr], seed)
        oof[te] = head.predict_proba(X[te])[:, 1]
        folds.append({'fold': k, 'n_test': int(len(te)), 'n_pos_test': int(y[te].sum()), 'C': float(C),
                      'auroc': float(roc_auc_score(y[te], oof[te]))})
    return oof, folds


def fmt(m):
    ci = m['ci95']
    return ' | '.join(f"{m[k]:.3f} ({ci[k][0]:.3f}-{ci[k][1]:.3f})"
                      for k in ['auroc', 'auprc', 'balanced_acc', 'sensitivity', 'specificity'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papila-labels', required=True)
    ap.add_argument('--papila-feats', required=True)
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--chaksu-feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--suspects', choices=['exclude', 'negative'], default='exclude')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- PAPILA ----
    pap, Xp = load(args.papila_labels, args.papila_feats)
    if args.suspects == 'exclude':
        keep = pap['diagnosis'] != 'suspect'
        pap, Xp = pap[keep].reset_index(drop=True), Xp[keep.to_numpy()]
    yp = (pap['diagnosis'] == 'glaucoma').astype(int).to_numpy()
    gp = pap['patient_id'].to_numpy()

    inputs = {'image': Xp,
              'clinical': pap[CLINICAL_COLS].to_numpy(float),
              'image+clinical': np.hstack([Xp, pap[CLINICAL_COLS].to_numpy(float)])}
    results = {'config': vars(args), 'papila': {}, 'chaksu': {}}
    preds = pap[['image_id', 'patient_id', 'eye', 'diagnosis']].copy()
    for name, X in inputs.items():
        oof, folds = papila_cv(X, yp, gp, args.folds, args.seed)
        results['papila'][name] = evaluate(yp, oof, gp, args.seed) | {'folds': folds}
        preds[f'p_{name}'] = oof
    preds.to_csv(out / 'papila_oof_predictions.csv', index=False)

    # ---- Chaksu (external, no retraining; image features only) ----
    head, C = fit_head(Xp, yp, gp, args.seed)
    results['papila_full_fit_C'] = float(C)
    cha, Xc = load(args.chaksu_labels, args.chaksu_feats)
    keep = cha['duplicate_of'].isna().to_numpy()
    cha, Xc = cha[keep].reset_index(drop=True), Xc[keep]
    yc = cha['label_majority'].astype(int).to_numpy()
    gc = cha['bosch_group'].fillna(cha['image_id']).to_numpy()  # paired Bosch images share an eye
    cha['p_glaucoma'] = head.predict_proba(Xc)[:, 1]
    results['chaksu']['all'] = evaluate(yc, cha['p_glaucoma'].to_numpy(), gc, args.seed)
    for cam, d in cha.groupby('camera'):
        results['chaksu'][cam] = evaluate(d['label_majority'].to_numpy(int), d['p_glaucoma'].to_numpy(),
                                          gc[d.index], args.seed)

    # Failure analysis: dangerous = experts unanimous (0/5 or 5/5) but model wrong; safe = experts split.
    cha['pred'] = (cha['p_glaucoma'] >= 0.5).astype(int)
    wrong = cha[cha['pred'] != cha['label_majority']]
    unanimous = wrong['n_glaucoma_votes'].isin([0, 5])
    results['chaksu_failures'] = {'n_wrong': int(len(wrong)), 'dangerous': int(unanimous.sum()),
                                  'safe': int((~unanimous).sum()),
                                  'by_votes': {int(k): int(v) for k, v in
                                               wrong['n_glaucoma_votes'].value_counts().sort_index().items()}}
    cha[['image_id', 'camera', 'split', 'n_glaucoma_votes', 'label_majority', 'p_glaucoma', 'pred']].to_csv(
        out / 'chaksu_predictions.csv', index=False)

    (out / 'metrics.json').write_text(json.dumps(results, indent=2))

    hdr = '| Setting | n (pos) | AUROC | AUPRC | Bal. acc | Sens | Spec |\n|---|---|---|---|---|---|---|\n'
    lines = [f'# RETFound-Green, frozen backbone + logistic regression\n',
             f'Binary glaucoma vs healthy; PAPILA suspects: {args.suspects}. Threshold 0.5 (balanced class weights). '
             f'95% CIs: 1000x cluster bootstrap.\n',
             f'## PAPILA, {args.folds}-fold patient-grouped CV (out-of-fold)\n', hdr]
    for name, m in results['papila'].items():
        lines.append(f"| {name} | {m['n']} ({m['n_pos']}) | " + fmt(m) + ' |\n')
    lines += [f'\n## Chaksu external test (trained on all PAPILA, C={C:.3g}; no retraining)\n', hdr]
    for cam, m in results['chaksu'].items():
        lines.append(f"| {cam} | {m['n']} ({m['n_pos']}) | " + fmt(m) + ' |\n')
    f = results['chaksu_failures']
    lines.append(f"\nChaksu errors: {f['n_wrong']} — dangerous (experts unanimous) {f['dangerous']}, "
                 f"safe (experts split) {f['safe']}; by glaucoma votes: {f['by_votes']}\n")
    (out / 'summary.md').write_text(''.join(lines))
    print(''.join(lines))


if __name__ == '__main__':
    main()
