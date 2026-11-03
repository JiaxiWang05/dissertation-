"""Positive ablation from the finished Optuna studies: one structural choice at a time, hyperparameters still tuned.

head      (CPU)  number of hidden layers on frozen features: 0 (logistic regression), 1, 2, 3
finetune  (GPU)  number of unfrozen ViT blocks: 0, 1, 2, 4, 6, 12
For each fold, the best COMPLETE trial with that structure (inner score only) is refit on the fold's training
patients and scored on its test fold; for Chakshu the same is done with the all-PAPILA study. No test data is used
for any choice. Each structure is compared with a reference by paired bootstrap (PAPILA: scores ranked within each
fold; Chakshu: eye-level clusters): the first structure, or with --baseline a run_linear_probe.py output folder
(e.g. the tuned frozen logistic regression). Predictions are saved for later comparisons.

Usage:
    python ablation_optuna.py head --study-dir R/optuna_head_resize --papila-labels papila_labels.csv \
        --papila-feats F/papila_rfg_resize.npz --chaksu-labels chaksu_labels.csv --chaksu-feats F/chaksu_rfg_resize.npz \
        --out R/ablation_head
    python ablation_optuna.py finetune --study-dir R/optuna_finetune_resize --papila-labels papila_labels.csv \
        --chaksu-labels chaksu_labels.csv --weights W.pth --baseline R/frozen_lr_tuned_resize --out R/ablation_finetune
"""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import rfg

STRUCTURE = {'head': ('n_layers', [0, 1, 2, 3]), 'finetune': ('unfrozen_blocks', [0, 1, 2, 4, 6, 12])}


def best_by_structure(storage, study_name, key, values):
    study = optuna.load_study(study_name=study_name, storage=storage)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    out = {}
    for v in values:
        ts = [t for t in done if t.params.get(key) == v]
        if ts:
            out[v] = max(ts, key=lambda t: t.value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['head', 'finetune'])
    ap.add_argument('--study-dir', required=True)
    ap.add_argument('--papila-labels', required=True)
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--papila-feats')
    ap.add_argument('--chaksu-feats')
    ap.add_argument('--weights')
    ap.add_argument('--baseline', help='run_linear_probe.py output folder to compare every structure with')
    ap.add_argument('--predictions-from', help='earlier output folder of this script: reuse its saved predictions '
                                               '(no training) and recompute the statistics')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key, values = STRUCTURE[args.mode]
    db = 'optuna_head.db' if args.mode == 'head' else 'optuna_finetune.db'
    storage = f"sqlite:///{Path(args.study_dir) / db}"

    pap = rfg.papila_binary(args.papila_labels, 'exclude')
    y, g, fold = pap['y'].to_numpy(), pap['patient_id'].to_numpy(), rfg.papila_folds(pap, 'official')
    cha = rfg.chaksu_external(args.chaksu_labels)

    if args.predictions_from:
        pass
    elif args.mode == 'head':
        import optuna_head as oh
        Xp, Xc = rfg.load_features(pap, args.papila_feats), rfg.load_features(cha, args.chaksu_feats)

        def run(tr, te_X, params):
            return oh.ensemble(Xp[tr], y[tr], te_X, params)

        suggest = oh.suggest
    else:
        import torch
        import optuna_finetune as of
        both = pd.concat([pap.assign(src='papila'), cha.assign(src='chaksu', patient_id=cha['group'])], ignore_index=True)
        fargs = argparse.Namespace(weights=args.weights, data_root=None, preproc='resize', batch_size=16,
                                   img_size=rfg.IMG_SIZE, cache_size=rfg.IMG_SIZE, workers=args.workers, seed=0,
                                   device=rfg.pick_device())
        cache = {}
        of.warm_cache(fargs, both, cache)
        n_pap = len(pap)

        def run(tr, te_idx, params):
            return of.train_eval(fargs, both, tr, te_idx, params, cache)

        suggest = of.suggest
        torch.backends.cudnn.benchmark = True

    res = {'structure': key, 'rows': []}
    oof = {v: np.full(len(y), np.nan) for v in values}
    ext, ext_p = {}, {}
    if args.predictions_from:
        src = Path(args.predictions_from)
        sp = pd.read_csv(src / 'papila_oof_predictions.csv').set_index('image_id').loc[pap['image_id']]
        sc = pd.read_csv(src / 'chaksu_predictions.csv').set_index('image_id').loc[cha['image_id']]
        assert (sp['y'].to_numpy() == y).all() and (sp['fold'].to_numpy() == fold + 1).all(), 'saved PAPILA rows differ'
        for v in values:
            if f'p_{key}_{v}' in sp:
                oof[v] = sp[f'p_{key}_{v}'].to_numpy(dtype=float)
            if f'p_{key}_{v}' in sc:
                ext_p[v] = sc[f'p_{key}_{v}'].to_numpy(dtype=float)
                ext[v] = rfg.evaluate(cha['y'], ext_p[v], cha['group'])
    for k in [] if args.predictions_from else np.unique(fold):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        for v, trial in best_by_structure(storage, f'fold{k + 1}', key, values).items():
            params = suggest(optuna.trial.FixedTrial(trial.params))
            te_arg = Xp[te] if args.mode == 'head' else te
            oof[v][te] = run(tr, te_arg, params)
            print(f'fold {k + 1}, {key}={v}: test AUROC {rfg.metrics(y[te], oof[v][te])["auroc"]:.3f}', flush=True)
    for v, trial in [] if args.predictions_from else best_by_structure(storage, 'all_papila', key, values).items():
        params = suggest(optuna.trial.FixedTrial(trial.params))
        if args.mode == 'head':
            p = run(np.arange(len(y)), Xc, params)
        else:
            p = run(np.arange(n_pap), np.arange(n_pap, len(both)), params)
        ext[v], ext_p[v] = rfg.evaluate(cha['y'], p, cha['group']), p
        print(f'Chaksu, {key}={v}: AUROC {ext[v]["auroc"]:.3f}', flush=True)

    pd.DataFrame({'image_id': pap['image_id'], 'patient_id': g, 'fold': fold + 1, 'y': y,
                  **{f'p_{key}_{v}': oof[v] for v in values}}).to_csv(out / 'papila_oof_predictions.csv', index=False)
    pd.DataFrame({'image_id': cha['image_id'], 'camera': cha['camera'], 'y': cha['y'],
                  **{f'p_{key}_{v}': p for v, p in ext_p.items()}}).to_csv(out / 'chaksu_predictions.csv', index=False)

    yc, eyes = cha['y'].to_numpy(), cha['group'].to_numpy()
    base = values[0]
    if args.baseline:
        bl = Path(args.baseline)
        bp = pd.read_csv(bl / 'papila_oof_predictions.csv').set_index('image_id').loc[pap['image_id']]
        bc = pd.read_csv(bl / 'chaksu_predictions.csv').set_index('image_id').loc[cha['image_id']]
        assert (bp['y'].to_numpy() == y).all() and (bc['y'].to_numpy() == yc).all(), 'baseline labels differ'
        assert (bp['fold'].to_numpy() == fold + 1).all(), 'baseline does not use the official folds'
        ref_name, ref_oof, ref_cha = bl.name, bp['p_image'].to_numpy(), bc['p_glaucoma'].to_numpy()
    else:
        ref_name, ref_oof, ref_cha = f'{key} {base}', oof[base], ext_p.get(base)
    res['reference'] = ref_name
    ref_rank = None if np.isnan(ref_oof).any() else rfg.rank_within_fold(ref_oof, fold)
    missing = [d for d, r in [('PAPILA', ref_rank), ('Chákṣu', ref_cha)] if r is None]
    if missing:
        print(f"WARNING: reference {ref_name} has no result for {', '.join(missing)}: no paired comparison there")
    res['reference_missing'] = missing
    for v in values:
        ok = ~np.isnan(oof[v])
        row = {key: v, 'folds_available': int(len(np.unique(fold[ok])))}
        if ok.all():
            row['papila'] = rfg.evaluate(y, oof[v], g) | {'per_fold_auroc': rfg.per_fold_auroc(y, oof[v], fold)}
        row['chaksu'] = ext.get(v)
        vs = {}
        if ok.all() and ref_rank is not None and (args.baseline or v != base):
            vs['papila'] = rfg.paired_bootstrap_diff(y, rfg.rank_within_fold(oof[v], fold), ref_rank, g)
        if v in ext_p and ref_cha is not None and (args.baseline or v != base):
            vs['chaksu'] = rfg.paired_bootstrap_diff(yc, ext_p[v], ref_cha, eyes)
        row['minus_reference'] = vs or None
        res['rows'].append(row)
    (out / 'ablation.json').write_text(json.dumps(res, indent=2, default=float))

    d = lambda r: f"{r['diff']:+.3f} ({r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})" if r else ''  # noqa: E731
    L = [f'# Ablation: {key} (best tuned setting per structure)\n\n',
         f'Paired differences are minus `{ref_name}` (PAPILA: scores ranked within each fold; Chákṣu: eye-level '
         'bootstrap). If the 95% CI contains 0 there is no clear difference.'
         + (f" The reference has no result for {', '.join(missing)}, so those columns are empty." if missing else '')
         + '\n\n',
         f'| {key} | PAPILA AUROC (95% CI) | Per fold | PAPILA vs reference | Chákṣu AUROC (95% CI) | '
         'Chákṣu vs reference |\n', '|---|---|---|---|---|---|\n']
    for r in res['rows']:
        p, c, vs = r.get('papila'), r.get('chaksu'), r.get('minus_reference') or {}
        pcell = f"{p['auroc']:.3f} ({p['ci95']['auroc'][0]:.2f}–{p['ci95']['auroc'][1]:.2f})" if p else f"{r['folds_available']}/5 folds"
        fcell = f"{p['per_fold_auroc']['mean']:.3f} ± {p['per_fold_auroc']['sd']:.3f}" if p else ''
        ccell = f"{c['auroc']:.3f} ({c['ci95']['auroc'][0]:.2f}–{c['ci95']['auroc'][1]:.2f})" if c else ''
        L.append(f"| {r[key]} | {pcell} | {fcell} | {d(vs.get('papila'))} | {ccell} | {d(vs.get('chaksu'))} |\n")
    (out / 'summary.md').write_text(''.join(L))
    print(''.join(L))

if __name__ == '__main__':
    main()
