"""Collect result folders into a few tidy CSV files for the results dashboard.

Reads every results/<run>/metrics.json (+ prediction CSVs) written by run_linear_probe.py and optuna_head.py,
and writes to OUT_DIR:
  metrics.csv   one row per run x dataset x subgroup: AUROC, AUPRC, sensitivity, ... with 95% CIs
  curves.csv    ROC and precision-recall points (main runs only)
  scores.csv    per-image probabilities for the main run (PAPILA out-of-fold, Chakshu external)
  paired.csv    paired-bootstrap AUROC differences
  failures.csv  Chakshu errors by number of expert glaucoma votes
  optuna.csv    best head per fold for the Optuna runs

Usage:
    python export_dashboard.py RESULTS_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rfg

MAIN_RUN = 'frozen_lr_paper_resize'
LABEL = {'image': 'Image features', 'clinical (age, sex)': 'Age + sex only', 'image + clinical': 'Image + age + sex',
         'Optuna head (frozen)': 'Image features', 'Logistic regression (step 1)': None}


def run_info(name):
    proto = 'paper' if '_paper_' in name else 'tuned'
    head = 'Optuna head' if name.startswith('optuna_head') else 'Logistic regression'
    if head == 'Optuna head':
        proto = 'Optuna'
    preproc = 'crop + pad' if name.endswith('fov_square') else 'resize'
    return {'run': name, 'head': head, 'protocol': proto, 'preproc': preproc,
            'run_label': f"{head}{'' if head == 'Optuna head' else f' ({proto})'}, {preproc}"}


def metric_row(info, dataset, group, setting, m):
    ci = m['ci95']
    row = info | {'dataset': dataset, 'group': group, 'setting': setting, 'n': m['n'], 'n_pos': m['n_pos'],
                  'prevalence': m['prevalence']}
    for k in rfg.METRICS:
        row[k], (row[f'{k}_lo'], row[f'{k}_hi']) = m.get(k, np.nan), ci.get(k, (np.nan, np.nan))
    for k in ['tp', 'fp', 'fn', 'tn', 'ppv_at_screening']:
        row[k] = m[k]
    pf = m.get('per_fold_auroc')
    row['auroc_fold_mean'], row['auroc_fold_sd'] = (pf['mean'], pf['sd']) if pf else (np.nan, np.nan)
    return row


def curve_rows(info, dataset, series, y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    fpr, tpr = rfg._roc_points(y, p)
    order = np.argsort(-p, kind='mergesort')
    ys, ps = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp = np.cumsum(ys)[last]
    prec, rec = tp / (last + 1), tp / ys.sum()
    out = [{'run': info['run'], 'dataset': dataset, 'series': series, 'curve': 'roc', 'x': float(a), 'y': float(b)}
           for a, b in zip(fpr, tpr)]
    out += [{'run': info['run'], 'dataset': dataset, 'series': series, 'curve': 'pr', 'x': float(a), 'y': float(b)}
            for a, b in zip(rec, prec)]
    return out


def main(results_dir, out_dir):
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, curves, paired, failures, optuna = [], [], [], [], []
    for folder in sorted(p for p in results_dir.iterdir() if (p / 'metrics.json').exists()):
        res, info = json.loads((folder / 'metrics.json').read_text()), run_info(folder.name)
        for key, m in res.get('papila', {}).items():
            if LABEL.get(key, key):
                metrics.append(metric_row(info, 'PAPILA', 'input', LABEL.get(key, key), m))
        for key, m in res.get('chaksu', {}).items():
            metrics.append(metric_row(info, 'Chákṣu', 'camera', key, m))
        for key, m in res.get('chaksu_by_agreement', {}).items():
            scope, agr = key.split(', experts ')
            metrics.append(metric_row(info, 'Chákṣu', f'agreement · {scope}', f'Experts {agr}', m))
        for key, d in res.get('papila_paired', {}).items():
            paired.append(info | {'comparison': key, 'diff': d['diff'], 'lo': d['ci95'][0], 'hi': d['ci95'][1],
                                  'verdict': d['verdict']})
        if 'paired' in res:
            d = res['paired']
            paired.append(info | {'comparison': 'Optuna head minus logistic regression', 'diff': d['diff'],
                                  'lo': d['ci95'][0], 'hi': d['ci95'][1], 'verdict': d['verdict']})
        fails = res.get('chaksu_failures', {})
        fails = fails if 'All cameras' in fails else {'All cameras': fails} if fails else {}
        for cam, f in fails.items():
            for v in range(6):
                n_wrong = f['wrong_by_votes'].get(str(v), f['wrong_by_votes'].get(v, 0))
                failures.append(info | {'camera': cam, 'votes': v, 'n_wrong': int(n_wrong),
                                        'kind': 'dangerous' if v in (0, 5) else 'safe'})
        for f in res.get('folds', []):
            if 'best' in f:
                b = f['best']
                optuna.append(info | {'fold': f['fold'], 'inner_auroc': f['inner_auroc'], 'test_auroc': f['test_auroc'],
                                      'head_type': 'logistic regression' if b['n_layers'] == 0 else f"MLP {b['n_layers']}×{b['width']}",
                                      'C': b.get('C', np.nan), 'l2': b.get('l2', np.nan)})
        if folder.name == MAIN_RUN or folder.name == MAIN_RUN.replace('paper', 'tuned'):
            pap = pd.read_csv(folder / 'papila_oof_predictions.csv')
            cha = pd.read_csv(folder / 'chaksu_predictions.csv')
            fold = pap['fold'].to_numpy()
            for key in ['image', 'clinical (age, sex)', 'image + clinical']:
                curves += curve_rows(info, 'PAPILA', LABEL[key], pap['y'], rfg.rank_within_fold(pap[f'p_{key}'], fold))
            curves += curve_rows(info, 'Chákṣu', 'All cameras', cha['y'], cha['p_glaucoma'])
            for cam, d in cha.groupby('camera'):
                curves += curve_rows(info, 'Chákṣu', cam, d['y'], d['p_glaucoma'])
            if folder.name == MAIN_RUN:
                s1 = pd.DataFrame({'dataset': 'PAPILA', 'camera': 'PAPILA', 'y': pap['y'],
                                   'p': pap['p_image'], 'threshold': pap['thr_image'], 'votes': np.nan})
                s2 = pd.DataFrame({'dataset': 'Chákṣu', 'camera': cha['camera'], 'y': cha['y'], 'p': cha['p_glaucoma'],
                                   'threshold': res['papila_full_fit_threshold'], 'votes': cha['n_glaucoma_votes']})
                pd.concat([s1, s2]).round(5).to_csv(out_dir / 'scores.csv', index=False)
    pd.DataFrame(metrics).round(5).to_csv(out_dir / 'metrics.csv', index=False)
    pd.DataFrame(curves).round(5).to_csv(out_dir / 'curves.csv', index=False)
    pd.DataFrame(paired).round(5).to_csv(out_dir / 'paired.csv', index=False)
    pd.DataFrame(failures).to_csv(out_dir / 'failures.csv', index=False)
    pd.DataFrame(optuna).round(5).to_csv(out_dir / 'optuna.csv', index=False)
    for f in sorted(out_dir.glob('*.csv')):
        print(f.name, sum(1 for _ in open(f)) - 1, 'rows', f.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
