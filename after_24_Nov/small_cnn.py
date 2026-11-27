"""24 Nov meeting: the CNN from scratch made as small as possible (Raheleh: one or two convolution blocks, 8 or 16
channels), class weight fixed at 4 (the weight chosen in the 17 Nov weight study), training curves.
Search space (Optuna, 5-fold mean AUPRC, same folds and images as ../after_17_Nov/cnn_cv.py): 1-2 blocks, first
channels 8 / 16 (doubling), kernel 3 / 5 / 7, head = global average pool or a 4 x 4 average pool, dropout, lr, wd, batch.
Modes:  python small_cnn.py               create the study
        WORKER=1 python small_cnn.py      Optuna worker (job array)
        FINAL=1 python small_cnn.py       fair score of the best setting (median best epoch of the other folds, 3 seeds),
                                          paired difference to the frozen SVM -> small_cnn_final.csv, small_cnn_oof.npz
        CURVES=1 python small_cnn.py      best setting, seed 0, every fold trained 80 epochs without stopping, per-epoch
                                          train / validation loss and validation AUPRC -> curves_small_cnn.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'after_17_Nov'))
from img_common import DEV, data, fit  # noqa: E402
from optuna.storages.journal import JournalFileBackend  # noqa: E402
assert os.environ.get('FOLDS') != 'pseudo', 'small_cnn.py uses the ordinary folds only'

N_TRIALS = 40
POS_WEIGHT = 4.0


class SmallCNN(nn.Module):
    def __init__(self, blocks, c0, k, pool, dropout):
        super().__init__()
        layers, c = [], 3
        for i in range(blocks):
            out = c0 * 2 ** i
            layers += [nn.Conv2d(c, out, k, padding=k // 2, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        g = 1 if pool == 'gap' else 4
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(g), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c * g * g, 1))

    def forward(self, x):
        return self.head(self.features(x))


def prep(x):
    return x.float() / 255.0 * 2 - 1


def params_of(trial):
    return {'blocks': trial.suggest_int('blocks', 1, 2), 'c0': trial.suggest_categorical('c0', [8, 16]),
            'k': trial.suggest_categorical('k', [3, 5, 7]), 'pool': trial.suggest_categorical('pool', ['gap', 'p4']),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5), 'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
            'wd': trial.suggest_float('wd', 1e-5, 1e-1, log=True), 'batch': trial.suggest_categorical('batch', [16, 32, 64])}


def run(p, X, y, splits, seed=0, fixed=None, select=None, hist_out=None):
    oof, ap, epochs = np.zeros(len(y)), [], []
    for k, (tr, va) in enumerate(splits):
        torch.manual_seed(seed * 100 + k)
        model = SmallCNN(p['blocks'], p['c0'], p['k'], p['pool'], p['dropout'])
        s, e, h = fit(model, X[tr], y[tr], X[va], y[va], prep, p['lr'], p['wd'], p['batch'], pos_weight=POS_WEIGHT,
                      max_epochs=fixed[k] if fixed else 80, patience=10, seed=seed,
                      select=(not fixed) if select is None else select)
        oof[va] = s; ap.append(average_precision_score(y[va], s)); epochs.append(e)
        if hist_out is not None:
            hist_out.append(h)
    n_par = sum(q.numel() for q in model.parameters())
    return ap, epochs, oof, n_par


if __name__ == '__main__':
    st = optuna.storages.JournalStorage(JournalFileBackend(str(HERE / 'optuna_small_cnn_chaksu.log')))
    seed = 42 + int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    study = optuna.create_study(study_name='small_cnn_chaksu', storage=st, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if not any(os.environ.get(m) for m in ('WORKER', 'FINAL', 'CURVES', 'SMOKE')):
        print('study ready, trials:', len(study.trials)); sys.exit()
    X, y, splits, lab = data('chaksu', 256)
    if os.environ.get('SMOKE'):
        p = {'blocks': 2, 'c0': 8, 'k': 5, 'pool': 'p4', 'dropout': 0.2, 'lr': 1e-3, 'wd': 1e-3, 'batch': 32}
        ap, ep, _, n = run(p, X, y, splits[:1], fixed=[2])
        print('smoke ok', round(ap[0], 3), 'parameters', n); sys.exit()
    if os.environ.get('WORKER'):
        def objective(trial):
            ap, epochs, _, n_par = run(params_of(trial), X, y, splits)
            trial.set_user_attr('kept_epochs', epochs); trial.set_user_attr('parameters', n_par)
            return float(np.mean(ap))
        study.optimize(objective, callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,))])
        sys.exit()
    p = dict(study.best_trial.params)
    print('best setting', p, 'search AUPRC', round(study.best_value, 4), 'trials', len(study.trials), flush=True)
    if os.environ.get('CURVES'):
        hist = []
        run(p, X, y, splits, seed=0, fixed=[80] * 5, select=False, hist_out=hist)
        json.dump({'model': 'small CNN', 'setting': p, 'folds': hist}, open(HERE / 'curves_small_cnn.json', 'w'))
        print('saved curves'); sys.exit()
    # FINAL: fair score, 3 seeds, paired difference to the frozen SVM (as ../after_17_Nov/final_compare.py)
    sys.path.insert(0, str(HERE.parent / 'after_10_Nov'))
    from common import bootstrap_ci, paired_difference, threshold_at_recall  # noqa: E402
    ref = pd.read_csv(HERE.parent / 'after_10_Nov' / 'step5_oof_scores.csv', float_precision='round_trip')
    assert (ref['image_id'].to_numpy() == lab['image_id'].to_numpy()).all()
    groups = lab['eye'].to_numpy()
    seeds, es = [], []
    for sd in range(3):
        _, epochs, o_es, n_par = run(p, X, y, splits, seed=sd)
        fixed = [max(1, int(np.median([e for j, e in enumerate(epochs) if j != k]))) for k in range(5)]
        _, _, o, _ = run(p, X, y, splits, seed=sd, fixed=fixed)
        seeds.append(o); es.append(o_es)
        print('seed', sd, 'fair AUPRC', round(average_precision_score(y, o), 4), 'AUROC', round(roc_auc_score(y, o), 4),
              '| epochs', epochs, 'fixed', fixed, flush=True)
    s0 = seeds[0]; t = threshold_at_recall(y, s0, 0.9)
    d, ci = paired_difference(average_precision_score, y, s0, ref['svm'].to_numpy(), groups)
    row = {'setting': p, 'parameters': n_par, 'search AUPRC': study.best_value, 'trials': len(study.trials),
           'AUPRC 3 seeds mean': np.mean([average_precision_score(y, o) for o in seeds]),
           'AUPRC range': [round(min(average_precision_score(y, o) for o in seeds), 3), round(max(average_precision_score(y, o) for o in seeds), 3)],
           'AUROC seed 0': roc_auc_score(y, s0), 'AUROC CI': np.round(bootstrap_ci(roc_auc_score, y, s0, groups), 3).tolist(),
           'minus frozen SVM AUPRC': d, 'CI': [round(v, 3) for v in ci],
           'healthy flagged % at 90 % sens': 100 * ((s0 >= t) & (y == 0)).sum() / (y == 0).sum()}
    pd.DataFrame([row]).to_csv(HERE / 'small_cnn_final.csv', index=False)
    np.savez(HERE / 'small_cnn_oof.npz', image_id=lab['image_id'].to_numpy().astype(str), y=y, **{f'seed{i}': o for i, o in enumerate(seeds)})
    print(row)
