"""17 Nov task: own small CNN trained from scratch on the images (no foundation model) = the baseline.
Optuna chooses the number of convolution blocks, channels, kernel size, dropout, learning rate, weight decay, batch size
(and on Chakshu the class-weight scale); each trial is scored by the 5-fold mean validation AUPRC at the kept epoch
(lowest validation loss; patience 10, up to 80 epochs). Same folds as the frozen-feature work; test parts unused.
Workers: WORKER=1 python cnn_cv.py chaksu|fairvlmed (job array, shared journal); without WORKER: create the study."""
import os
import sys

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from img_common import DEV, data, fit, storage

DS = sys.argv[1]
N_TRIALS = {'chaksu': 60, 'fairvlmed': 30}[DS]


class CNN(nn.Module):
    """blocks x [conv k x k -> batch norm -> ReLU -> max-pool 2], channels doubling from c0; global average pool;
    dropout; one output (the glaucoma score)."""
    def __init__(self, in_ch, blocks, c0, k, dropout):
        super().__init__()
        layers, c = [], in_ch
        for i in range(blocks):
            out = c0 * 2 ** i
            layers += [nn.Conv2d(c, out, k, padding=k // 2, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, 1))

    def forward(self, x):
        return self.head(self.features(x))


def prep(x):
    return x.float() / 255.0 * 2 - 1        # uint8 -> [-1, 1]


def params_of(trial):
    p = {'blocks': trial.suggest_int('blocks', 2, 5), 'c0': trial.suggest_categorical('c0', [8, 16, 32]),
         'k': trial.suggest_categorical('k', [3, 5]), 'dropout': trial.suggest_float('dropout', 0.0, 0.5),
         'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True), 'wd': trial.suggest_float('wd', 1e-5, 1e-1, log=True),
         'batch': trial.suggest_categorical('batch', [16, 32, 64])}
    p['w_scale'] = trial.suggest_float('w_scale', 0.25, 2.0, log=True) if DS == 'chaksu' else 1.0
    return p


def run(p, X, y, splits, seed=0):
    ap, auc, epochs, oof = [], [], [], np.zeros(len(y))
    from sklearn.metrics import average_precision_score, roc_auc_score
    for tr, va in splits:
        ratio = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        model = CNN(X.shape[1], p['blocks'], p['c0'], p['k'], p['dropout'])
        s, e, _ = fit(model, X[tr], y[tr], X[va], y[va], prep, p['lr'], p['wd'], p['batch'],
                      pos_weight=ratio * p['w_scale'], max_epochs=80, patience=10, seed=seed)
        oof[va] = s; ap.append(average_precision_score(y[va], s)); auc.append(roc_auc_score(y[va], s)); epochs.append(e)
    n_par = sum(q.numel() for q in model.parameters())
    return ap, auc, epochs, oof, n_par


if __name__ == '__main__':
    st = storage(f'cnn_{DS}')
    seed = 42 + int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    study = optuna.create_study(study_name=f'cnn_{DS}', storage=st, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if not os.environ.get('WORKER'):
        print('study ready, trials:', len(study.trials)); sys.exit()
    X, y, splits, _ = data(DS, 256)
    print(DS, 'images', tuple(X.shape), 'glaucoma', int(y.sum()), 'device', DEV, flush=True)

    def objective(trial):
        ap, auc, epochs, _, n_par = run(params_of(trial), X, y, splits)
        trial.set_user_attr('fold_auprc', ap); trial.set_user_attr('mean_auroc', float(np.mean(auc)))
        trial.set_user_attr('kept_epochs', epochs); trial.set_user_attr('parameters', n_par)
        return float(np.mean(ap))

    study.optimize(objective, callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,))])
    b = study.best_trial
    print('best', round(b.value, 4), b.params, b.user_attrs)
