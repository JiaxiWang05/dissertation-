"""17 Nov task (if time): open the last 0-3 transformer blocks of RETFound-Green on Chakshu.
Model: RETFound-Green (ViT-Small, 12 blocks) loaded as in ../pipeline/rfg.py; everything frozen except the last k blocks
and the final norm (k chosen by Optuna from 0, 1, 2, 3; k = 0 is the frozen model with a head trained on images);
head: linear or one hidden layer (32 / 64 / 128). Input: plain resize to 392 x 392, mean / std 0.5, random flip and
shift. Weighted BCE with the class-weight scale searched; AdamW, head learning rate and a smaller backbone rate.
Each trial: 5-fold mean validation AUPRC at the kept epoch (lowest validation loss; patience 6, up to 30 epochs).
Same folds as before; test part unused.  WORKER=1 python unfreeze_cv.py (GPU job array; without WORKER: create study)"""
import os
import sys

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '../pipeline')
import rfg  # noqa: E402
from img_common import DEV, data, fit, storage  # noqa: E402

WEIGHTS = '/nobackup/tggq61/weights/retfoundgreen_statedict.pth'
N_TRIALS = 24
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


class Net(nn.Module):
    def __init__(self, k, hidden, dropout):
        super().__init__()
        self.backbone = rfg.load_retfound_green(WEIGHTS)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in (self.backbone.blocks[len(self.backbone.blocks) - k:] if k else []):
            for p in blk.parameters():
                p.requires_grad = True
        if k:
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        d = self.backbone.num_features
        self.head = (nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1)) if hidden == 0 else
                     nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)))

    def forward(self, x):
        return self.head(self.backbone(x))


def prep(x):
    return (x.float() / 255.0 - 0.5) / 0.5          # official RETFound-Green normalisation (mean = std = 0.5)


def run(p, X, y, splits, seed=0, fixed=None):
    """fixed: list of epochs per fold -> train exactly that many epochs, no early stopping (fair out-of-fold scores)."""
    ap, auc, epochs, oof = [], [], [], np.zeros(len(y))
    for k, (tr, va) in enumerate(splits):
        ratio = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        model = Net(p['k'], p['hidden'], p['dropout'])
        groups = [{'params': model.head.parameters(), 'lr': p['lr_head']}]
        bb = [q for q in model.backbone.parameters() if q.requires_grad]
        if bb:
            groups.append({'params': bb, 'lr': p['lr_head'] * p['lr_ratio']})
        s, e, _ = fit(model, X[tr], y[tr], X[va], y[va], prep, p['lr_head'], p['wd'], 32,
                      pos_weight=ratio * p['w_scale'], max_epochs=fixed[k] if fixed else 30, patience=6, seed=seed,
                      param_groups=groups, select=not fixed)
        oof[va] = s; ap.append(average_precision_score(y[va], s)); auc.append(roc_auc_score(y[va], s)); epochs.append(e)
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    return ap, auc, epochs, oof, n_train


def params_of(trial):
    return {'k': trial.suggest_categorical('k', [0, 1, 2, 3]),
            'hidden': trial.suggest_categorical('hidden', [0, 32, 64, 128]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.6),
            'lr_head': trial.suggest_float('lr_head', 1e-4, 3e-3, log=True),
            'lr_ratio': trial.suggest_float('lr_ratio', 0.01, 0.3, log=True),    # backbone lr = head lr x ratio
            'wd': trial.suggest_float('wd', 1e-5, 1e-1, log=True),
            'w_scale': trial.suggest_float('w_scale', 0.25, 2.0, log=True)}


if __name__ == '__main__':
    st = storage('unfreeze_chaksu')
    seed = 42 + int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    study = optuna.create_study(study_name='unfreeze_chaksu', storage=st, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if not os.environ.get('WORKER'):
        print('study ready, trials:', len(study.trials)); sys.exit()
    X, y, splits, _ = data('chaksu', 392)
    print('images', tuple(X.shape), 'glaucoma', int(y.sum()), 'device', DEV, flush=True)

    def objective(trial):
        ap, auc, epochs, _, n_train = run(params_of(trial), X, y, splits)
        trial.set_user_attr('fold_auprc', ap); trial.set_user_attr('mean_auroc', float(np.mean(auc)))
        trial.set_user_attr('kept_epochs', epochs); trial.set_user_attr('trainable', n_train)
        return float(np.mean(ap))

    study.optimize(objective, callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,))])
    b = study.best_trial
    print('best', round(b.value, 4), b.params, b.user_attrs)
