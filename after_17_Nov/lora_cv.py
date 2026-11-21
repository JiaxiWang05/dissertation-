"""LoRA on RETFound-Green (after the 17 Nov meeting): adapt all 12 blocks with low-rank add-ons instead of opening blocks.
Every pre-trained weight stays frozen. Each chosen linear layer W (attention qkv and proj; optionally the MLP fc1 and
fc2) gets a trainable low-rank update:  y = W x + (alpha / r) * B(A(dropout(x))),  A: in -> r, B: r -> out, B starts at
zero, so training starts exactly from the pre-trained model. Head: linear (as chosen in the unfreezing search).
Optuna chooses the rank r (4 / 8 / 16), alpha / r (1 or 2), the target layers, the LoRA and head learning rates,
LoRA dropout, head dropout, weight decay and the class-weight scale. Same folds, images (392 px, flip and shift),
training loop and scoring as unfreeze_cv.py; test part unused.
WORKER=1 python lora_cv.py (GPU job array; without WORKER: create the study)"""
import math
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
N_TRIALS = 40
TARGETS = {'attn': ('attn.qkv', 'attn.proj'), 'all': ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')}


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update (Hu et al., 2021)."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)                       # update = 0 at the start
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(self.drop(x)))


class Net(nn.Module):
    def __init__(self, r, alpha_ratio, targets, lora_dropout, dropout):
        super().__init__()
        self.backbone = rfg.load_retfound_green(WEIGHTS)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks:
            for name in TARGETS[targets]:
                parent_name, child = name.split('.')
                parent = getattr(blk, parent_name)
                setattr(parent, child, LoRALinear(getattr(parent, child), r, r * alpha_ratio, lora_dropout))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.num_features, 1))

    def forward(self, x):
        return self.head(self.backbone(x))


def prep(x):
    return (x.float() / 255.0 - 0.5) / 0.5          # official RETFound-Green normalisation (mean = std = 0.5)


def run(p, X, y, splits, seed=0, fixed=None):
    """fixed: list of epochs per fold -> train exactly that many epochs, no early stopping (fair out-of-fold scores)."""
    ap, auc, epochs, oof = [], [], [], np.zeros(len(y))
    for k, (tr, va) in enumerate(splits):
        ratio = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        torch.manual_seed(seed * 100 + k)               # reproducible LoRA A and head initialisation
        model = Net(p['r'], p['alpha_ratio'], p['targets'], p['lora_dropout'], p['dropout'])
        lora = [q for n, q in model.backbone.named_parameters() if q.requires_grad]
        groups = [{'params': model.head.parameters(), 'lr': p['lr_head']}, {'params': lora, 'lr': p['lr_lora']}]
        s, e, _ = fit(model, X[tr], y[tr], X[va], y[va], prep, p['lr_head'], p['wd'], 32,
                      pos_weight=ratio * p['w_scale'], max_epochs=fixed[k] if fixed else 30, patience=6, seed=seed,
                      param_groups=groups, select=not fixed)
        oof[va] = s; ap.append(average_precision_score(y[va], s)); auc.append(roc_auc_score(y[va], s)); epochs.append(e)
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    return ap, auc, epochs, oof, n_train


def params_of(trial):
    return {'r': trial.suggest_categorical('r', [4, 8, 16]),
            'alpha_ratio': trial.suggest_categorical('alpha_ratio', [1, 2]),
            'targets': trial.suggest_categorical('targets', ['attn', 'all']),
            'lora_dropout': trial.suggest_float('lora_dropout', 0.0, 0.3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.6),
            'lr_head': trial.suggest_float('lr_head', 1e-4, 3e-3, log=True),
            'lr_lora': trial.suggest_float('lr_lora', 1e-5, 2e-3, log=True),
            'wd': trial.suggest_float('wd', 1e-5, 1e-1, log=True),
            'w_scale': trial.suggest_float('w_scale', 0.25, 2.0, log=True)}


if __name__ == '__main__':
    st = storage('lora_chaksu')
    seed = 42 + int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    study = optuna.create_study(study_name='lora_chaksu', storage=st, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if os.environ.get('SMOKE'):                      # one fold, two epochs: checks shapes, memory and trainable count
        X, y, splits, _ = data('chaksu', 392)
        p = {'r': 8, 'alpha_ratio': 2, 'targets': 'all', 'lora_dropout': 0.1, 'dropout': 0.3, 'lr_head': 5e-4,
             'lr_lora': 5e-4, 'wd': 1e-2, 'w_scale': 0.3}
        ap, auc, ep, _, n = run(p, X, y, splits[:1], fixed=[2])
        print('smoke ok: fold-1 AUPRC', round(ap[0], 3), 'trainable', n,
              'GPU memory GB', round(torch.cuda.max_memory_allocated() / 1e9, 1)); sys.exit()
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
