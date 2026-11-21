"""Shared code for the image-level models (17 Nov): data, folds, training loop, Optuna journal storage.
Folds are exactly those of the frozen-feature work (Chakshu: ../after_10_Nov/common.cv_split, label x camera,
Bosch pairs and copies together; FairVLMed: ../fairvlmed/common.cv_split, label), so all results are comparable.
The test parts are never loaded into training or scoring."""
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

CACHE = Path('/nobackup/tggq61/retfound_green/cache')
HERE = Path(__file__).parent
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def data(ds, size=256):
    """Images of training + validation (uint8 tensor, N x C x size x size), labels, and the 5 (train, val) index splits."""
    if ds == 'chaksu':
        cwd = os.getcwd(); os.chdir(HERE.parent / 'after_10_Nov')
        c = _load('c10', HERE.parent / 'after_10_Nov' / 'common.py')
        lab, _, y = c.load_data(); keep, splits = c.cv_split(lab); os.chdir(cwd)
        ids = np.load(CACHE / 'chaksu_ids.npy'); imgs = np.load(CACHE / f'chaksu_{size}.npy', mmap_mode='r')
    else:
        cwd = os.getcwd(); os.chdir(HERE.parent / 'fairvlmed')
        c = _load('cfv', HERE.parent / 'fairvlmed' / 'common.py')
        import pandas as pd
        lab = pd.read_csv('fairvlmed_labels.csv'); y = lab['label'].to_numpy(); keep, splits = c.cv_split(lab); os.chdir(cwd)
        ids = np.load(CACHE / 'fairvlmed_ids.npy'); imgs = np.load(CACHE / 'fairvlmed_256.npy', mmap_mode='r')
    if ds == 'chaksu' and os.environ.get('GROUPS') == 'pseudo':
        # likely fellow eyes (consecutive Remidio numbers, opposite disc sides; ../after_10_Nov/pseudo_patients.csv)
        # are kept in the same fold, on top of the Bosch pairs and copies
        import pandas as pd
        from sklearn.model_selection import StratifiedGroupKFold
        pp = pd.read_csv(HERE.parent / 'after_10_Nov' / 'pseudo_patients.csv').set_index('image_id')['pseudo_patient']
        sub = lab.loc[keep]
        strata = sub['label_majority'].astype(str) + '_' + sub['camera']
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(sub, strata,
                                                                                 sub['image_id'].map(pp)))
    pos = {k: i for i, k in enumerate(ids)}
    idx = np.array([pos[k] for k in lab.loc[keep, 'image_id']])
    X = torch.from_numpy(np.ascontiguousarray(imgs[idx]))
    return X, y[keep].astype(np.float32), splits, lab.loc[keep].reset_index(drop=True)


def augment(x):
    """Random horizontal flip and a small random shift (training only); x is a float batch on the GPU."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    dx, dy = np.random.randint(-8, 9, 2)
    return torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))


def fit(model, Xtr, ytr, Xva, yva, prep, lr, wd, batch, pos_weight, max_epochs, patience, seed, param_groups=None,
        select=True):
    """Weighted BCE; early stopping on validation loss; returns the validation scores of the kept epoch, the kept epoch
    and the history (train / val loss, val AUROC / AUPRC per epoch).
    select=False: train exactly max_epochs and return the LAST epoch's scores (the validation images then choose nothing;
    used for the fair out-of-fold comparison, with max_epochs = median kept epoch of the other folds)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV)
    opt = torch.optim.AdamW(param_groups or model.parameters(), lr=lr, weight_decay=wd)
    pw = torch.tensor([pos_weight], device=DEV)
    yt, yv = torch.tensor(ytr, device=DEV), torch.tensor(yva, device=DEV)
    best, best_state, best_epoch, wait, hist = 1e9, None, 0, 0, []

    def scores(X):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            for s in range(0, len(X), 128):
                out.append(model(prep(X[s:s + 128].to(DEV))).float().squeeze(1))
        return torch.cat(out)

    for epoch in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tl = 0.0
        for s in range(0, len(Xtr), batch):
            b = perm[s:s + batch]
            x = augment(prep(Xtr[b].to(DEV)))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
                loss = F.binary_cross_entropy_with_logits(model(x).float().squeeze(1), yt[b], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(b)
        sv = scores(Xva)
        vl = F.binary_cross_entropy_with_logits(sv, yv, pos_weight=pw).item()
        p = sv.cpu().numpy()
        hist.append({'epoch': epoch, 'train_loss': tl / len(Xtr), 'val_loss': vl,
                     'val_auroc': roc_auc_score(yva, p), 'val_auprc': average_precision_score(yva, p)})
        if not select:
            keep, best_epoch = p, epoch
            continue
        if vl < best:
            best, best_state, best_epoch, wait, keep = vl, copy.deepcopy(model.state_dict()), epoch, 0, p
        else:
            wait += 1
            if wait >= patience:
                break
    return keep, best_epoch, hist


def storage(name):
    import optuna
    from optuna.storages.journal import JournalFileBackend
    return optuna.storages.JournalStorage(JournalFileBackend(str(HERE / f'optuna_{name}.log')))
