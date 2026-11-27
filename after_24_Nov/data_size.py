"""24 Nov follow-up: how performance changes with the number of training images (learning curve over data size).
Every model keeps its own chosen setting; only the number of training images changes.
Protocol (per model):
  * the same 5 folds as every other Chakshu result; the validation fold is never subsampled, so all points of one model
    are scored on the same images
  * the training part of each fold is subsampled to 10 / 25 / 50 / 75 / 100 % by whole image groups (`eye`: Bosch
    pairs and copies stay together), separately within each label x camera stratum, so every subset keeps the
    glaucoma share and the camera mix; the subsets are nested (10 % is inside 25 % ...), one random group order per
    repeat; 3 repeats (repeat r = group order r and training seed r; at 100 % only the training seed differs)
  * trained models use the fair rule of all earlier results: pass 1 early stopping on each fold, pass 2 trains each fold
    for the median best epoch of the OTHER folds at the same size and repeat, its last epoch is scored
  * the same budget of optimiser STEPS at every size: the epoch cap and the patience are divided by the fraction
    (at 10 % an epoch is 10 x shorter), so small subsets are not cut off before they have trained
  * the settings (learning rate, etc.) were tuned on these folds at 100 %, so the 100 % point is slightly optimistic
    against the smaller ones; this is stated next to the figure
  * reported per (size, repeat): pooled out-of-fold validation AUPRC / AUROC (headline curve); for the overfitting gap
    the per-fold validation and per-fold training-subset AUPRC of the same pass-2 models (eval mode, no augmentation),
    both averaged over folds the same way; and the seconds each repeat took (computing cost)
Models: svm (frozen RETFound-Green features, 10 Nov SVM, CPU), cnn (smallest CNN, 24 Nov search), blocks2, lora,
        mask025, lora_blocks (vit_runs.py settings)
Run:    MODEL=<name> FRAC=<0.1|0.25|0.5|0.75|1.0> python data_size.py   ->  data_size/<model>_<frac>.json
        SMOKE=1 MODEL=<name> python data_size.py                         one fold, 10 %, 1 epoch, nothing saved"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data_size'
MODEL = os.environ['MODEL']
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
REPEATS = 3
SMOKE = os.environ.get('SMOKE', '') not in ('', '0', 'false')
FRAC = 0.1 if SMOKE else float(os.environ['FRAC'])
assert FRAC in FRACS, FRAC
assert os.environ.get('FOLDS') != 'pseudo', 'ordinary folds only'


def subsample(tr, lab, frac, repeat):
    """Rows of `tr` kept at this fraction: whole groups, per label x camera stratum, nested across fractions."""
    if frac >= 1.0:
        return np.sort(tr)
    sub = lab.iloc[tr]
    strata = (sub['label_majority'].astype(str) + '_' + sub['camera']).to_numpy()
    groups = sub['eye'].to_numpy()
    rng = np.random.default_rng(1000 + repeat)
    kept = []
    g_first = pd.Series(strata).groupby(groups).first()              # a group's stratum = its first image's stratum
    for s in sorted(set(strata)):
        gs = np.array(sorted(g_first.index[g_first == s]))
        order = rng.permutation(len(gs))                              # one order per repeat -> nested subsets
        n = min(len(gs), max(2, int(round(frac * len(gs)))))         # at least 2 groups per stratum
        kept.extend(gs[order[:n]])
    m = np.isin(groups, kept)
    return np.sort(tr[m])


def fair_epochs(best):
    return [max(1, int(np.median([e for j, e in enumerate(best) if j != k]))) for k in range(len(best))]


def metrics(y, s):
    return {'auprc': float(average_precision_score(y, s)), 'auroc': float(roc_auc_score(y, s))}


# ------------------------------------------------------------------ models
if MODEL == 'svm':
    os.chdir(HERE.parent / 'after_10_Nov'); sys.path.insert(0, str(HERE.parent / 'after_10_Nov'))
    import optuna
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from common import cv_split, load_data
    lab_all, Xall, yall = load_data()
    keep, splits = cv_split(lab_all)
    X, y, lab = Xall[keep], yall[keep].astype(float), lab_all.loc[keep].reset_index(drop=True)
    hp = optuna.load_study(study_name='svm_auprc', storage='sqlite:///optuna.db').best_params

    def train_fold(tr, va, k, seed):
        sc = StandardScaler().fit(X[tr])
        m = SVC(kernel=hp['kernel'], C=hp['C'], gamma=hp.get('gamma', 'scale'), class_weight='balanced').fit(sc.transform(X[tr]), y[tr])
        return m.decision_function(sc.transform(X[va])), m.decision_function(sc.transform(X[tr]))

    TRAINED = False
else:
    import torch
    sys.path.insert(0, str(HERE.parent / 'after_17_Nov')); sys.path.insert(0, str(HERE.parent / 'pipeline'))
    sys.path.insert(0, str(HERE.parent / 'after_22_Nov')); sys.path.insert(0, str(HERE))
    from img_common import CACHE, DEV, data
    from img_common import fit as fit17
    TRAINED = True
    if MODEL == 'cnn':
        import optuna
        import small_cnn as S
        from optuna.storages.journal import JournalFileBackend
        st = optuna.storages.JournalStorage(JournalFileBackend(str(HERE / 'optuna_small_cnn_chaksu.log')))
        P = dict(optuna.load_study(study_name='small_cnn_chaksu', storage=st).best_trial.params)
        X, y, splits, lab = data('chaksu', 256)

        def make(seed):
            torch.manual_seed(seed)
            return S.SmallCNN(P['blocks'], P['c0'], P['k'], P['pool'], P['dropout'])

        def fit_one(model, tr, va, seed, max_epochs, select):
            s, e, _ = fit17(model, X[tr], y[tr], X[va], y[va], S.prep, P['lr'], P['wd'], P['batch'], pos_weight=S.POS_WEIGHT,
                            max_epochs=max_epochs, patience=int(np.ceil(10 / FRAC)), seed=seed, select=select)
            return s, e

        def score(model, idx):
            model.eval(); out = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
                for s in range(0, len(idx), 128):
                    out.append(model(S.prep(X[idx[s:s + 128]].to(DEV))).float().squeeze(1))
            return torch.cat(out).cpu().numpy()
        MAX_EPOCHS = 80
    else:
        os.environ['MODEL'] = 'blocks2' if MODEL == 'blocks2' else MODEL     # vit_runs reads MODEL at import
        import vit_runs as V
        import unfreeze_cv
        X, y, splits, lab = data('chaksu', 392)
        ids = np.load(CACHE / 'chaksu_ids.npy'); pos = {k: i for i, k in enumerate(ids)}
        M = torch.from_numpy(np.ascontiguousarray(np.load(CACHE / 'chaksu_mask392.npy', mmap_mode='r')[[pos[k] for k in lab['image_id']]]))
        PB = V.P_BLK
        MAX_EPOCHS = 30

        def make(seed):
            torch.manual_seed(seed)
            if MODEL == 'blocks2':
                return unfreeze_cv.Net(PB['k'], PB['hidden'], PB['dropout'])
            return V.build(seed)

        def fit_one(model, tr, va, seed, max_epochs, select):
            ratio = (y[tr] == 0).sum() / (y[tr] == 1).sum()
            if MODEL == 'blocks2':
                g = [{'params': model.head.parameters(), 'lr': PB['lr_head']},
                     {'params': [q for q in model.backbone.parameters() if q.requires_grad], 'lr': PB['lr_head'] * PB['lr_ratio']}]
                s, e, _ = fit17(model, X[tr], y[tr], X[va], y[va], unfreeze_cv.prep, PB['lr_head'], PB['wd'], 32,
                                ratio * PB['w_scale'], max_epochs, int(np.ceil(6 / FRAC)), seed, param_groups=g, select=select)
            else:
                s, e, _ = V.fit_hist(model, X[tr], M[tr], y[tr], X[va], y[va], ratio * V.P_LORA['w_scale'], max_epochs, int(np.ceil(6 / FRAC)),
                                     seed, select)
            return s, e

        def score(model, idx):
            model.eval(); out = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for s in range(0, len(idx), 64):
                    x = X[idx[s:s + 64]].to(DEV)
                    if MODEL == 'blocks2':
                        out.append(model(unfreeze_cv.prep(x)).float().squeeze(1))
                    else:
                        out.append(model((x.float() / 255 - 0.5) / 0.5)[0].float().squeeze(1))
            return torch.cat(out).cpu().numpy()


CAP = int(np.ceil(MAX_EPOCHS / FRAC)) if TRAINED else 0     # same number of optimiser steps at every size


def run_repeat(r, folds):
    """One repeat at FRAC: pooled validation scores and per-fold training-subset metrics."""
    t0 = time.time()
    subs = [subsample(tr, lab, FRAC, r) for tr, va in folds]
    oof, train_m, info = np.full(len(y), np.nan), [], {}
    if not TRAINED:
        for k, ((tr, va), t) in enumerate(zip(folds, subs)):
            sv, st = train_fold(t, va, k, r)
            oof[va] = sv; train_m.append(metrics(y[t], st))
    else:
        best = []
        for k, ((tr, va), t) in enumerate(zip(folds, subs)):     # pass 1: early stopping (only to choose epochs)
            seed = r * 100 + k
            _, e = fit_one(make(seed), t, va, seed, 1 if SMOKE else CAP, True)
            best.append(e)
        fixed = [1] * len(folds) if SMOKE else fair_epochs(best)
        for k, ((tr, va), t) in enumerate(zip(folds, subs)):     # pass 2: fixed epochs, last epoch scored
            seed = r * 100 + k
            model = make(seed)
            sv, _ = fit_one(model, t, va, seed, fixed[k], False)
            oof[va] = sv; train_m.append(metrics(y[t], score(model, t)))
            del model
            torch.cuda.empty_cache()
        info = {'best_epochs': [int(e) for e in best], 'fixed_epochs': [int(e) for e in fixed], 'epoch_cap': CAP,
                'hit_cap': int(sum(e >= CAP for e in best))}
    va_all = np.concatenate([va for _, va in folds])
    return {'repeat': r, 'n_train': [int(len(t)) for t in subs], 'n_pos': [int(y[t].sum()) for t in subs],
            'val': metrics(y[va_all], oof[va_all]),
            'val_folds': [metrics(y[va], oof[va]) for _, va in folds],
            'train': {m: float(np.mean([d[m] for d in train_m])) for m in ('auprc', 'auroc')},
            'train_folds': train_m, 'val_fold_mean': {m: float(np.mean([metrics(y[va], oof[va])[m] for _, va in folds]))
                                                         for m in ('auprc', 'auroc')},
            'seconds': round(time.time() - t0, 1), **info}


if __name__ == '__main__':
    t0 = time.time()
    if SMOKE:
        tr, va = splits[0]
        t = subsample(tr, lab, 0.1, 0)
        t2 = subsample(tr, lab, 0.25, 0)
        assert set(t) <= set(t2), 'subsets not nested'
        assert not set(lab['eye'].iloc[t]) & set(lab['eye'].iloc[np.setdiff1d(tr, t)]), 'group split'
        res = run_repeat(0, [splits[0]])
        print('smoke ok', MODEL, 'n_train', res['n_train'], 'pos', res['n_pos'], 'val', res['val'], 'train', res['train'],
              round(time.time() - t0), 's'); sys.exit()
    OUT.mkdir(exist_ok=True)
    reps = []
    for r in range(REPEATS):                                  # saved after every repeat, so a stopped job keeps its work
        reps.append(run_repeat(r, splits))
        json.dump({'model': MODEL, 'frac': FRAC, 'prevalence_val': float(y.mean()), 'n_full': [int(len(tr)) for tr, _ in splits],
                   'device': (torch.cuda.get_device_name(0) if TRAINED and DEV == 'cuda' else 'cpu'), 'repeats': reps,
                   'seconds': round(time.time() - t0)}, open(OUT / f'{MODEL}_{FRAC}.json', 'w'))
    for d in reps:
        print(MODEL, FRAC, 'repeat', d['repeat'], 'n', d['n_train'], 'val AUPRC', round(d['val']['auprc'], 4),
              'train AUPRC', round(d['train']['auprc'], 4), d.get('fixed_epochs', ''), flush=True)
    print('saved', MODEL, FRAC, round(time.time() - t0), 's')
