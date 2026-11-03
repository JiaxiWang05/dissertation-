"""Step 2b: Optuna over how many RETFound-Green blocks to unfreeze and the head on top (GPU, Hamilton).

Hyperparameters searched (as discussed on 16 Oct):
  unfrozen blocks: 0, 1, 2, 4, 6 or all 12 (ViT-S has 12 transformer blocks; 12 = full fine-tuning)
  head: 0-2 hidden layers, width 128/256/512, dropout
  learning rates (backbone, head), weight decay, epochs
Nested, so the PAPILA test folds are never used for tuning:
  outer = PAPILA's official patient-level 5 folds
  inner = one patient-grouped, stratified 80/20 split of the outer-training patients (Optuna maximises
          validation AUROC at the last epoch; weak trials are pruned early)
The best setting is retrained on the whole outer-training set and scored on the outer test fold.
Progress is saved after every fold and every trial (SQLite), so a killed job resumes where it stopped.

Usage on Hamilton (see run_optuna_finetune.slurm):
    python optuna_finetune.py --papila-labels papila_labels.csv --weights W.pth --out R/optuna_finetune
Quick local smoke test (CPU/MPS, tiny):
    python optuna_finetune.py ... --data-root ~/data/dissertation --smoke
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

import rfg

BLOCK_CHOICES = [0, 1, 2, 4, 6, 12]


class Classifier(nn.Module):
    def __init__(self, backbone, n_layers, width, dropout):
        super().__init__()
        self.backbone = backbone
        layers, d = [], backbone.num_features
        for _ in range(n_layers):
            layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]
            d = width
        layers += [nn.Dropout(dropout), nn.Linear(d, 1)] if n_layers == 0 else [nn.Linear(d, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def build_model(args, p):
    backbone = rfg.load_retfound_green(args.weights, dynamic_img_size=args.img_size != rfg.IMG_SIZE)
    for prm in backbone.parameters():
        prm.requires_grad = False
    k = p['unfrozen_blocks']
    if k >= len(backbone.blocks):
        for prm in backbone.parameters():
            prm.requires_grad = True
    elif k > 0:
        for blk in backbone.blocks[-k:]:
            for prm in blk.parameters():
                prm.requires_grad = True
        for prm in backbone.norm.parameters():
            prm.requires_grad = True
    return Classifier(backbone, p['n_layers'], p.get('width', 0), p['dropout'])


def suggest(trial):
    p = {'unfrozen_blocks': trial.suggest_categorical('unfrozen_blocks', BLOCK_CHOICES),
         'n_layers': trial.suggest_int('n_layers', 0, 2),
         'dropout': trial.suggest_float('dropout', 0.0, 0.5),
         'lr_head': trial.suggest_float('lr_head', 1e-4, 3e-3, log=True),
         'l2_head': trial.suggest_float('l2_head', 1e-5, 1.0, log=True),
         'epochs': trial.suggest_int('epochs', 5, 20, step=5)}
    if p['n_layers'] > 0:
        p['width'] = trial.suggest_categorical('width', [128, 256, 512])
    if p['unfrozen_blocks'] > 0:
        p['lr_backbone'] = trial.suggest_float('lr_backbone', 1e-6, 1e-4, log=True)
        p['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    return p


def loaders(args, df, tr_idx, te_idx, cache):
    paths = rfg.local_paths(df['path'], args.data_root)
    y = df['y'].to_numpy()
    tr = rfg.FundusImages([paths[i] for i in tr_idx], rfg.train_transform(args.img_size), args.preproc, y[tr_idx])
    te = rfg.FundusImages([paths[i] for i in te_idx], rfg.eval_transform(args.img_size), args.preproc, y[te_idx])
    # share the decoded-image cache between trials (keyed by row index in df)
    for ds, idx in [(tr, tr_idx), (te, te_idx)]:
        ds.cache_size = args.cache_size
        ds.cache = _CacheView(cache, idx)
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=False)
    return DataLoader(tr, shuffle=True, drop_last=len(tr) > args.batch_size, **kw), DataLoader(te, **kw)


class _CacheView(dict):
    """Maps a dataset's local index to the shared cache keyed by row index in the full table."""

    def __init__(self, cache, idx):
        super().__init__()
        self.cache, self.idx = cache, list(idx)

    def __contains__(self, i):
        return self.idx[i] in self.cache

    def __getitem__(self, i):
        return self.cache[self.idx[i]]

    def __setitem__(self, i, v):
        self.cache[self.idx[i]] = v


def warm_cache(args, df, cache):
    """Decode every image once (crop/pad if fov_square, resize to cache_size) before training starts, so
    DataLoader workers forked later share the decoded copies instead of re-reading JPEGs every epoch."""
    paths = rfg.local_paths(df['path'], args.data_root)
    ds = rfg.FundusImages(paths, None, args.preproc, cache_size=args.cache_size)
    t0 = time.time()
    for i in range(len(ds)):
        if i not in cache:
            cache[i] = ds._load(i)
    print(f'decoded {len(cache)} images in {time.time() - t0:.0f}s', flush=True)


def predict(model, loader, device, amp):
    model.eval()
    out = np.zeros(len(loader.dataset), dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
        for x, _, idx in loader:
            out[idx.numpy()] = torch.sigmoid(model(x.to(device)).float()).cpu().numpy()
    return out


def train_eval(args, df, tr_idx, te_idx, p, cache, trial=None):
    """Train on tr_idx, return P(glaucoma) on te_idx. Reports per-epoch AUROC to Optuna for pruning."""
    device = args.device
    amp = device == 'cuda'
    torch.manual_seed(args.seed)
    model = build_model(args, p).to(device)
    # head: explicit L2 penalty in the loss (AdamW's decoupled decay barely regularises a linear head);
    # unfrozen backbone blocks: AdamW weight decay
    groups = [{'params': model.head.parameters(), 'lr': p['lr_head'], 'weight_decay': 0.0}]
    bb = [q for q in model.backbone.parameters() if q.requires_grad]
    if bb:
        groups.append({'params': bb, 'lr': p['lr_backbone'], 'weight_decay': p['weight_decay']})
    opt = torch.optim.AdamW(groups)
    head_w = [m.weight for m in model.head if isinstance(m, nn.Linear)]
    tr_dl, te_dl = loaders(args, df, tr_idx, te_idx, cache)
    steps = p['epochs'] * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    y_tr = df['y'].to_numpy()[tr_idx]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1 - y_tr).sum() / max(y_tr.sum(), 1)), dtype=torch.float32, device=device))
    y_te = df['y'].to_numpy()[te_idx]
    for ep in range(p['epochs']):
        model.train()
        if p['unfrozen_blocks'] == 0:
            model.backbone.eval()  # frozen backbone: keep it in eval mode
        for x, yb, _ in tr_dl:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
                loss = loss_fn(model(x.to(device)).float(), yb.to(device))
            loss = loss + p['l2_head'] * sum((w.float() ** 2).sum() for w in head_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        # report at the fraction of the schedule completed (0-10), so 5- and 20-epoch trials are compared
        # at the same point of their cosine schedules
        step = int(10 * (ep + 1) / p['epochs'])
        if trial is not None and step != int(10 * ep / p['epochs']):
            prob = predict(model, te_dl, device, amp)
            trial.report(rfg.metrics(y_te, prob)['auroc'], step)
            trial.set_user_attr('val_prob', prob.tolist())
            if trial.should_prune():
                raise optuna.TrialPruned()
    return predict(model, te_dl, device, amp)


def tune_and_test(args, df, tr_idx, te_idx, name, storage, cache):
    y, g = df['y'].to_numpy(), df['patient_id'].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    in_tr, in_va = next(sgkf.split(tr_idx, y[tr_idx], g[tr_idx]))
    in_tr, in_va = tr_idx[in_tr], tr_idx[in_va]

    def objective(trial):
        p = suggest(trial)
        prob = train_eval(args, df, in_tr, in_va, p, cache, trial)
        return rfg.metrics(y[in_va], prob)['auroc']

    study = optuna.create_study(study_name=name, direction='maximize', storage=storage, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    ok = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    done = len([t for t in study.trials if t.state in ok])
    # reseed on resume so a restarted job doesn't replay the same random start-up trials
    study.sampler = optuna.samplers.TPESampler(seed=args.seed + len(study.trials))
    if done < args.n_trials:
        study.optimize(objective, n_trials=args.n_trials - done, gc_after_trial=True)
    best = suggest(optuna.trial.FixedTrial(study.best_params))
    # threshold from the best trial's inner-validation predictions (training patients only)
    thr = rfg.youden_threshold(y[in_va], np.array(study.best_trial.user_attrs['val_prob']))
    prob = train_eval(args, df, tr_idx, te_idx, best, cache)
    return study, best, prob, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papila-labels', required=True)
    ap.add_argument('--chaksu-labels', default=None, help='if given, also tune on all PAPILA and test on Chaksu')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--preproc', choices=['resize', 'fov_square'], default='resize')
    ap.add_argument('--n-trials', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--img-size', type=int, default=rfg.IMG_SIZE)
    ap.add_argument('--cache-size', type=int, default=rfg.IMG_SIZE,
                    help='decoded images are kept at this size in RAM (392 = one resize, as in step 1)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='tiny run to check the code: 60 images, 2 trials, 1 fold')
    args = ap.parse_args()
    args.device = rfg.pick_device()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out / 'optuna_finetune.db'}"

    df = rfg.papila_binary(args.papila_labels, 'exclude')
    fold = rfg.papila_folds(df, 'official')
    folds = np.unique(fold)
    if args.smoke:
        keep = np.concatenate([np.flatnonzero(df['y'] == c)[:30] for c in (0, 1)])
        df, fold = df.iloc[keep].reset_index(drop=True), fold[keep]
        folds = folds[:1]
        args.img_size, args.cache_size, args.n_trials, args.batch_size = 196, 196, 2, 8
        BLOCK_CHOICES[:] = [0, 2]
        global suggest
        _suggest = suggest
        suggest = lambda t: _suggest(t) | {'epochs': 1}  # noqa: E731
    print(f'{len(df)} images, device={args.device}, folds={list(folds + 1)}, trials/fold={args.n_trials}', flush=True)

    y, g = df['y'].to_numpy(), df['patient_id'].to_numpy()
    state_path = out / 'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'folds': {}}
    oof, thr_arr = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for k_str, f in state['folds'].items():
        oof[fold == int(k_str) - 1], thr_arr[fold == int(k_str) - 1] = f['test_prob'], f['threshold']
    cache = {}
    warm_cache(args, df, cache)
    for k in folds:
        key = str(int(k) + 1)
        if key in state['folds']:
            continue
        t0 = time.time()
        tr_idx, te_idx = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        assert not set(g[tr_idx]) & set(g[te_idx])
        study, best, prob, thr = tune_and_test(args, df, tr_idx, te_idx, f'fold{key}', storage, cache)
        oof[te_idx], thr_arr[te_idx] = prob, thr
        state['folds'][key] = {'best': best, 'threshold': thr, 'inner_val_auroc_selection': study.best_value,
                               'test_auroc': rfg.metrics(y[te_idx], prob)['auroc'], 'test_prob': prob.tolist(),
                               'minutes': (time.time() - t0) / 60}
        state_path.write_text(json.dumps(state, indent=1))
        print(f"fold {key}: inner (selection) {study.best_value:.3f}, test {state['folds'][key]['test_auroc']:.3f}, "
              f"best {best}, {state['folds'][key]['minutes']:.1f} min", flush=True)

    done = ~np.isnan(oof)
    res = {'config': {k: v for k, v in vars(args).items()}, 'folds': state['folds']}
    if done.all() and len(np.unique(y)) == 2:
        res['papila'] = rfg.evaluate(y, oof, g, thr=thr_arr) | {'per_fold_auroc': rfg.per_fold_auroc(y, oof, fold)}
    pd.DataFrame({'image_id': df['image_id'], 'y': y, 'fold': fold + 1, 'p_finetune': oof}).to_csv(
        out / 'papila_oof_predictions.csv', index=False)

    if args.chaksu_labels and not args.smoke and 'chaksu' not in state:
        cha = rfg.chaksu_external(args.chaksu_labels)
        both = pd.concat([df.assign(src='papila'), cha.assign(src='chaksu', patient_id=cha['group'])],
                         ignore_index=True)
        tr_idx, te_idx = np.flatnonzero(both['src'] == 'papila'), np.flatnonzero(both['src'] == 'chaksu')
        cache = {}  # row indices change in the combined table
        warm_cache(args, both, cache)
        study, best, prob, thr = tune_and_test(args, both, tr_idx, te_idx, 'all_papila', storage, cache)
        cha['p_glaucoma'] = prob
        cha[['image_id', 'camera', 'n_glaucoma_votes', 'y', 'p_glaucoma']].to_csv(out / 'chaksu_predictions.csv',
                                                                                index=False)
        state['chaksu'] = {'best': best, 'threshold': thr, 'all': rfg.evaluate(cha['y'], prob, cha['group'], thr=thr),
                           'by_camera': {c: rfg.evaluate(d['y'], d['p_glaucoma'], d['group'], thr=thr)
                                         for c, d in cha.groupby('camera')},
                           'failures': rfg.failure_split(cha['y'], prob, cha['n_glaucoma_votes'], thr)}
        state_path.write_text(json.dumps(state, indent=1))
    res['chaksu'] = state.get('chaksu')
    (out / 'metrics.json').write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != 'folds'}, indent=1, default=float)[:3000])


if __name__ == '__main__':
    main()
