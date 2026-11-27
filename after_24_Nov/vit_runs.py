"""24 Nov meeting: (1) training curves for the RETFound-Green models, (2) LoRA combined with the last two blocks opened.
Models (all with their own 17 / 22 Nov settings unchanged):
  blocks2      last 2 blocks + final norm opened (../after_17_Nov/unfreeze_cv.py, its Optuna setting)
  lora         LoRA rank 4 in the attention of all 12 blocks (../after_17_Nov/lora_cv.py setting)
  mask025      LoRA + training-only disc-mask head, loss weight 0.25 (../after_22_Nov/grounding_cv.py)
  lora_blocks  LoRA in all 12 blocks AND blocks 11-12 + final norm opened; the opened blocks learn at the 2-block
               model's backbone rate (lr_head x lr_ratio = 0.00014) and weight decay (0.032), LoRA and head at their LoRA rates
Modes:  CURVES=1 MODEL=<name> python vit_runs.py   seed 0, every fold trained 30 epochs without stopping; per epoch the
                                                   training loss (classification part), validation loss and validation
                                                   AUPRC / AUROC -> curves_<name>.json
        FINAL=1 MODEL=lora_blocks python vit_runs.py   fair score (median best epoch of the other folds, 3 seeds)
                                                       -> g_lora_blocks.npz (same format as ../after_22_Nov/g_*.npz)"""
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
A17, A22 = HERE.parent / 'after_17_Nov', HERE.parent / 'after_22_Nov'
sys.path.insert(0, str(A17)); sys.path.insert(0, str(HERE.parent / 'pipeline')); sys.path.insert(0, str(A22))
MODEL = os.environ['MODEL']
os.environ['CFG'] = 'mask025' if MODEL == 'mask025' else 'base'
from img_common import CACHE, DEV, data, fit as fit17  # noqa: E402
import grounding_cv as G  # noqa: E402
import unfreeze_cv  # noqa: E402

LAM = 0.25 if MODEL == 'mask025' else 0.0
P_LORA = G.P
P_BLK = ast.literal_eval(pd.read_csv(A17 / 'final_chaksu.csv').set_index('model').loc['unfreeze', 'setting'])


def open_last_blocks(gnet, k=2):
    bb = gnet.net.backbone
    for blk in bb.blocks[len(bb.blocks) - k:]:
        for n, q in blk.named_parameters():
            q.requires_grad = True                      # base weights of the opened blocks (LoRA A, B already trainable)
    for q in bb.norm.parameters():
        q.requires_grad = True


def fit_hist(model, Xtr, Mtr, ytr, Xva, yva, pos_weight, max_epochs, patience, seed, select):
    """grounding_cv.fit with a per-epoch history and an optional group for opened blocks."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV)
    named = list(model.net.backbone.named_parameters())
    lora = [q for n, q in named if q.requires_grad and ('.A.' in n or '.B.' in n)]
    opened = [q for n, q in named if q.requires_grad and not ('.A.' in n or '.B.' in n)]
    groups = [{'params': model.net.head.parameters(), 'lr': P_LORA['lr_head']}, {'params': lora, 'lr': P_LORA['lr_lora']}]
    if opened:
        groups.append({'params': opened, 'lr': P_BLK['lr_head'] * P_BLK['lr_ratio'], 'weight_decay': P_BLK['wd']})
    if model.seg is not None:
        groups.append({'params': model.seg.parameters(), 'lr': P_LORA['lr_head']})
    opt = torch.optim.AdamW(groups, lr=P_LORA['lr_head'], weight_decay=P_LORA['wd'])
    pw = torch.tensor([pos_weight], device=DEV)
    yt, yv = torch.tensor(ytr, device=DEV), torch.tensor(yva, device=DEV)
    has_all = (Mtr.flatten(1).amax(1) > 0).float().to(DEV)
    best, best_epoch, wait, keep, hist = 1e9, 0, 0, None, []
    for epoch in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tl = 0.0
        for s in range(0, len(Xtr), 32):
            b = perm[s:s + 32]
            x = Xtr[b].to(DEV).float() / 255
            m = Mtr[b].to(DEV).float()[:, None] / 255 if LAM else None
            x, m = G.augment(x, m, False)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg, sg = model((x - 0.5) / 0.5)
            lc = F.binary_cross_entropy_with_logits(lg.float().squeeze(1), yt[b], pos_weight=pw)
            loss = lc + (LAM * G.mask_loss(sg.float(), m, has_all[b.to(DEV)]) if LAM else 0)
            opt.zero_grad(); loss.backward(); opt.step(); tl += lc.item() * len(b)
        model.eval(); out = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for s in range(0, len(Xva), 64):
                out.append(model((Xva[s:s + 64].to(DEV).float() / 255 - 0.5) / 0.5)[0].float().squeeze(1))
        sv = torch.cat(out); vl = F.binary_cross_entropy_with_logits(sv, yv, pos_weight=pw).item(); pv = sv.cpu().numpy()
        hist.append({'epoch': epoch, 'train_loss': tl / len(Xtr), 'val_loss': vl, 'val_auprc': average_precision_score(yva, pv),
                     'val_auroc': roc_auc_score(yva, pv)})
        if not select:
            keep, best_epoch = pv, epoch; continue
        if vl < best:
            best, best_epoch, wait, keep = vl, epoch, 0, pv
        else:
            wait += 1
            if wait >= patience:
                break
    return keep, best_epoch, hist


def build(k):
    torch.manual_seed(k)
    m = G.GNet(P_LORA, LAM > 0)
    if MODEL == 'lora_blocks':
        open_last_blocks(m)
    return m


def run(X, M, y, splits, seed, fixed=None, select=None, hists=None):
    oof, epochs = np.zeros(len(y)), []
    for k, (tr, va) in enumerate(splits):
        ratio = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        torch.manual_seed(seed * 100 + k)
        model = build(seed * 100 + k)
        s, e, h = fit_hist(model, X[tr], M[tr], y[tr], X[va], y[va], ratio * P_LORA['w_scale'], fixed[k] if fixed else 30, 6,
                           seed, (not fixed) if select is None else select)
        oof[va] = s; epochs.append(e)
        if hists is not None:
            hists.append(h)
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    return oof, epochs, n_train


if __name__ == '__main__':
    X, y, splits, lab = data('chaksu', 392)
    ids = np.load(CACHE / 'chaksu_ids.npy'); pos = {k: i for i, k in enumerate(ids)}
    M = torch.from_numpy(np.ascontiguousarray(np.load(CACHE / 'chaksu_mask392.npy', mmap_mode='r')[[pos[k] for k in lab['image_id']]]))
    if os.environ.get('SMOKE'):
        assert MODEL != 'blocks2', 'blocks2 is only used in CURVES'
        o, e, n = run(X, M, y, splits[:1], 0, fixed=[1])
        print('smoke ok', MODEL, 'trainable', n, 'GPU GB', round(torch.cuda.max_memory_allocated() / 1e9, 1)); sys.exit()
    if os.environ.get('CURVES'):
        hists = []
        if MODEL == 'blocks2':                       # the 17 Nov 2-block model through its own training loop
            for k, (tr, va) in enumerate(splits):
                ratio = (y[tr] == 0).sum() / (y[tr] == 1).sum(); torch.manual_seed(k)
                m = unfreeze_cv.Net(P_BLK['k'], P_BLK['hidden'], P_BLK['dropout'])
                g = [{'params': m.head.parameters(), 'lr': P_BLK['lr_head']},
                     {'params': [q for q in m.backbone.parameters() if q.requires_grad], 'lr': P_BLK['lr_head'] * P_BLK['lr_ratio']}]
                _, _, h = fit17(m, X[tr], y[tr], X[va], y[va], unfreeze_cv.prep, P_BLK['lr_head'], P_BLK['wd'], 32,
                                ratio * P_BLK['w_scale'], 30, 6, 0, param_groups=g, select=False)
                hists.append(h); print('fold', k, 'done', flush=True)
            n = sum(q.numel() for q in m.parameters() if q.requires_grad)
        else:
            _, _, n = run(X, M, y, splits, 0, fixed=[30] * 5, select=False, hists=hists)
        json.dump({'model': MODEL, 'trainable': n, 'folds': hists}, open(HERE / f'curves_{MODEL}.json', 'w'))
        print('saved curves', MODEL, 'trainable', n); sys.exit()
    # FINAL (lora_blocks): fair score as ../after_22_Nov/grounding_cv.py
    assert MODEL == 'lora_blocks', 'FINAL is only for lora_blocks'
    res = {}
    for sd in range(3):
        o_es, ep, n = run(X, M, y, splits, sd)
        fixed = [max(1, int(np.median([e for j, e in enumerate(ep) if j != k]))) for k in range(5)]
        o, _, _ = run(X, M, y, splits, sd, fixed=fixed)
        res[f'seed{sd}'] = o; res[f'es{sd}'] = o_es; res[f'epochs{sd}'] = np.array(ep); res[f'fixed{sd}'] = np.array(fixed)
        print(MODEL, 'seed', sd, 'fair AUPRC', round(average_precision_score(y, o), 4), 'AUROC', round(roc_auc_score(y, o), 4),
              '| epochs', ep, 'fixed', fixed, 'trainable', n, flush=True)
    np.savez(HERE / f'g_{MODEL}.npz', image_id=lab['image_id'].to_numpy().astype(str), y=y, **res)
    print('saved')
