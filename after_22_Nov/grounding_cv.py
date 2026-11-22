"""22 Nov: three ideas borrowed from EndoWAM (arXiv 2608.01221), tested on LoRA RETFound-Green (the best 17 Nov model,
its Optuna setting unchanged: r 4, attention only, same learning rates, dropout, weight decay, class weight).
  base      the LoRA model rerun in this code (reference for every comparison below)
  strong    stronger augmentation (EndoWAM: rotation augmentation was its second-largest effect): flip, rotation up to
            +-15 deg, shift up to 8 px, brightness and contrast +-10 %  (base: flip and shift only, as on 17 Nov)
  mask025 / mask1   training-only grounding head (EndoWAM's largest effect): a linear layer on the 784 patch tokens
            predicts the expert optic-disc mask on the 28 x 28 patch grid; loss = classification + lambda x (BCE + Dice)
            of the mask, lambda 0.25 or 1 (two values fixed in advance, both reported). Masked loss as in EndoWAM:
            images without a mask contribute no mask gradient (every Chakshu image has one). The head is not used for
            the glaucoma score, so inference is unchanged.
  disc      explicit input grounding (EndoWAM-EG): the input is a square crop around the optic disc (3 disc diameters,
            box from the expert mask = an oracle location) instead of the whole image
Scoring as final_compare.py: 5 folds, pass 1 early stopping gives each fold's best epoch, pass 2 trains each fold for the
median best epoch of the other folds; seeds 0-2.
Camera hold-out (CFG=camera_<variant>): train on two cameras' training + validation images, score the third camera;
7 epochs (the median fair epoch of the 17 Nov LoRA run), no early stopping, seeds 0-2.
GPU: CFG=base python grounding_cv.py  ->  g_<CFG>.npz (out-of-fold scores, epochs, disc Dice)"""
import ast
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'after_17_Nov')); sys.path.insert(0, str(HERE.parent / 'pipeline'))
from img_common import CACHE, DEV, data  # noqa: E402
import lora_cv  # noqa: E402

CFG = os.environ.get('CFG', 'base')
VARIANT = CFG.replace('camera_', '')
SETTINGS = {'base': dict(aug='base', lam=0.0, inp='full'), 'strong': dict(aug='strong', lam=0.0, inp='full'),
            'mask025': dict(aug='base', lam=0.25, inp='full'), 'mask1': dict(aug='base', lam=1.0, inp='full'),
            'disc': dict(aug='base', lam=0.0, inp='disc')}
S = SETTINGS[VARIANT]
P = ast.literal_eval(pd.read_csv(HERE.parent / 'after_17_Nov' / 'final_chaksu_lora.csv')['setting'][0])
CAMERA_EPOCHS = 7


class GNet(nn.Module):
    """LoRA RETFound-Green (lora_cv.Net) + optional mask head on the patch tokens."""
    def __init__(self, p, seg):
        super().__init__()
        self.net = lora_cv.Net(p['r'], p['alpha_ratio'], p['targets'], p['lora_dropout'], p['dropout'])
        self.npt = self.net.backbone.num_prefix_tokens
        self.seg = nn.Linear(self.net.backbone.num_features, 1) if seg else None

    def forward(self, x):
        patches = self.net.backbone.forward_features(x)[:, self.npt:]      # (B, 784, 384), after the final norm
        logit = self.net.head(patches.mean(1))                               # = lora_cv.Net(x)
        seg = self.seg(patches).squeeze(-1) if self.seg is not None else None
        return logit, seg


def augment(x, m, strong):
    """x: (B, 3, H, W) in [0, 1]; m: (B, 1, H, W) mask in [0, 1] or None; the mask gets the same geometric change."""
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if m is not None:
        m = torch.where(flip[:, None, None, None], m.flip(3), m)
    if not strong:                                       # as img_common.augment: one shift of up to 8 px per batch
        dx, dy = np.random.randint(-8, 9, 2)
        x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
        m = None if m is None else torch.roll(m, shifts=(int(dy), int(dx)), dims=(2, 3))
        return x, m
    ang = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(15)
    t = (torch.rand(B, 2, device=x.device) * 2 - 1) * (2 * 8 / x.shape[-1])     # up to 8 px, in [-1, 1] coordinates
    theta = torch.stack([torch.stack([ang.cos(), -ang.sin(), t[:, 0]], 1), torch.stack([ang.sin(), ang.cos(), t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
    if m is not None:
        m = F.grid_sample(m, F.affine_grid(theta, list(m.shape), align_corners=False), padding_mode='zeros', align_corners=False)
    b = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * 0.1
    c = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * 0.1
    mu = x.mean((1, 2, 3), keepdim=True)
    return (((x - mu) * c + mu) * b).clamp(0, 1), m


def mask_loss(seg, m, has):
    """Per image BCE + soft Dice on the 28 x 28 patch grid; images without a mask (has = 0) give no gradient."""
    tgt = F.avg_pool2d(m, 14).flatten(1)                # share of each 14 x 14 patch inside the disc
    bce = F.binary_cross_entropy_with_logits(seg, tgt, reduction='none').mean(1)
    p = torch.sigmoid(seg)
    dice = 1 - (2 * (p * tgt).sum(1) + 1) / (p.sum(1) + tgt.sum(1) + 1)
    return ((bce + dice) * has).sum() / has.sum().clamp(min=1)


def dice_score(seg, m):
    tgt = (F.avg_pool2d(m, 14).flatten(1) >= 0.5).float(); p = (seg > 0).float()
    return ((2 * (p * tgt).sum(1) + 1) / (p.sum(1) + tgt.sum(1) + 1)).mean().item()


def fit(model, Xtr, Mtr, ytr, Xva, Mva, yva, pos_weight, max_epochs, patience, seed, select):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV)
    lora = [q for q in model.net.backbone.parameters() if q.requires_grad]
    groups = [{'params': model.net.head.parameters(), 'lr': P['lr_head']}, {'params': lora, 'lr': P['lr_lora']}]
    if model.seg is not None:
        groups.append({'params': model.seg.parameters(), 'lr': P['lr_head']})
    opt = torch.optim.AdamW(groups, lr=P['lr_head'], weight_decay=P['wd'])
    pw = torch.tensor([pos_weight], device=DEV)
    yt, yv = torch.tensor(ytr, device=DEV), torch.tensor(yva, device=DEV)
    has_all = (Mtr.flatten(1).amax(1) > 0).float().to(DEV)
    best, best_epoch, wait, keep, keep_dice = 1e9, 0, 0, None, float('nan')

    def evaluate():
        model.eval(); out, d = [], []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for s in range(0, len(Xva), 64):
                x = (Xva[s:s + 64].to(DEV).float() / 255 - 0.5) / 0.5
                lg, sg = model(x)
                out.append(lg.float().squeeze(1))
                if sg is not None:
                    d.append(dice_score(sg.float(), Mva[s:s + 64].to(DEV).float()[:, None] / 255) * len(x))
        return torch.cat(out), (sum(d) / len(Xva) if d else float('nan'))

    for epoch in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 32):
            b = perm[s:s + 32]
            x = Xtr[b].to(DEV).float() / 255
            m = Mtr[b].to(DEV).float()[:, None] / 255 if S['lam'] else None
            x, m = augment(x, m, S['aug'] == 'strong')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg, sg = model((x - 0.5) / 0.5)
            loss = F.binary_cross_entropy_with_logits(lg.float().squeeze(1), yt[b], pos_weight=pw)
            if S['lam']:
                loss = loss + S['lam'] * mask_loss(sg.float(), m, has_all[b.to(DEV)])
            opt.zero_grad(); loss.backward(); opt.step()
        sv, dv = evaluate()
        vl = F.binary_cross_entropy_with_logits(sv, yv, pos_weight=pw).item()
        if not select:
            keep, keep_dice, best_epoch = sv.cpu().numpy(), dv, epoch
            continue
        if vl < best:
            best, best_epoch, wait, keep, keep_dice = vl, epoch, 0, sv.cpu().numpy(), dv
        else:
            wait += 1
            if wait >= patience:
                break
    return keep, best_epoch, keep_dice


def run(X, M, y, splits, seed, fixed=None):
    oof, epochs, dice = np.full(len(y), np.nan), [], []
    for k, (tr, va) in enumerate(splits):
        ratio = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        torch.manual_seed(seed * 100 + k)
        model = GNet(P, S['lam'] > 0)
        s, e, d = fit(model, X[tr], M[tr], y[tr], X[va], M[va], y[va], ratio * P['w_scale'],
                      fixed[k] if fixed else 30, 6, seed, select=not fixed)
        oof[va] = s; epochs.append(e); dice.append(d)
    return oof, epochs, dice


if __name__ == '__main__':
    X, y, splits, lab = data('chaksu', 392)
    ids = np.load(CACHE / 'chaksu_ids.npy'); pos = {k: i for i, k in enumerate(ids)}
    idx = np.array([pos[k] for k in lab['image_id']])
    M = torch.from_numpy(np.ascontiguousarray(np.load(CACHE / 'chaksu_mask392.npy', mmap_mode='r')[idx]))
    if S['inp'] == 'disc':
        X = torch.from_numpy(np.ascontiguousarray(np.load(CACHE / 'chaksu_disc392.npy', mmap_mode='r')[idx]))
    print(CFG, S, 'images', tuple(X.shape), 'glaucoma', int(y.sum()), 'masks', int((M.flatten(1).amax(1) > 0).sum()), flush=True)
    if os.environ.get('SMOKE'):
        o, e, d = run(X, M, y, splits[:1], 0, fixed=[1])
        va = splits[0][1]
        print('smoke ok: fold-1 AUPRC', round(average_precision_score(y[va], o[va]), 3), 'dice', d,
              'GPU GB', round(torch.cuda.max_memory_allocated() / 1e9, 1)); sys.exit()
    res = {}
    if CFG.startswith('camera_'):
        cams = lab['camera'].to_numpy()
        csplits = [(np.flatnonzero(cams != c), np.flatnonzero(cams == c)) for c in ('Remidio', 'Bosch', 'Forus')]
        for seed in range(3):
            o, _, d = run(X, M, y, csplits, seed, fixed=[CAMERA_EPOCHS] * 3)
            res[f'seed{seed}'] = o; res[f'dice{seed}'] = np.array(d)
            print(CFG, 'seed', seed, {c: round(average_precision_score(y[cams == c], o[cams == c]), 4)
                                      for c in ('Remidio', 'Bosch', 'Forus')}, 'dice', d, flush=True)
    else:
        for seed in range(3):
            o_es, ep, _ = run(X, M, y, splits, seed)
            fixed = [max(1, int(np.median([e for j, e in enumerate(ep) if j != k]))) for k in range(len(splits))]
            o, _, d = run(X, M, y, splits, seed, fixed=fixed)
            res[f'seed{seed}'] = o; res[f'es{seed}'] = o_es; res[f'epochs{seed}'] = np.array(ep); res[f'fixed{seed}'] = np.array(fixed)
            res[f'dice{seed}'] = np.array(d)
            print(CFG, 'seed', seed, 'fair AUPRC', round(average_precision_score(y, o), 4), 'AUROC', round(roc_auc_score(y, o), 4),
                  '| early-stopped', round(average_precision_score(y, o_es), 4), '| epochs', ep, 'fixed', fixed, 'dice', d, flush=True)
    np.savez(HERE / f'g_{CFG}.npz', image_id=lab['image_id'].to_numpy().astype(str), y=y, camera=lab['camera'].to_numpy().astype(str), **res)
    print('saved', f'g_{CFG}.npz')
