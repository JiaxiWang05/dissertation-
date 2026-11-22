"""22 Nov (EndoWAM showed attention maps as evidence that grounding moves the model's focus onto the target): where in
the image does each model find its glaucoma evidence?
Every RETFound-Green model here ends in a linear head on the MEAN of the 784 patch tokens, so its score splits exactly
into one contribution per patch: c_i = w . t_i / 784 (frozen logistic regression: w / feature s.d. after scaling). The
CNN ends in global average pooling + linear, so class activation mapping is exact as well: c_ij = w . f_ij / 64 on its
8 x 8 grid. No gradient approximation is needed.
Measure: share of the positive evidence inside the expert optic-disc mask, sum(relu(c) * disc) / sum(relu(c)), against
the disc's share of the image (what a map spread evenly would give). Models trained on fold 1 (seed 0, the fixed epoch
count of their 17 Nov fair run for fold 1), maps on fold 1's 228 validation images. Test untouched.
Figure: heat maps with the disc outline only (no photographs: some Chakshu images carry burned-in patient details).
GPU: python attention_maps.py -> attention_maps.csv, attention_maps.npz, figures/attention_maps.png"""
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'after_17_Nov')); sys.path.insert(0, str(HERE.parent / 'pipeline')); sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], 'chaksu']                     # cnn_cv reads the dataset from argv
os.environ['CFG'] = 'mask1'                            # grounding_cv: the mask-head model with lambda 1
from img_common import CACHE, DEV, data, fit  # noqa: E402
import cnn_cv, lora_cv, unfreeze_cv  # noqa: E402
import grounding_cv  # noqa: E402

A17 = HERE.parent / 'after_17_Nov'
setting = lambda csv, model: ast.literal_eval(pd.read_csv(A17 / csv).set_index('model').loc[model, 'setting'])
EPOCHS = {'cnn': 26, 'unfreeze': 6, 'lora': 5, 'mask1': 5}   # fold 1, seed 0 fixed epochs of the 17 Nov fair runs (mask: LoRA's)

X392, y, splits, lab = data('chaksu', 392)
X256, _, _, _ = data('chaksu', 256)
ids = np.load(CACHE / 'chaksu_ids.npy'); pos = {k: i for i, k in enumerate(ids)}
idx = np.array([pos[k] for k in lab['image_id']])
M = torch.from_numpy(np.ascontiguousarray(np.load(CACHE / 'chaksu_mask392.npy', mmap_mode='r')[idx]))
tr, va = splits[0]
ratio = (y[tr] == 0).sum() / (y[tr] == 1).sum()
prep = lambda x: (x.float() / 255.0 - 0.5) / 0.5


def disc_frac(grid):
    return F.avg_pool2d(M[va].float()[:, None] / 255, 392 // grid).flatten(1).numpy()     # (n, grid*grid)


@torch.no_grad()
def vit_maps(backbone, w, X, scale=None):
    out = []
    backbone.eval()
    for s in range(0, len(va), 32):
        x = prep(X[va[s:s + 32]].to(DEV))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            t = backbone.forward_features(x)[:, backbone.num_prefix_tokens:].float()
        out.append((t @ (w if scale is None else w / scale)).cpu().numpy() / t.shape[1])
    return np.concatenate(out)


def last_linear(head):
    return [m for m in head.modules() if isinstance(m, nn.Linear)][-1].weight.detach().float().squeeze(0)


maps = {}
# frozen features + logistic regression (C of the 8 Nov winner)
torch.manual_seed(0)
bb = lora_cv.rfg.load_retfound_green(lora_cv.WEIGHTS).to(DEV).eval()
feats = np.load(CACHE / 'chaksu_layers.npz')['feats'][11]
sc = StandardScaler().fit(feats[tr])
lr = LogisticRegression(C=0.007250175039761133, class_weight='balanced', max_iter=5000).fit(sc.transform(feats[tr]), y[tr])
maps['Frozen + logistic regression'] = vit_maps(bb, torch.tensor(lr.coef_[0], device=DEV, dtype=torch.float32), X392,
                                                 torch.tensor(sc.scale_, device=DEV, dtype=torch.float32))
del bb
# two open blocks
p = setting('final_chaksu.csv', 'unfreeze'); torch.manual_seed(0)
m = unfreeze_cv.Net(p['k'], p['hidden'], p['dropout'])
g = [{'params': m.head.parameters(), 'lr': p['lr_head']}, {'params': [q for q in m.backbone.parameters() if q.requires_grad], 'lr': p['lr_head'] * p['lr_ratio']}]
fit(m, X392[tr], y[tr], X392[va], y[va], unfreeze_cv.prep, p['lr_head'], p['wd'], 32, ratio * p['w_scale'], EPOCHS['unfreeze'], 6, 0, param_groups=g, select=False)
maps['2 open blocks'] = vit_maps(m.backbone, last_linear(m.head).to(DEV), X392); del m
# LoRA
p = setting('final_chaksu_lora.csv', 'lora'); torch.manual_seed(0)
m = lora_cv.Net(p['r'], p['alpha_ratio'], p['targets'], p['lora_dropout'], p['dropout'])
g = [{'params': m.head.parameters(), 'lr': p['lr_head']}, {'params': [q for q in m.backbone.parameters() if q.requires_grad], 'lr': p['lr_lora']}]
fit(m, X392[tr], y[tr], X392[va], y[va], lora_cv.prep, p['lr_head'], p['wd'], 32, ratio * p['w_scale'], EPOCHS['lora'], 6, 0, param_groups=g, select=False)
maps['LoRA'] = vit_maps(m.backbone, last_linear(m.head).to(DEV), X392); del m
# LoRA + disc-mask head (lambda 1)
torch.manual_seed(0)
m = grounding_cv.GNet(grounding_cv.P, True)
grounding_cv.fit(m, X392[tr], M[tr], y[tr], X392[va], M[va], y[va], ratio * grounding_cv.P['w_scale'], EPOCHS['mask1'], 6, 0, select=False)
maps['LoRA + disc-mask head'] = vit_maps(m.net.backbone, last_linear(m.net.head).to(DEV), X392); del m
# CNN from scratch (class activation map, 8 x 8)
p = setting('final_chaksu.csv', 'cnn'); torch.manual_seed(0)
m = cnn_cv.CNN(3, p['blocks'], p['c0'], p['k'], p['dropout'])
fit(m, X256[tr], y[tr], X256[va], y[va], cnn_cv.prep, p['lr'], p['wd'], p['batch'], ratio * p['w_scale'], EPOCHS['cnn'], 10, 0, select=False)
m.eval(); w = last_linear(m.head).to(DEV); out = []
with torch.no_grad():
    for s in range(0, len(va), 32):
        f = m.features(cnn_cv.prep(X256[va[s:s + 32]].to(DEV))).float()
        out.append(torch.einsum('bchw,c->bhw', f, w).flatten(1).cpu().numpy() / (f.shape[2] * f.shape[3]))
maps['CNN from scratch'] = np.concatenate(out)

rows, yv = [], y[va]
for name, c in maps.items():
    grid = int(round(np.sqrt(c.shape[1]))); d = disc_frac(grid); cp = np.clip(c, 0, None)
    share = (cp * d).sum(1) / np.maximum(cp.sum(1), 1e-12); chance = d.mean(1)
    for grp, sel in (('all', np.ones(len(va), bool)), ('glaucoma', yv == 1), ('healthy', yv == 0)):
        rows.append({'model': name, 'images': grp, 'n': int(sel.sum()), 'grid': f'{grid}x{grid}',
                     'evidence inside disc %': 100 * share[sel].mean(), 'disc share of image %': 100 * chance[sel].mean(),
                     'times chance': (share[sel] / chance[sel]).mean()})
out = pd.DataFrame(rows); out.to_csv(HERE / 'attention_maps.csv', index=False); print(out.to_string())
np.savez(HERE / 'attention_maps.npz', image_id=lab['image_id'].to_numpy()[va].astype(str), y=yv,
         disc28=disc_frac(28), **{k: v for k, v in maps.items()})

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt  # noqa: E402
gl = np.flatnonzero(yv == 1)[:4]; names = list(maps)
fig, axes = plt.subplots(len(gl), len(names), figsize=(2.3 * len(names), 2.3 * len(gl)))
d28 = disc_frac(28)
for r, i in enumerate(gl):
    for cidx, name in enumerate(names):
        c = maps[name][i]; g_ = int(round(np.sqrt(c.size)))
        img = F.interpolate(torch.tensor(np.clip(c, 0, None)).view(1, 1, g_, g_), size=(28, 28), mode='bilinear', align_corners=False)[0, 0].numpy()
        ax = axes[r, cidx]; ax.imshow(img, cmap='magma'); ax.contour(d28[i].reshape(28, 28), levels=[0.5], colors='#4fc3f7', linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(name, fontsize=9)
fig.suptitle('Positive glaucoma evidence per patch (fold-1 glaucoma images); blue line = expert optic-disc outline', fontsize=10)
fig.tight_layout(); (HERE / 'figures').mkdir(exist_ok=True); fig.savefig(HERE / 'figures' / 'attention_maps.png', dpi=180)
print('saved figure')
