"""22 Nov figures: block-by-block frozen features, EndoWAM-inspired LoRA variants, camera hold-out.
python make_figures.py  (needs layer_features.csv, grounding_summary.csv, camera_holdout.csv, g_*.npz)"""
import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent; OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
TEXT, MUTED, GREY, BLUE, ORANGE, GREEN = '#1a1a1a', '#666666', '#9a9a9a', '#2f6b9a', '#c0583a', '#3c8c5a'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

# 1. frozen features block by block
L = pd.read_csv(HERE / 'layer_features.csv')
fig, ax = plt.subplots(figsize=(8.5, 3.6))
for kind, col, lab in (('logreg', BLUE, 'logistic regression'), ('svm', ORANGE, 'SVM')):
    d = L[(L.classifier == kind) & L.features.str.match(r'block \d+$')]
    x = d.features.str.extract(r'(\d+)')[0].astype(int)
    ci = np.array([ast.literal_eval(c) for c in d['AUPRC CI']])
    ax.plot(x, d.AUPRC, 'o-', color=col, label=lab)
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=col, alpha=0.12)
    j = L[(L.classifier == kind) & (L.features == 'blocks 9-12 joined')].AUPRC.iloc[0]
    ax.scatter([13.2], [j], marker='D', color=col)
ax.axhline(0.14, color=GREY, ls=':', lw=1); ax.text(0.6, 0.15, 'guessing 0.14', color=MUTED, fontsize=9)
ax.set_xticks(list(range(1, 13)) + [13.2]); ax.set_xticklabels([str(i) for i in range(1, 13)] + ['9-12\njoined'])
ax.set_xlabel('RETFound-Green block the features are taken from'); ax.set_ylabel('out-of-fold AUPRC')
ax.set_title('Frozen features from each block (setting tuned inside each fold; band = 95 % CI)', fontsize=11, loc='left')
ax.legend(frameon=False, loc='upper left'); fig.tight_layout(); fig.savefig(OUT / 'layers.png', dpi=200); plt.close(fig)

# 2. LoRA variants
G = pd.read_csv(HERE / 'grounding_summary.csv')
names = {'base': 'LoRA (rerun)', 'strong': 'Stronger augmentation', 'mask025': 'Disc-mask head, weight 0.25',
         'mask1': 'Disc-mask head, weight 1', 'disc': 'Disc-crop input (expert box)'}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={'width_ratios': [1.25, 1, 1]})
y = np.arange(len(G))[::-1]
for i, r in G.iterrows():
    rng = ast.literal_eval(r['AUPRC range']); col = GREY if r.variant == 'base' else BLUE
    axes[0].barh(y[i], r['AUPRC 3-seed mean'], color=col, height=0.6)
    axes[0].plot(rng, [y[i]] * 2, color=TEXT, lw=1.2)
    axes[0].text(r['AUPRC 3-seed mean'] + 0.005, y[i] + 0.22, f"{r['AUPRC 3-seed mean']:.2f}", fontsize=9)
axes[0].axvline(0.427, color=ORANGE, ls='--', lw=1); axes[0].text(0.435, y[0] + 0.45, 'frozen SVM 0.43', color=ORANGE, fontsize=9)
axes[0].set_yticks(y); axes[0].set_yticklabels([names[v] for v in G.variant]); axes[0].set_xlim(0, 0.68)
axes[0].set_title('AUPRC (bar: mean of 3 seeds; line: range)', fontsize=10, loc='left')
for ax, col, cic, title in ((axes[1], 'minus base AUPRC', 'CI vs base', 'AUPRC minus LoRA rerun (seed 0, 95 % CI)'),
                            (axes[2], 'minus base AUROC', 'AUROC CI vs base', 'AUROC minus LoRA rerun (seed 0, 95 % CI)')):
    for i, r in G.iterrows():
        if r.variant == 'base':
            continue
        lo, hi = ast.literal_eval(r[cic]); c = GREEN if lo > 0 else GREY
        ax.plot([lo, hi], [y[i]] * 2, color=c, lw=2.2); ax.plot(r[col], y[i], 'o', color=c)
    ax.axvline(0, color=TEXT, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.set_title(title, fontsize=10, loc='left')
fig.text(0.01, 0.01, 'Green: better than the LoRA rerun (CI above 0). Grey: tied. Healthy flagged at 90 % sensitivity: '
         + ', '.join(f"{lab} {p:.0f} %" for lab, p in zip(['LoRA rerun', 'stronger augmentation', 'mask head 0.25', 'mask head 1', 'disc crop'], G['healthy flagged % at 90 % sens'])),
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / 'variants.png', dpi=200); plt.close(fig)

# 3. camera hold-out
C = pd.read_csv(HERE / 'camera_holdout.csv')
cams = ['Remidio', 'Bosch', 'Forus']; models = list(C.model)
cols = [GREY, '#c4c4c4', BLUE, '#6f9fc4', ORANGE]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
for k, cam in enumerate(cams):
    ax = axes[k]; x = np.arange(len(models))
    ax.bar(x, C[f'{cam} held out'], color=cols, width=0.65)
    ax.scatter(x, C[f'{cam} in-distribution'], marker='_', s=300, color=TEXT, zorder=3)
    for xi, v in zip(x, C[f'{cam} held out']):
        ax.text(xi, v + 0.015, f'{v:.2f}', ha='center', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([m.replace('frozen ', 'frozen\n').replace('LoRA ', 'LoRA\n').replace('mask1', 'mask head') for m in models], fontsize=8.5)
    g, n = C[f'{cam} glaucoma / images'].iloc[0].split(' / ')
    ax.set_title(f'{cam} held out\n{g} glaucoma of {n} images; trained on the other {1141 - int(n)}', fontsize=9.5, loc='left')
axes[0].set_ylabel('AUPRC on the held-out camera')
fig.text(0.01, 0.01, 'Bar: trained on the other two cameras. Black line: same camera, ordinary 5-fold out-of-fold score. '
         'Image models: mean of 3 seeds, 7 epochs. Bosch and Forus have only 18 and 15 glaucoma images.', fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / 'camera_holdout.png', dpi=200); plt.close(fig)
print('saved')
