"""22 Nov, revised evidence maps (replaces the positive-only figure of attention_maps.py; same saved maps).
Method (exact, no gradients): each RETFound-Green model here scores the MEAN of its 784 patch tokens with a linear head,
so its logit = bias + sum_i c_i with c_i = w . t_i / 784 (frozen logistic regression: w / feature s.d.; the mean term
goes into the bias). The CNN ends in global average pooling + linear, so logit = bias + sum_ij c_ij, c_ij = w . f_ij / 64
on its 8 x 8 grid (class activation map). c > 0 pushes towards glaucoma, c < 0 towards healthy; both are shown.
Metrics per image (fold 1, 229 validation images), against the expert disc mask (share of each patch inside the disc):
  decision weight in disc = sum |c| x disc / sum |c|   (compare with the disc's share of the image, 1.8 %)
  disc contrast           = (mean c over disc patches - mean c over the other patches) / s.d. of c in that image
  positive / negative mass = sum of c > 0 / sum of c < 0
python evidence_figure.py  ->  figures/evidence_maps.png, evidence_metrics.csv"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
z = np.load(HERE / 'attention_maps.npz', allow_pickle=False)
y = z['y']; names = ['Frozen + logistic regression', '2 open blocks', 'LoRA', 'LoRA + disc-mask head', 'CNN from scratch']


D28 = z['disc28']   # share of each 14 x 14 patch inside the expert disc (saved by attention_maps.py)


def disc_frac(grid):
    if grid == 28:
        return D28
    full = torch.tensor(D28, dtype=torch.float32).view(-1, 1, 28, 28).repeat_interleave(14, 2).repeat_interleave(14, 3)
    return F.avg_pool2d(full, 392 // grid).flatten(1).numpy()   # CNN 8 x 8 cells, from the 28 x 28 mask grid


rows = []
for n in names:
    c = z[n]; g = int(round(np.sqrt(c.shape[1]))); d = disc_frac(g); inside = d >= 0.5
    w_in = (np.abs(c) * d).sum(1) / np.abs(c).sum(1)
    contrast = np.array([(c[i][inside[i]].mean() - c[i][~inside[i]].mean()) / c[i].std() if inside[i].any() else np.nan
                         for i in range(len(c))])
    pm, nm = np.clip(c, 0, None).sum(1), np.clip(-c, 0, None).sum(1)
    for grp, s in (('glaucoma', y == 1), ('healthy', y == 0)):
        rows.append({'model': n, 'images': grp, 'n': int(s.sum()), 'grid': f'{g}x{g}',
                     'decision weight in disc %': 100 * w_in[s].mean(), 'disc share of image %': 100 * d[s].mean(),
                     'disc contrast (s.d.)': np.nanmean(contrast[s]), 'positive patches %': 100 * (c[s] > 0).mean(),
                     'positive mass': pm[s].mean(), 'negative mass': -nm[s].mean(), 'sum of c (logit minus bias)': c[s].sum(1).mean()})
met = pd.DataFrame(rows); met.to_csv(HERE / 'evidence_metrics.csv', index=False); print(met.round(3).to_string())

# figure: 4 glaucoma images x 5 models, signed maps, one symmetric scale per model (99th percentile of |c| over all images)
gl = np.flatnonzero(y == 1)[:4]
d28 = disc_frac(28)
fig = plt.figure(figsize=(15, 13.5))
gs = fig.add_gridspec(len(gl) + 3, len(names), height_ratios=[1] * len(gl) + [0.06, 0.32, 1.25], hspace=0.3, wspace=0.08, top=0.93, bottom=0.04)
for ci, n in enumerate(names):
    c = z[n]; g = int(round(np.sqrt(c.shape[1]))); lim = np.percentile(np.abs(c), 99)
    for r, i in enumerate(gl):
        ax = fig.add_subplot(gs[r, ci])
        im = ax.imshow(c[i].reshape(g, g), cmap='RdBu_r', vmin=-lim, vmax=lim, extent=(0, 28, 28, 0), interpolation='nearest')
        ax.contour(np.linspace(0.5, 27.5, 28), np.linspace(0.5, 27.5, 28), d28[i].reshape(28, 28), levels=[0.5], colors='#00a000', linewidths=1.4)
        ax.set_xticks([]); ax.set_yticks([])
        s = c[i].sum()
        ax.text(0.5, -0.02, f'sum {s:+.2f}', transform=ax.transAxes, ha='center', va='top', fontsize=8, color='#444444')
        if r == 0:
            ax.set_title(n, fontsize=10)
    cax = fig.add_subplot(gs[len(gl), ci]); p_ = cax.get_position()
    cax.set_position([p_.x0 + 0.12 * p_.width, p_.y0, 0.76 * p_.width, p_.height])
    e = 10 ** np.floor(np.log10(lim)); t = float(np.floor(lim / e) * e)        # lim rounded down to 1 significant digit
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-t, 0, t]); cb.ax.tick_params(labelsize=7.5)
    cb.ax.set_xticklabels([f'{-t:g}', '0', f'{t:g}'])
    cb.set_label('contribution to the logit per patch' + (' (8 x 8 cells)' if g == 8 else ''), fontsize=7.5)
# metrics panels
mg = met[met.images == 'glaucoma'].set_index('model').loc[names]; mh = met[met.images == 'healthy'].set_index('model').loc[names]
sub = gs[len(gl) + 2, :].subgridspec(1, 3, wspace=0.35)
x = np.arange(len(names)); short = ['Frozen', '2 blocks', 'LoRA', 'LoRA +\nmask head', 'CNN']
ax = fig.add_subplot(sub[0]); ax.bar(x - 0.2, mg['decision weight in disc %'], 0.4, color='#c0583a', label='glaucoma images')
ax.bar(x + 0.2, mh['decision weight in disc %'], 0.4, color='#9a9a9a', label='healthy images')
ax.axhline(mg['disc share of image %'].mean(), color='#00a000', ls='--', lw=1.2); ax.text(-0.45, mg['disc share of image %'].mean() + 0.08, 'disc area', color='#00a000', fontsize=8, ha='left')
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8); ax.set_ylabel('% of total |contribution|'); ax.set_title('Decision weight inside the disc', fontsize=10, loc='left'); ax.legend(frameon=False, fontsize=8, loc='upper right', bbox_to_anchor=(1, 1.02))
ax.set_ylim(0, 3.6)
ax = fig.add_subplot(sub[1]); ax.bar(x - 0.2, mg['disc contrast (s.d.)'], 0.4, color='#c0583a'); ax.bar(x + 0.2, mh['disc contrast (s.d.)'], 0.4, color='#9a9a9a')
ax.axhline(0, color='#1a1a1a', lw=0.8); ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8); ax.set_ylabel('s.d. units')
ax.set_title('Disc contrast: disc patches minus the rest', fontsize=10, loc='left')
ax = fig.add_subplot(sub[2]); ax.bar(x - 0.2, mg['positive mass'], 0.4, color='#b2182b', label='positive (towards glaucoma)')
ax.bar(x + 0.2, mg['negative mass'], 0.4, color='#2166ac', label='negative (towards healthy)'); ax.axhline(0, color='#1a1a1a', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8); ax.set_ylabel('sum of contributions'); ax.set_title('Positive and negative evidence (glaucoma images)', fontsize=10, loc='left'); ax.legend(frameon=False, fontsize=8)
fig.suptitle('Signed per-patch contributions to the glaucoma logit, four fold-1 glaucoma images (red = towards glaucoma, blue = towards healthy; '
             'green line = expert disc).\nExact decomposition: logit = bias + sum of patch contributions. One colour scale per model (symmetric, 99th percentile of |c|); '
             'the CNN map is 8 x 8 and its scale differs. No photographs shown.', fontsize=10, y=0.985)
fig.savefig(HERE / 'figures' / 'evidence_maps.png', dpi=170); print('saved')
