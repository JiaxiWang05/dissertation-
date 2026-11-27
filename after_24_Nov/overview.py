"""Two summary pictures for the Notion follow-up page.
figures/results_overview.png   one row per Chakshu model: fair AUPRC (3-seed mean and range), healthy images flagged at
                               90 % sensitivity (seed 0) and trained weights. Numbers read from the result files.
figures/small_cnn_diagram.png  the chosen smallest CNN, layer by layer, with the shape after each layer."""
import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

HERE = Path(__file__).resolve().parent; R = HERE.parent; OUT = HERE / 'figures'
TEXT, MUTED = '#1a1a1a', '#666666'
GREY, GOLD, BLUE, RED, GREEN, PURPLE = '#7a7a7a', '#b08d2a', '#2f6b9a', '#c0583a', '#2e8b57', '#7b4fa0'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def rng(s):
    v = ast.literal_eval(s) if isinstance(s, str) else s; return float(v[0]), float(v[1])


f17 = pd.read_csv(R / 'after_17_Nov' / 'final_chaksu.csv').set_index('model')
fl = pd.read_csv(R / 'after_17_Nov' / 'final_chaksu_lora.csv').set_index('model')
g22 = pd.read_csv(R / 'after_22_Nov' / 'grounding_summary.csv').set_index('variant')
cnn = pd.read_csv(HERE / 'small_cnn_final.csv').iloc[0]
svm = pd.read_csv(R / 'after_10_Nov' / 'step5_comparison.csv').set_index('classifier').loc['svm']
conf = pd.read_csv(HERE / 'confusion_best.csv').set_index('model')['healthy flagged %']
rows = [('Frozen features + SVM', svm['AUPRC'], (svm['AUPRC'], svm['AUPRC']), conf['Frozen features + SVM'], None, GREY, 'kernel SVM on 384 frozen features'),
        ('Small CNN from scratch (24 Nov)', cnn['AUPRC 3 seeds mean'], rng(cnn['AUPRC range']), cnn['healthy flagged % at 90 % sens'],
         int(cnn['parameters']), GOLD, ''),
        ('CNN from scratch, 5 blocks (17 Nov)', f17.loc['cnn', 'AUPRC 3 seeds mean'], rng(f17.loc['cnn', 'AUPRC 3 seeds range']),
         f17.loc['cnn', 'healthy flagged % at 90 % sens'], int(f17.loc['cnn', 'parameters (trainable)']), GOLD, ''),
        ('2 open blocks', f17.loc['unfreeze', 'AUPRC 3 seeds mean'], rng(f17.loc['unfreeze', 'AUPRC 3 seeds range']),
         f17.loc['unfreeze', 'healthy flagged % at 90 % sens'], 3_551_617, BLUE, ''),
        ('LoRA', fl.loc['lora', 'AUPRC 3 seeds mean'], rng(fl.loc['lora', 'AUPRC 3 seeds range']),
         fl.loc['lora', 'healthy flagged % at 90 % sens'], 110_977, RED, ''),
        ('LoRA + disc-mask head', g22.loc['mask025', 'AUPRC 3-seed mean'], rng(g22.loc['mask025', 'AUPRC range']),
         g22.loc['mask025', 'healthy flagged % at 90 % sens'], 110_977, GREEN, '+385 training only')]
if (HERE / 'g_lora_blocks.npz').exists():
    g = np.load(HERE / 'g_lora_blocks.npz'); aps = [average_precision_score(g['y'], g[f'seed{i}']) for i in range(3)]
    hf = conf.get('LoRA + 2 open blocks', np.nan)
    rows.append(('LoRA + 2 open blocks', float(np.mean(aps)), (min(aps), max(aps)), hf, 3_662_209, PURPLE, ''))

fig, axes = plt.subplots(1, 3, figsize=(14, 0.62 * len(rows) + 1.9), gridspec_kw={'width_ratios': [2.2, 1.3, 1.3], 'wspace': 0.08})
ys = np.arange(len(rows))[::-1]
a, b, c = axes
for y, (n, m, (lo, hi), hfl, w, col, note) in zip(ys, rows):
    a.plot([lo, hi], [y, y], color=col, lw=6, alpha=0.35, solid_capstyle='round'); a.plot(m, y, 'o', color=col, ms=11, mec='white', mew=1.5)
    a.text(m, y + 0.3, f'{m:.2f}', ha='center', fontsize=9.5, color=col, fontweight='bold')
    b.barh(y, hfl, color=col, height=0.55, alpha=0.85); b.text(hfl + 1.5, y, f'{hfl:.0f} %', va='center', fontsize=9.5)
    if w:
        c.barh(y, w, color=col, height=0.55, alpha=0.85); c.text(w * 1.25, y, f'{w:,}' + (f'  {note}' if note else ''), va='center', fontsize=8.5)
    else:
        c.text(120, y, note, va='center', fontsize=8.5, color=MUTED)
a.axvline(0.14, color='#999999', ls=':', lw=1.2); a.text(0.145, ys[-1] - 0.7, 'guessing 0.14', fontsize=8.5, color=MUTED)
a.axvline(svm['AUPRC'], color=GREY, ls='--', lw=0.8, alpha=0.6)
a.set_yticks(ys); a.set_yticklabels([r[0] for r in rows], fontsize=10.5); a.set_xlim(0.1, 0.7); a.set_ylim(-0.8, len(rows) - 0.3)
a.set_xlabel('AUPRC, fair out-of-fold (higher is better)\ndot = 3-seed mean, bar = seed range'); a.grid(axis='x', color='#eeeeee')
b.set_yticks([]); b.set_xlim(0, 105); b.set_xlabel('healthy images flagged at 90 % sensitivity\nlower is better')
c.set_yticks([]); c.set_xscale('log'); c.set_xlim(100, 3e9); c.set_xlabel('trained weights (log scale)')
for ax in (b, c):
    ax.spines['left'].set_visible(False); ax.set_ylim(a.get_ylim())
fig.suptitle('All Chákṣu models at a glance (1,141 images, the same 5 folds)', x=0.02, ha='left', fontsize=13, fontweight='bold')
fig.savefig(OUT / 'results_overview.png', dpi=170, bbox_inches='tight'); plt.close(fig); print('saved overview', [r[0] for r in rows])

# ------------------------------------------------------------------ small CNN diagram
p = ast.literal_eval(cnn['setting'])
c0, k = p['c0'], p['k']
layers = [('photo', '3 x 256 x 256', '#eeeeee'),
          (f'conv {k}x{k}, {c0} filters\nbatch norm + ReLU', f'{c0} x 256 x 256', '#dce8f2'), ('max pool 2x2', f'{c0} x 128 x 128', '#f1f1f1'),
          (f'conv {k}x{k}, {2 * c0} filters\nbatch norm + ReLU', f'{2 * c0} x 128 x 128', '#dce8f2'), ('max pool 2x2', f'{2 * c0} x 64 x 64', '#f1f1f1'),
          ('average pool\nto 4 x 4', f'{2 * c0} x 4 x 4 = {2 * c0 * 16}', '#f1f1f1'), (f'dropout {p["dropout"]:.2f}\nlinear -> 1', 'glaucoma score', '#f6dccf')]
fig, ax = plt.subplots(figsize=(14, 2.9)); ax.set_xlim(0, len(layers) * 2); ax.set_ylim(0, 3); ax.axis('off')
for i, (t, shape, col) in enumerate(layers):
    x = i * 2 + 0.1
    ax.add_patch(FancyBboxPatch((x, 1.0), 1.65, 1.2, boxstyle='round,pad=0.03,rounding_size=0.12', fc=col, ec='#888888', lw=0.8))
    ax.text(x + 0.825, 1.6, t, ha='center', va='center', fontsize=9.5)
    ax.text(x + 0.825, 0.7, shape, ha='center', va='center', fontsize=8.5, color=MUTED)
    if i < len(layers) - 1:
        ax.annotate('', xy=(x + 1.95, 1.6), xytext=(x + 1.72, 1.6), arrowprops=dict(arrowstyle='->', color='#555555'))
w1, w2 = 3 * c0 * k * k + 2 * c0, c0 * 2 * c0 * k * k + 4 * c0; wh = 2 * c0 * 16 + 1
ax.text(0.1, 2.65, f'Chosen smallest CNN: {int(cnn["parameters"]):,} trained weights (conv 1: {w1:,}, conv 2: {w2:,}, head: {wh:,}); '
        f'learning rate {p["lr"]:.4f}, weight decay {p["wd"]:.3f}, batch {p["batch"]}, class weight 4', fontsize=10.5, fontweight='bold')
fig.savefig(OUT / 'small_cnn_diagram.png', dpi=170, bbox_inches='tight'); plt.close(fig); print('saved cnn diagram')
