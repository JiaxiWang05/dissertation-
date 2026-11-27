"""24 Nov figures: training curves (figures/training_curves.png) and confusion matrices of the best models side by side
(figures/confusion_best.png), plus a summary of LoRA + 2 open blocks (lora_blocks_summary.csv).
Inputs: curves_*.json, small_cnn_oof.npz, g_lora_blocks.npz (this folder); ../after_17_Nov/final_chaksu_oof.npz and
final_chaksu_lora_oof.npz; ../after_22_Nov/g_mask025.npz, g_base.npz; ../after_10_Nov/step5_oof_scores.csv.
Threshold for every confusion matrix: the highest score that still finds 90 % of the glaucoma images (out-of-fold)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent; R = HERE.parent; OUT = HERE / 'figures'
sys.path.insert(0, str(R / 'after_10_Nov'))
TEXT, MUTED, BLUE, ORANGE, RED = '#1a1a1a', '#666666', '#2f6b9a', '#c0583a', '#b5443b'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def thr90(y, s):
    p = np.sort(s[y == 1])[::-1]; return p[int(np.ceil(0.9 * len(p))) - 1]


# ------------------------------------------------------------------ training curves
CUR = [('small_cnn', 'Small CNN from scratch'), ('blocks2', '2 open blocks'), ('lora', 'LoRA'),
       ('mask025', 'LoRA + disc-mask head'), ('lora_blocks', 'LoRA + 2 open blocks')]
FAIR = {'blocks2': (6, 8), 'lora': (5, 8), 'mask025': (5, 8)}         # fixed epochs of the fair runs (17 / 22 Nov logs)
if (HERE / 'g_lora_blocks.npz').exists():
    g = np.load(HERE / 'g_lora_blocks.npz'); f = np.concatenate([g[f'fixed{i}'] for i in range(3)]); FAIR['lora_blocks'] = (f.min(), f.max())
if (HERE / 'small_cnn_final.csv').exists():
    import ast
    FAIR['small_cnn'] = tuple(ast.literal_eval(pd.read_csv(HERE / 'small_cnn_final.csv')['fixed epochs'][0])) if 'fixed epochs' in pd.read_csv(HERE / 'small_cnn_final.csv') else None
avail = [(k, n) for k, n in CUR if (HERE / f'curves_{k}.json').exists()]
if avail:
    fig, axes = plt.subplots(2, len(avail), figsize=(3.6 * len(avail), 6.4), squeeze=False)
    for j, (k, name) in enumerate(avail):
        d = json.load(open(HERE / f'curves_{k}.json')); folds = d['folds']
        ep = np.array([h['epoch'] for h in folds[0]])
        tl = np.array([[h['train_loss'] for h in f] for f in folds]); vl = np.array([[h['val_loss'] for h in f] for f in folds])
        va = np.array([[h['val_auprc'] for h in f] for f in folds])
        a = axes[0, j]
        for f_ in range(len(folds)):
            a.plot(ep, tl[f_], color=BLUE, alpha=0.18, lw=0.8); a.plot(ep, vl[f_], color=ORANGE, alpha=0.18, lw=0.8)
        a.plot(ep, tl.mean(0), color=BLUE, lw=2, label='training loss'); a.plot(ep, vl.mean(0), color=ORANGE, lw=2, label='validation loss')
        b = axes[1, j]
        for f_ in range(len(folds)):
            b.plot(ep, va[f_], color=RED, alpha=0.18, lw=0.8)
        b.plot(ep, va.mean(0), color=RED, lw=2, label='validation AUPRC'); b.axhline(0.14, color='#999999', ls=':', lw=1)
        best = int(ep[np.argmin(vl.mean(0))])
        for a_ in (a, b):
            if FAIR.get(k):
                a_.axvspan(FAIR[k][0], FAIR[k][1], color='#2e8b57', alpha=0.12, lw=0)
            a_.axvline(best, color='#2e8b57', ls='--', lw=1)
        a.set_title(f"{name}\n{d.get('trainable', '')} trained weights" if d.get('trainable') else name, fontsize=10.5)
        a.set_ylim(0, min(2.5, max(tl.max(), vl.max()) * 1.05)); b.set_ylim(0, 1); b.set_xlabel('epoch')
        if j == 0:
            a.set_ylabel('weighted cross-entropy'); b.set_ylabel('validation AUPRC'); a.legend(frameon=False, fontsize=8.5)
    fig.suptitle('Training curves, seed 0, all 5 folds trained without early stopping (thin: each fold; thick: mean). Dashed: epoch of the lowest mean '
                 'validation loss; green band: epochs used in the fair runs; dotted: AUPRC of guessing (0.14)', fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / 'training_curves.png', dpi=170); plt.close(fig)
    print('curves:', [k for k, _ in avail])

# ------------------------------------------------------------------ confusion matrices, best models side by side
ref = pd.read_csv(R / 'after_10_Nov' / 'step5_oof_scores.csv', float_precision='round_trip')
ids = ref['image_id'].to_numpy().astype(str); y = ref['label'].to_numpy()
M = [('Frozen features + SVM', ref['svm'].to_numpy())]
src = [('Small CNN from scratch', HERE / 'small_cnn_oof.npz', 'seed0'), ('2 open blocks', R / 'after_17_Nov' / 'final_chaksu_oof.npz', 'unfreeze'),
       ('LoRA', R / 'after_17_Nov' / 'final_chaksu_lora_oof.npz', 'lora'), ('LoRA + disc-mask head', R / 'after_22_Nov' / 'g_mask025.npz', 'seed0'),
       ('LoRA + 2 open blocks', HERE / 'g_lora_blocks.npz', 'seed0')]
for name, f, key in src:
    if f.exists():
        z = np.load(f); assert (z['image_id'].astype(str) == ids).all(), name; M.append((name, z[key]))
fig, axes = plt.subplots(1, len(M), figsize=(3.3 * len(M), 3.9))
rows = []
for a, (name, s) in zip(np.atleast_1d(axes), M):
    t = thr90(y, s); pred = s >= t
    m = np.array([[((y == 0) & ~pred).sum(), ((y == 0) & pred).sum()], [((y == 1) & ~pred).sum(), ((y == 1) & pred).sum()]])
    pct = 100 * m / m.sum(1, keepdims=True)
    a.imshow(pct, cmap='Blues', vmin=0, vmax=160)
    for i in range(2):
        for jj in range(2):
            a.text(jj, i, f'{m[i, jj]}\n({pct[i, jj]:.0f} %)', ha='center', va='center', fontsize=11, color=TEXT)
    a.set_xticks([0, 1]); a.set_xticklabels(['healthy', 'glaucoma']); a.set_yticks([0, 1]); a.set_yticklabels(['healthy', 'glaucoma'])
    a.set_xlabel('predicted'); a.set_title(f"{name}\nAUPRC {average_precision_score(y, s):.2f}, AUROC {roc_auc_score(y, s):.2f}", fontsize=10)
    rows.append({'model': name, 'healthy flagged %': pct[0, 1], 'glaucoma found %': pct[1, 1], 'AUPRC': average_precision_score(y, s)})
np.atleast_1d(axes)[0].set_ylabel('true')
fig.suptitle('Out-of-fold confusion matrices on the 1,141 Chákṣu images at the threshold that finds 90 % of glaucoma '
             '(image models: seed 0). The comparison is in the healthy row.', fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / 'confusion_best.png', dpi=170); plt.close(fig)
pd.DataFrame(rows).to_csv(HERE / 'confusion_best.csv', index=False); print(pd.DataFrame(rows).round(3).to_string())

# ------------------------------------------------------------------ LoRA + 2 open blocks: paired comparisons
if (HERE / 'g_lora_blocks.npz').exists():
    from common import bootstrap_ci, paired_difference, load_data, cv_split  # noqa: E402
    import os
    os.chdir(R / 'after_10_Nov'); lab, _, _ = load_data(); keep, _ = cv_split(lab); os.chdir(HERE)
    groups = lab[keep]['eye'].to_numpy()
    g = np.load(HERE / 'g_lora_blocks.npz'); base = np.load(R / 'after_22_Nov' / 'g_base.npz')
    aps = [average_precision_score(y, g[f'seed{i}']) for i in range(3)]
    out = {'AUPRC 3-seed mean': np.mean(aps), 'range': [round(min(aps), 3), round(max(aps), 3)],
           'AUROC seed 0': roc_auc_score(y, g['seed0'])}
    for nm, s2 in (('LoRA rerun', base['seed0']), ('frozen SVM', ref['svm'].to_numpy())):
        for met, f_ in (('AUPRC', average_precision_score), ('AUROC', roc_auc_score)):
            dd, ci = paired_difference(f_, y, g['seed0'], s2, groups); out[f'{met} minus {nm}'] = f'{dd:+.3f} ({ci[0]:.3f} to {ci[1]:.3f})'
    pd.DataFrame([out]).to_csv(HERE / 'lora_blocks_summary.csv', index=False); print(out)
