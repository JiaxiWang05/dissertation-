"""17 Nov task: confusion matrices as percentages (so Chakshu and FairVLMed can be compared) and Chakshu split by camera.
Rows are normalised: glaucoma row = % of glaucoma images found / missed (sensitivity / miss rate); healthy row = % of
healthy images flagged / cleared (false-alarm rate / specificity). Out-of-fold scores only; no test set is used.
Inputs: ../after_10_Nov/step5_confusion.csv, step5_oof_scores.csv; ../fairvlmed/step5_confusion.csv (counts only).
python confusion_percent.py  ->  confusion_percent.csv, confusion_by_camera.csv, figures/fig1..fig2"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, '../after_10_Nov')
NAMES = {'logreg': 'Logistic regression', 'svm': 'SVM', 'knn': 'kNN', 'mlp_small': 'Small MLP'}


def pct(r):
    g, h = r['TP'] + r['FN'], r['FP'] + r['TN']
    return {'glaucoma found %': 100 * r['TP'] / g, 'glaucoma missed %': 100 * r['FN'] / g,
            'healthy flagged %': 100 * r['FP'] / h, 'healthy cleared %': 100 * r['TN'] / h}


rows = []
for ds, f in [('Chakshu', '../after_10_Nov/step5_confusion.csv'), ('FairVLMed', '../fairvlmed/step5_confusion.csv')]:
    c = pd.read_csv(f)
    c = c[c['classifier'].isin(NAMES)]
    for _, r in c.iterrows():
        rows.append({'dataset': ds, 'classifier': r['classifier'], 'threshold rule': r['threshold rule'], **pct(r)})
P = pd.DataFrame(rows)
P.to_csv('confusion_percent.csv', index=False)
print(P.round(1).to_string(index=False))

# Chakshu by camera, at each classifier's own out-of-fold thresholds
oof = pd.read_csv('../after_10_Nov/step5_oof_scores.csv', float_precision='round_trip')
cam = pd.read_csv('../pipeline/chaksu_labels.csv').set_index('image_id')['camera']
oof['camera'] = oof['image_id'].map(cam)
thr = pd.read_csv('../after_10_Nov/step5_confusion.csv').set_index(['classifier', 'threshold rule'])['threshold']
rows = []
for n in NAMES:
    for rule in ['90% sensitivity', "Youden's J"]:
        t = thr[(n, rule)]
        for camname, g in oof.groupby('camera'):
            pred, y = (g[n] >= t).astype(int), g['label']
            r = {'TP': int(((pred == 1) & (y == 1)).sum()), 'FN': int(((pred == 0) & (y == 1)).sum()),
                 'FP': int(((pred == 1) & (y == 0)).sum()), 'TN': int(((pred == 0) & (y == 0)).sum())}
            rows.append({'classifier': n, 'threshold rule': rule, 'camera': camname, 'images': len(g),
                         'glaucoma': int(y.sum()), **r, **pct(r)})
C = pd.DataFrame(rows)
C.to_csv('confusion_by_camera.csv', index=False)
print(C.round(1).to_string(index=False))


def draw(ax, r, title):
    m = np.array([[r['glaucoma found %'], r['glaucoma missed %']], [r['healthy flagged %'], r['healthy cleared %']]])
    ax.imshow(m, cmap='Blues', vmin=0, vmax=100)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f'{v:.0f} %', ha='center', va='center', fontsize=11, color='white' if v > 60 else 'black')
    ax.set_xticks([0, 1], ['called\nglaucoma', 'called\nhealthy'], fontsize=8)
    ax.set_yticks([0, 1], ['glaucoma', 'healthy'], fontsize=8)
    ax.set_title(title, fontsize=8.5)


for rule, tag in [('90% sensitivity', 'sens90'), ("Youden's J", 'youden')]:
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.2))
    for i, ds in enumerate(['Chakshu', 'FairVLMed']):
        for j, n in enumerate(NAMES):
            r = P[(P['dataset'] == ds) & (P['classifier'] == n) & (P['threshold rule'] == rule)].iloc[0]
            draw(axs[i, j], r, f'{"Chákṣu" if ds == "Chakshu" else ds}: {NAMES[n]}')
    fig.suptitle(f'Confusion matrices in percent of each true class, {rule} threshold (out-of-fold; '
                 'Chákṣu 1,141 images, FairVLMed 8,000)', fontsize=10)
    fig.tight_layout(); fig.savefig(f'figures/fig1_confusion_percent_{tag}.png', dpi=200); plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(10, 6.4))
for i, rule in enumerate(['90% sensitivity', "Youden's J"]):
    for j, camname in enumerate(['Remidio', 'Bosch', 'Forus']):
        r = C[(C['classifier'] == 'logreg') & (C['threshold rule'] == rule) & (C['camera'] == camname)].iloc[0]
        draw(axs[i, j], r, f'{camname} ({r["images"]} images, {r["glaucoma"]} glaucoma)\n{rule} threshold')
fig.suptitle('Chákṣu, logistic regression (the chosen model): confusion matrices by camera, percent of each true class',
             fontsize=10)
fig.tight_layout(); fig.savefig('figures/fig2_confusion_by_camera.png', dpi=200); plt.close(fig)
print('figures saved')
