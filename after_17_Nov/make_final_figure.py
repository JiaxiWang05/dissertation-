"""Figure 3 (17 Nov work): fair out-of-fold AUPRC of the image models against the frozen-feature reference.
Reads every final_<tag>.csv present (chaksu, chaksu_pseudo, fairvlmed); nothing is trained here.
python make_final_figure.py  ->  figures/fig3_final_comparison.png"""
import ast
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
PANELS = [('chaksu', 'Chákṣu, 5 folds (1,141 images)'),
          ('chaksu_pseudo', 'Chákṣu, likely fellow eyes in the same fold'),
          ('fairvlmed', 'FairVLMed, 5 folds (8,000 patients)')]
NAMES = {'cnn': 'CNN from scratch', 'unfreeze': 'RETFound-Green,\nlast 2 blocks open', 'lora': 'RETFound-Green,\nLoRA, all 12 blocks'}
REF = {'svm': 'SVM', 'logreg': 'Logistic regression', 'mlp_small': 'Small MLP', 'knn': 'kNN'}
def load(t):
    d = pd.read_csv(f'final_{t}.csv')
    if Path(f'final_{t}_lora.csv').exists():          # LoRA was run separately (MODELS=lora)
        d = pd.concat([d, pd.read_csv(f'final_{t}_lora.csv')], ignore_index=True)
    return d


panels = [(t, title, load(t)) for t, title in PANELS if Path(f'final_{t}.csv').exists()]
fig, axs = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 3.0), squeeze=False)
for ax, (tag, title, d) in zip(axs[0], panels):
    ref_name = REF.get(d['best frozen-feature model'][0], d['best frozen-feature model'][0])
    labels = [f'Frozen features\n({ref_name})'] + [NAMES[m] for m in d['model']]
    mean = [d['frozen AUPRC'][0]] + list(d['AUPRC 3 seeds mean'])
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys[0], mean[0], color='0.7', height=0.6)
    for y, (_, r) in zip(ys[1:], d.iterrows()):
        lo, hi = ast.literal_eval(r['AUPRC 3 seeds range'])
        ax.barh(y, r['AUPRC 3 seeds mean'], color='tab:blue', height=0.6)
        ax.plot([lo, hi], [y, y], c='k', lw=1.2)
        ax.plot(r['best search AUPRC'], y, 'D', mfc='none', mec='tab:red', ms=6)
        dlo, dhi = ast.literal_eval(r['diff CI'])
        tied = dlo <= 0 <= dhi
        ax.text(0.02, y, f'{r["AUPRC 3 seeds mean"]:.2f}', va='center', color='white', fontsize=8.5)
        ax.text(max(hi, r['best search AUPRC']) + 0.04, y, 'tied' if tied else 'better' if dlo > 0 else 'worse',
                va='center', fontsize=8.5, color='k' if tied else 'tab:green' if dlo > 0 else 'tab:red')
    ax.text(0.02, ys[0], f'{mean[0]:.2f}', va='center', fontsize=8.5)
    ax.set_yticks(ys, labels)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('out-of-fold AUPRC')
fig.text(0.5, 0.005, textwrap.fill('Bar = fair out-of-fold AUPRC, mean of 3 seeds; black line = range over seeds; red diamond = search '
         'score (optimistic). tied / better / worse: 95 % CI of the paired AUPRC difference to the frozen model (seed 0).', 60 * len(panels)),
         ha='center', fontsize=7.5)
fig.tight_layout(rect=(0, 0.12 if len(panels) == 1 else 0.06, 1, 1))
Path('figures').mkdir(exist_ok=True)
fig.savefig('figures/fig3_final_comparison.png', dpi=200)
print('made figures/fig3_final_comparison.png from', [t for t, _, _ in panels])
