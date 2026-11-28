"""Column-width figure for the literature review: validation AUPRC against training images (data_size.csv)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
HERE = Path(__file__).resolve().parent
s = pd.read_csv(HERE / 'data_size.csv')
M = [('svm', 'Frozen + SVM', '#7a7a7a'), ('blocks2', '2 open blocks', '#2f6b9a'), ('lora', 'LoRA', '#c0583a'),
     ('mask025', 'LoRA + mask head', '#2e8b57'), ('lora_blocks', 'LoRA + 2 blocks', '#7b4fa0'), ('cnn', 'Small CNN', '#b08d2a')]
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(3.45, 1.45))
for m, n, c in M:
    d = s[s['model'] == m].sort_values('n')
    ax.fill_between(d['n'], d['vlo'], d['vhi'], color=c, alpha=0.15, lw=0)
    ax.plot(d['n'], d['val'], '-o', color=c, lw=1.2, ms=2.5, label=n)
ax.axhline(0.14, color='#999999', ls=':', lw=0.8)
ax.set_xscale('log'); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([92, 228, 456, 684, 913]); ax.set_xticklabels(['92', '228', '456', '684', '913'])
ax.set_xlabel('training images per fold (10, 25, 50, 75, 100 %)'); ax.set_ylabel('out-of-fold AUPRC')
ax.set_ylim(0.1, 0.78); ax.legend(frameon=False, fontsize=6, ncol=2, loc='upper left')
fig.tight_layout(pad=0.2)
out = HERE.parent / 'reports' / 'figures' / 'fig_data_size.pdf'; fig.savefig(out); print('saved', out)
