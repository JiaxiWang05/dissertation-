"""Computing cost of every Hamilton task, from the Slurm accounting (sacct) of all jobs since October.
Inputs (refresh on Hamilton, then copy here):
  sacct -u $USER -S 2026-10-01 -X --parsable2 --format=JobID,JobName%30,Partition,State,Elapsed,ElapsedRaw,AllocTRES%80,WorkDir%80,Start > sacct_raw.txt
  sacct_modes.txt: first 'host' line of each log file (which script a generic job name ran)
Cost measures: wall-clock hours; GPU hours in whole-H200 units (a full H200 NVL = 1, a 1g.18gb MIG slice = 1/7 of one);
CPU core-hours (cores allocated x hours). Failed and cancelled jobs count, they used the machine too; the interactive
Jupyter session is left out.  ->  compute_cost.csv, figures/compute_cost.png"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = HERE / 'figures'
d = pd.read_csv(HERE / 'sacct_raw.txt', sep='|')
d = d[d['ElapsedRaw'] > 0].copy()
d['dir'] = d['WorkDir'].fillna('').str.replace(r'.*/glaucoma/?', '', regex=True).replace('', 'root')
d = d[~d['JobName'].str.contains('jupyter')]
tres = d['AllocTRES'].fillna('')
d['mig'] = tres.str.contains('1g.18gb'); d['gpu'] = tres.str.contains('gres/gpu=')
d['cores'] = tres.str.extract(r'cpu=(\d+)')[0].astype(float)
d['h'] = d['ElapsedRaw'] / 3600
d['gpu_h'] = d['h'] / 7
d.loc[d['gpu'] & ~d['mig'], 'gpu_h'] = d['h']
d.loc[~d['gpu'], 'gpu_h'] = 0
d['core_h'] = d['h'] * d['cores']

modes = {}
for line in open(HERE / 'sacct_modes.txt'):
    f, _, rest = line.partition(':')
    m = re.search(r'_(\d+)_(\d+)\.out$', f)
    if m:
        jid = m.group(1) if m.group(2) == '4294967294' else f'{m.group(1)}_{m.group(2)}'
        modes[jid] = rest
d['mode'] = d['JobID'].astype(str).map(modes).fillna('')


def task(r):
    n, dr, md = r['JobName'], r['dir'], r['mode']
    if n == 'extract':
        return '12 Oct', 'Feature extraction (RETFound-Green, all images)'
    if dr == 'retfound_green':
        return '19-27 Oct', 'First experiments: frozen vs fine-tuned, ablations, statistics'
    if dr in ('after_3_Nov', 'learn'):
        return '3-8 Nov', 'MLP overfitting, cross-validation, label noise, edge checks'
    if dr == 'after_10_Nov':
        return '10 Nov', 'Frozen-feature classifier searches (Chákṣu)'
    if dr == 'fairvlmed':
        return '11-12 Nov', 'FairVLMed: features and frozen-feature classifiers'
    if n in ('n17f1', 'n17cpu') or (dr == 'after_17_Nov' and n == 'python'):
        return '17 Nov', 'Class-weight search'
    if 'cnn_cv.py' in md:
        return '17 Nov', 'CNN from scratch, 5 blocks: Optuna search'
    if 'unfreeze_cv.py' in md:
        return '17 Nov', '2 open blocks: Optuna search'
    if 'lora_cv.py' in md:
        return '17 Nov', 'LoRA: Optuna search'
    if 'final_compare.py' in md or (dr == 'after_17_Nov'):
        return '17 Nov', 'Fair 3-seed scoring of the 17 Nov models'
    if n.startswith('g_') or (n == 'n22gpu' and 'grounding' in md):
        return '22 Nov', 'Disc grounding variants and unseen camera'
    if dr == 'after_22_Nov':
        return '22 Nov', 'Layer-by-layer features, evidence maps'
    if n in ('cnn_small', 'final_cnn', 'smoke_cnn'):
        return '24 Nov', 'Smallest CNN: search and fair score'
    if n.startswith('curve_'):
        return '24 Nov', 'Training curves (5 models)'
    if n in ('final_lb', 'smoke_vit'):
        return '24 Nov', 'LoRA + 2 open blocks: fair score'
    if n.startswith('ds_'):
        return '24 Nov', 'Training-set size experiment'
    return '?', f'{dr}/{n}'


d[['when', 'task']] = d.apply(lambda r: pd.Series(task(r)), axis=1)
d['hw'] = d.apply(lambda r: 'MIG 1/7 H200' if r['mig'] else ('H200 NVL' if r['gpu'] else 'CPU'), axis=1)
order = list(dict.fromkeys(d.sort_values('Start')['task']))
t = d.groupby('task').agg(when=('when', 'first'), jobs=('h', 'size'), wall_h=('h', 'sum'), gpu_h=('gpu_h', 'sum'),
                          core_h=('core_h', 'sum'), hardware=('hw', lambda s: ' + '.join(sorted(set(s)))),
                          longest_h=('h', 'max')).loc[order].reset_index()
t.to_csv(HERE / 'compute_cost.csv', index=False)
print(t.round(2).to_string()); print('total GPU h (H200 units)', round(t.gpu_h.sum(), 2), '| core-h', round(t.core_h.sum(), 1),
                                     '| jobs', int(t.jobs.sum()))

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 0.42 * len(t) + 1.8), sharey=True, gridspec_kw={'wspace': 0.05})
ys = range(len(t))[::-1]
for y, r in zip(ys, t.itertuples()):
    col = '#c0583a' if 'MIG' in r.hardware else ('#2f6b9a' if 'H200 NVL' in r.hardware else '#9a9a9a')
    a.barh(y, r.gpu_h, color=col, height=0.6)
    a.text(max(r.gpu_h, 0) + 0.05, y, f'{r.gpu_h:.2f} h' if r.gpu_h else 'CPU only', va='center', fontsize=8.5,
           color='#1a1a1a' if r.gpu_h else '#888888')
    b.barh(y, r.core_h, color='#9a9a9a', height=0.6)
    b.text(r.core_h + 0.5, y, f'{r.core_h:.0f} core-h  ({r.jobs} jobs, {r.wall_h:.1f} h wall)', va='center', fontsize=8.5)
a.set_yticks(list(ys)); a.set_yticklabels([f'{r.when}   {r.task}' for r in t.itertuples()], fontsize=9.5)
a.set_xlabel('GPU hours (in whole H200 GPUs; a MIG slice counts 1/7)'); b.set_xlabel('CPU core-hours')
a.set_xlim(0, t.gpu_h.max() * 1.35); b.set_xlim(0, t.core_h.max() * 1.6)
from matplotlib.patches import Patch  # noqa: E402
a.legend(handles=[Patch(color='#2f6b9a', label='full H200 NVL'), Patch(color='#c0583a', label='MIG slice, 1/7 of an H200 (18 GB)')],
         frameon=False, loc='lower right', fontsize=8.5)
fig.suptitle(f'Computing cost of every task on Hamilton (Slurm accounting). Total: {t.gpu_h.sum():.1f} H200 GPU-hours, '
             f'{t.core_h.sum():.0f} CPU core-hours, {int(t.jobs.sum())} jobs', x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.savefig(OUT / 'compute_cost.png', dpi=170, bbox_inches='tight'); print('saved')
