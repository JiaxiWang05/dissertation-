"""Figures for the Notion pages on the work since the 17 Nov meeting. Nothing is trained here; numbers come from the
saved result files (after_10_Nov, fairvlmed, after_17_Nov). Run on Hamilton (glaucoma env) from this folder:
python make_17nov_figures.py  ->  since_17_nov.png, class_weight.png, searches.png, datasets.png, chaksu_ladder.png"""
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

HERE = Path(__file__).parent
A17 = HERE.parent / 'after_17_Nov'
GREY, LEARN, RESULT = '#eeeeee', '#f4ecdc', '#f6e0e0'
EDGE, TEXT, MUTED = '#444444', '#1a1a1a', '#666666'
BLUE, RED, GRAY = '#2f6b9a', '#b5443b', '#9a9a9a'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def final(tag):
    f = A17 / f'final_{tag}.csv'
    if not f.exists():
        return None
    d = pd.read_csv(f)
    lo = A17 / f'final_{tag}_lora.csv'                 # LoRA was run separately (MODELS=lora)
    if lo.exists():
        d = pd.concat([d, pd.read_csv(lo)], ignore_index=True)
    return d.set_index('model')


def fair_text(tag, model):
    d = final(tag)
    if d is None or model not in d.index:
        return None
    r = d.loc[model]
    lo, hi = ast.literal_eval(r['diff CI'])
    verdict = 'better than' if lo > 0 else 'worse than' if hi < 0 else 'tied with'
    return r, verdict


# ---------------------------------------------------------------- block diagram
def block_diagram():
    ch, fv, ps = final('chaksu'), final('fairvlmed'), final('chaksu_pseudo')

    def res(d, m, frozen_label):
        if d is None:
            return 'fair comparison\nstill running'
        r = d.loc[m]
        lo, hi = ast.literal_eval(r['diff CI'])
        v = 'better than' if lo > 0 else 'worse than' if hi < 0 else 'tied with'
        return f'fair AUPRC {r["AUPRC 3 seeds mean"]:.2f}\n(3 seeds {ast.literal_eval(r["AUPRC 3 seeds range"])[0]:.2f}-' \
               f'{ast.literal_eval(r["AUPRC 3 seeds range"])[1]:.2f});\n{v} {frozen_label} {r["frozen AUPRC"]:.2f}'

    pseudo = 'running' if ps is None else (
        f"fine-tuned {ps.loc['unfreeze', 'AUPRC 3 seeds mean']:.2f}: still better\n"
        f"than frozen {ps['frozen AUPRC'].iloc[0]:.2f}; CNN {ps.loc['cnn', 'AUPRC 3 seeds mean']:.2f} (worse).\n"
        "Not a fellow-eye effect")
    rows = [
        ('Confusion matrices in %;\nChákṣu split by camera',
         'Rows normalised by the true\nclass; out-of-fold scores;\nthree Chákṣu cameras',
         'At 90 % sensitivity: healthy\nflagged ~50 % (Chákṣu),\n~60 % (FairVLMed); same on\nevery camera (47-55 %)'),
        ('Class weight 6.1 too strong?\nLet Optuna tune it (F1)',
         'Weights 0.25-12.3 on logistic\nregression and small MLP;\nOptuna search with F1 at 0.5\n(200 trials)',
         'Weight moves the threshold,\nnot the ranking: AUPRC flat\nfrom w = 2 to 12; F1 at 0.5\nbest near w = 3-4'),
        ('Own CNN from scratch\nas the baseline (Chákṣu)',
         'Optuna: blocks, channels,\nkernel, dropout, lr, weight;\n61 trials, then fair retraining\nwith 3 seeds',
         'search 0.54, but ' + res(ch, 'cnn', 'frozen SVM')),
        ('Repeat on FairVLMed\n(50 % glaucoma)',
         'Same CNN search on\n8,000 patients (31 trials),\nfair retraining, 3 seeds',
         'search 0.82, ' + res(fv, 'cnn', 'frozen MLP') if fv is not None else 'search 0.82;\nfair comparison\nstill running'),
        ('If time: open RETFound-\nGreen blocks (0-3)',
         'Optuna chooses open blocks,\nhead and learning rates;\n26 trials, then fair retraining',
         'search 0.59 (2 blocks), ' + res(ch, 'unfreeze', 'frozen SVM')),
        ('Check: fellow eyes in the\nsame fold (pseudo-patients)',
         'Folds rebuilt so likely\nfellow eyes (433 pairs)\nstay together; same models',
         pseudo),
    ]
    W, x0, num_w, cgap = 11.0, 0.15, 0.5, 0.34
    x1 = W - 0.15
    cx0 = x0 + num_w + 0.1
    cw = (x1 - cx0 - 2 * cgap) / 3
    cols = [cx0 + i * (cw + cgap) for i in range(3)]
    rh, rgap = 1.25, 0.24
    H = 0.3 + 0.45 + 0.55 + len(rows) * (rh + rgap) + 0.1
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    def box(x, y, w, h, t, fc, ec=EDGE, lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=11, color=TEXT, linespacing=1.3)

    top = H - 0.3
    ax.text(x0, top, 'Since the 17 November meeting', fontsize=13, fontweight='bold', va='top', color=TEXT)
    hy = top - 0.75
    for cx, t in zip(cols, ['Asked on 17 November', 'What I did', 'Result']):
        ax.text(cx + cw / 2, hy, t, ha='center', va='center', fontsize=10.5, color=MUTED, fontweight='bold')
    ry = hy - 0.3 - rh
    for k, r in enumerate(rows):
        yc = ry + rh / 2
        ax.add_patch(Circle((x0 + num_w / 2, yc), 0.19, fc='#555555', ec='#555555'))
        ax.text(x0 + num_w / 2, yc, str(k + 1), ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        for i, (cx, t, c) in enumerate(zip(cols, r, [GREY, LEARN, RESULT])):
            strong = k == 4 and i == 2
            box(cx, ry, cw, rh, t, c, ec='#8b2e2e' if strong else EDGE, lw=2.0 if strong else 1.2)
            if i:
                ax.annotate('', xy=(cx - 0.03, yc), xytext=(cx - cgap + 0.03, yc),
                            arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.3, mutation_scale=13))
        ry -= rh + rgap
    fig.savefig(HERE / 'since_17_nov.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- class weight
def class_weight():
    g = pd.read_csv(A17 / 'weight_grid.csv').groupby(['model', 'weight']).mean(numeric_only=True).reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, model in zip(axs[:2], ['logistic regression', 'small MLP']):
        d = g[g['model'] == model]
        ax.plot(d['weight'], d['sens@0.5'], 'o-', c=RED, label='sensitivity')
        ax.plot(d['weight'], d['spec@0.5'], 's-', c=BLUE, label='specificity')
        ax.plot(d['weight'], d['F1@0.5'], '^-', c='k', label='F1')
        ax.axvline(6.13, ls=':', c=GRAY); ax.text(6.3, 0.03, 'w = 6.13\n(used so far)', fontsize=8, color=MUTED)
        ax.set_xscale('log'); ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ['0.25', '0.5', '1', '2', '4', '8'])
        ax.set_ylim(0, 1.03); ax.set_xlabel('weight on the glaucoma class')
        ax.set_title(f'{model[0].upper() + model[1:]}: at the default 0.5 threshold', fontsize=10)
    axs[0].legend(frameon=False, fontsize=8.5, loc='center left')
    for model, c, m in [('logistic regression', BLUE, 'o'), ('small MLP', RED, 's')]:
        d = g[g['model'] == model]
        axs[2].plot(d['weight'], d['AUPRC'], m + '-', c=c, label=model)
    axs[2].axhline(0.14, ls=':', c=GRAY); axs[2].text(0.26, 0.15, 'guessing 0.14', fontsize=8, color=MUTED)
    axs[2].set_xscale('log'); axs[2].set_xticks([0.25, 0.5, 1, 2, 4, 8], ['0.25', '0.5', '1', '2', '4', '8'])
    axs[2].set_ylim(0, 0.5); axs[2].set_xlabel('weight on the glaucoma class')
    axs[2].set_title('Ranking (AUPRC) hardly moves above w = 2', fontsize=10); axs[2].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(HERE / 'class_weight.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- searches
def searches():
    import optuna
    sys.path.insert(0, str(A17))
    import os
    cwd = os.getcwd(); os.chdir(A17)
    from img_common import storage
    st = {n: optuna.load_study(study_name=n, storage=storage(n)).trials_dataframe() for n in
          ['cnn_chaksu', 'cnn_fairvlmed', 'unfreeze_chaksu']}
    os.chdir(cwd)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (n, title, ref, reflab) in zip(axs[:3], [
            ('cnn_chaksu', 'CNN from scratch, Chákṣu', 0.427, 'frozen SVM 0.43'),
            ('cnn_fairvlmed', 'CNN from scratch, FairVLMed', 0.805, 'frozen MLP 0.81'),
            ('unfreeze_chaksu', 'RETFound-Green, blocks opened, Chákṣu', 0.427, 'frozen SVM 0.43')]):
        d = st[n]; d = d[d['state'] == 'COMPLETE'].sort_values('number')
        ax.plot(range(1, len(d) + 1), d['value'], 'o', ms=4, c=GRAY, label='one trial')
        ax.plot(range(1, len(d) + 1), d['value'].cummax(), c=BLUE, lw=2, label='best so far')
        ax.axhline(ref, ls='--', c=RED); ax.text(1, ref + 0.006, reflab, color=RED, fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_xlabel('trial'); ax.set_ylabel('5-fold mean AUPRC (search)')
    axs[0].legend(frameon=False, fontsize=8, loc='lower right')
    d = st['unfreeze_chaksu']; d = d[d['state'] == 'COMPLETE']
    ks = sorted(d['params_k'].unique())
    for i, k in enumerate(ks):
        v = d.loc[d['params_k'] == k, 'value']
        axs[3].scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, c=BLUE, s=18)
        axs[3].plot([i - 0.25, i + 0.25], [v.max()] * 2, c='k', lw=1.5)
    axs[3].set_xticks(range(len(ks)), [f'{int(k)} open\n({(d["params_k"] == k).sum()} trial' + ('s)' if (d["params_k"] == k).sum() > 1 else ')') for k in ks])
    axs[3].axhline(0.427, ls='--', c=RED)
    axs[3].set_title('How many RETFound-Green blocks to open', fontsize=10); axs[3].set_ylabel('search AUPRC')
    fig.tight_layout()
    fig.savefig(HERE / 'searches.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- datasets
def datasets():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw={'width_ratios': [1.3, 0.9, 1.2]})
    names, n, g = ['Chákṣu', 'PAPILA', 'FairVLMed'], [1345, 420, 10000], [188, 87, 5048]
    pct = [100 * b / a for a, b in zip(n, g)]
    axs[0].barh(names[::-1], [100] * 3, color='#e6e6e6')
    axs[0].barh(names[::-1], pct[::-1], color=RED)
    for i, (a, b, q) in enumerate(zip(n[::-1], g[::-1], pct[::-1])):
        axs[0].text(q + 2, i, f'{q:.0f} % glaucoma  ({b:,} of {a:,} images)', va='center', fontsize=8.5)
    axs[0].set_xlim(0, 100); axs[0].set_xlabel('% of images'); axs[0].set_title('Size and share of glaucoma', fontsize=10)
    cams, cn = ['Remidio', 'Bosch', 'Forus'], [1074, 145, 126]
    axs[1].bar(cams, cn, color=[BLUE, '#7fa7c7', '#c4d6e5'])
    for i, v in enumerate(cn):
        axs[1].text(i, v + 15, str(v), ha='center', fontsize=9)
    axs[1].set_title('Chákṣu: three cameras', fontsize=10); axs[1].set_ylabel('images')
    parts = [('Chákṣu', [939, 202, 204]), ('FairVLMed', [7000, 1000, 2000])]
    for i, (nm, p) in enumerate(parts):
        tot = sum(p); left = 0
        for v, c, lab in zip(p, [BLUE, '#7fa7c7', '#dddddd'], ['training', 'validation', 'test (held out)']):
            axs[2].barh(i, 100 * v / tot, left=left, color=c, label=lab if i == 0 else None)
            axs[2].text(left + 50 * v / tot, i, f'{v:,}' if v < 1000 else f'{v / 1000:.0f}k', ha='center', va='center', fontsize=8.5)
            left += 100 * v / tot
    axs[2].set_yticks([0, 1], [p[0] for p in parts]); axs[2].set_xlabel('% of images')
    axs[2].set_title('Splits (5-fold CV on training + validation)', fontsize=10)
    axs[2].legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.45, -0.3), ncol=3)
    fig.tight_layout()
    fig.savefig(HERE / 'datasets.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- where Chákṣu stands
def ladder():
    s5 = pd.read_csv(HERE.parent / 'after_10_Nov' / 'step5_comparison.csv').set_index('classifier')
    ch = final('chaksu')
    rows = [('Guessing', 0.14, None, '#dddddd')]
    for k, lab in [('knn', 'kNN, frozen features'), ('mlp_small', 'Small MLP, frozen features'),
                   ('logreg', 'Logistic regression, frozen features'), ('svm', 'SVM, frozen features')]:
        rows.append((lab, s5.loc[k, 'AUPRC'], s5.loc[k, 'AUROC'], GRAY))
    if ch is not None:
        rows.append(('CNN from scratch (no foundation model)', ch.loc['cnn', 'AUPRC 3 seeds mean'], ch.loc['cnn', 'AUROC seed 0'], '#7fa7c7'))
        rows.append(('RETFound-Green, last 2 blocks opened', ch.loc['unfreeze', 'AUPRC 3 seeds mean'], ch.loc['unfreeze', 'AUROC seed 0'], BLUE))
        if 'lora' in ch.index:
            rows.append(('RETFound-Green, LoRA in all 12 blocks', ch.loc['lora', 'AUPRC 3 seeds mean'], ch.loc['lora', 'AUROC seed 0'], '#1d5f8a'))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.0))
    ys = np.arange(len(rows))
    for y, (lab, ap, auc, c) in zip(ys, rows):
        ax.barh(y, ap, color=c, height=0.62)
        ax.text(ap + 0.008, y, f'{ap:.2f}' + (f'   (AUROC {auc:.2f})' if auc else ''), va='center', fontsize=9)
    ax.set_yticks(ys, [r[0] for r in rows]); ax.set_xlim(0, 0.75)
    ax.set_xlabel('out-of-fold AUPRC, Chákṣu (1,141 training + validation images, 160 glaucoma)')
    ax.set_title('Chákṣu: out-of-fold AUPRC of every model (image models: mean of 3 seeds)', fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / 'chaksu_ladder.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- three approaches
def approaches():
    ch = final('chaksu')
    cnn = f"{ch.loc['cnn', 'AUPRC 3 seeds mean']:.2f}" if ch is not None else '...'
    unf = f"{ch.loc['unfreeze', 'AUPRC 3 seeds mean']:.2f}" if ch is not None else '...'
    INP, FIX, LEARN = '#e8edf7', '#dfe9df', '#f4ecdc'
    rows = [('A  Frozen features (to 17 Nov)', [('Fundus image\n392 px', INP), ('RETFound-Green\n12 blocks, all frozen', FIX),
                                                ('384 features', FIX), ('Logistic regression,\nSVM or MLP', LEARN)], '0.43', GRAY),
            ('B  Fine-tuned (since 17 Nov)', [('Fundus image\n392 px', INP), ('RETFound-Green\nblocks 1-10 frozen', FIX),
                                              ('Blocks 11-12\ntrained', LEARN), ('Linear head', LEARN)], unf, BLUE),
            ('C  From scratch (since 17 Nov)', [('Fundus image\n256 px', INP), ('5 conv blocks\n32-512 channels', LEARN),
                                                ('Global pooling', LEARN), ('Linear head', LEARN)], cnn, '#7fa7c7')]
    if ch is not None and 'lora' in ch.index:
        rows.append(('D  LoRA (since 21 Nov)', [('Fundus image\n392 px', INP), ('RETFound-Green\n12 blocks, all frozen', FIX),
                                              ('LoRA add-ons in\nall 12 blocks', LEARN), ('Linear head', LEARN)],
                     f"{ch.loc['lora', 'AUPRC 3 seeds mean']:.2f}", '#1d5f8a'))
    W, H = 13.0, 5.6 + 1.45 * (len(rows) - 3)
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    ax.text(0.2, H - 0.3, ('Three' if len(rows) == 3 else 'Four') + ' ways to use the images (Chákṣu, out-of-fold AUPRC; guessing 0.14)', fontsize=13,
            fontweight='bold', va='top', color=TEXT)
    bw, bh, gap, x0 = 1.75, 0.95, 0.38, 2.95
    for k, (title, boxes, score, col) in enumerate(rows):
        y = H - 1.55 - k * 1.45
        ax.text(0.2, y + bh / 2, title.replace('  ', '\n', 1), fontsize=11, va='center', color=TEXT, fontweight='bold', linespacing=1.4)
        for i, (t, c) in enumerate(boxes):
            x = x0 + i * (bw + gap)
            ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle='round,pad=0,rounding_size=0.08', fc=c, ec=EDGE, lw=1.1))
            ax.text(x + bw / 2, y + bh / 2, t, ha='center', va='center', fontsize=10, color=TEXT, linespacing=1.3)
            if i:
                ax.annotate('', xy=(x - 0.03, y + bh / 2), xytext=(x - gap + 0.03, y + bh / 2),
                            arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.2, mutation_scale=12))
        xs = x0 + 4 * (bw + gap) + 0.1
        ax.add_patch(Circle((xs + 0.55, y + bh / 2), 0.52, fc=col, ec='none'))
        ax.text(xs + 0.55, y + bh / 2, score, ha='center', va='center', fontsize=15, color='white', fontweight='bold')
    for dx, (c, lab) in zip([0, 1.3, 4.3], [(INP, 'input'), (FIX, 'pre-trained, not changed'), (LEARN, 'trained on Chákṣu')]):
        ax.add_patch(FancyBboxPatch((x0 + dx, 0.25), 0.4, 0.25, boxstyle='round,pad=0,rounding_size=0.04', fc=c, ec=EDGE, lw=1))
        ax.text(x0 + dx + 0.5, 0.375, lab, va='center', fontsize=10, color=MUTED)
    fig.savefig(HERE / 'approaches.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- confusion at 0.5 for several weights
def weight_confusion():
    g = pd.read_csv(A17 / 'weight_grid.csv')
    g = g[g['model'] == 'logistic regression'].groupby('weight').mean(numeric_only=True)
    ws = [6.13, 4.0, 2.0, 1.0]
    fig, axs = plt.subplots(1, len(ws), figsize=(3.0 * len(ws), 3.1))
    for ax, w in zip(axs, ws):
        r = g.loc[min(g.index, key=lambda v: abs(v - w))]
        m = 100 * np.array([[r['sens@0.5'], 1 - r['sens@0.5']], [1 - r['spec@0.5'], r['spec@0.5']]])
        ax.imshow(m, cmap='Blues', vmin=0, vmax=100)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f'{v:.0f} %', ha='center', va='center', fontsize=12, color='white' if v > 60 else 'black')
        ax.set_xticks([0, 1], ['called\nglaucoma', 'called\nhealthy']); ax.set_yticks([0, 1], ['glaucoma', 'healthy'])
        ax.set_title(f'weight {w:g}' + ('  (used so far)' if w == 6.13 else '') + f'\nF1 {r["F1@0.5"]:.2f}', fontsize=10)
    fig.suptitle('Logistic regression on Chákṣu at the default 0.5 threshold: a smaller glaucoma weight flags fewer healthy '
                 'images but misses more glaucoma', fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / 'weight_confusion.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- what "opening the last two blocks" means
def finetune_diagram():
    FROZEN, TRAIN, IO = '#ecebe6', '#f8e0d6', '#ffffff'
    FE, TE = '#8a8a84', '#c0583a'
    W, H = 12.0, 10.4
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    x, w = 0.6, 4.2
    cx = x + w / 2

    def box(y, h, t, fc, ec, bold=False, fs=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06', fc=fc, ec=ec, lw=1.0))
        ax.text(cx, y + h / 2, t, ha='center', va='center', fontsize=fs, color=TEXT, fontweight='bold' if bold else 'normal')

    def arrow(y1, y2):
        ax.annotate('', xy=(cx, y2), xytext=(cx, y1), arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.1, mutation_scale=11))

    def note(y, title, lines, col):
        ax.text(5.55, y, title, fontsize=12, fontweight='bold', color=col, va='top')
        for i, l in enumerate(lines):
            ax.text(5.55, y - 0.36 - 0.3 * i, l, fontsize=11, color=TEXT, va='top')

    def bracket(y1, y2, col):
        ax.plot([4.95, 5.1, 5.1, 4.95], [y2, y2, y1, y1], color=col, lw=1.2)

    ax.text(0.6, H - 0.25, 'Opening the last two RETFound-Green blocks', fontsize=14, fontweight='bold', va='top', color=TEXT)
    y = H - 1.25
    box(y, 0.5, 'Fundus image, 392 x 392 px (flip and shift)', IO, EDGE); arrow(y, y - 0.25)
    y -= 0.85; box(y, 0.6, 'Patch embedding: 784 patches, 384 numbers each', FROZEN, FE); top_frozen = y + 0.6; arrow(y, y - 0.25)
    y -= 0.3
    for b in range(1, 13):
        y -= 0.34
        fr = b <= 10
        box(y, 0.28, f'Block {b}  ' + ('frozen' if fr else 'trained'), FROZEN if fr else TRAIN, FE if fr else TE,
            bold=not fr, fs=10)
        if b == 10:
            bot_frozen = y
        if b == 11:
            top_train = y + 0.28
    arrow(y, y - 0.25)
    y -= 0.7; box(y, 0.45, 'Final norm  trained', TRAIN, TE, bold=True); bot_train = y; arrow(y, y - 0.25)
    y -= 0.7; box(y, 0.45, 'Average of the patch tokens: 384 features', FROZEN, FE); arrow(y, y - 0.25)
    y -= 0.75; box(y, 0.5, 'Linear head: 384 -> 1', TRAIN, TE, bold=True); head = y; arrow(y, y - 0.25)
    y -= 0.75; box(y, 0.5, 'Glaucoma score', IO, EDGE); out = y
    bracket(bot_frozen, top_frozen, FE)
    note((top_frozen + bot_frozen) / 2 + 0.6, 'Frozen (about 18 M weights)',
         ['Patch embedding and blocks 1-10', 'Locked: no gradient, never changed', 'Keep the general retinal features (vessels, disc, texture)',
          'learned in pre-training on 75,000 public images'], '#5f5e5a')
    bracket(bot_train, top_train, TE)
    note(top_train + 0.25, 'Trained on Chákṣu (3.5 M weights)',
         ['Blocks 11-12 and the final norm', 'Adapt the top features to glaucoma', 'Learning rate 0.00014 (0.3 x the head rate)'], TE)
    bracket(head, head + 0.5, TE)
    note(head + 0.55, 'Trained: linear head (385 weights)', ['Learning rate 0.00047, dropout 0.37'], TE)
    note(out + 0.1, 'Training', ['Weighted cross-entropy, glaucoma x 1.8', 'AdamW, batch 32, 6-8 epochs per fold',
                                 'Optuna chose 2 open blocks of 0-3 (26 trials)'], TEXT)
    fig.savefig(HERE / 'finetune_diagram.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- project timeline
def timeline():
    cards = [('16-20 Oct', 'PAPILA, frozen features', 'Trained on PAPILA, tested on\nChákṣu: AUROC 0.71-0.73', GRAY),
             ('27 Oct', 'Chákṣu becomes main dataset', 'Test set held out; classifiers\ncompared; learning curves', GRAY),
             ('3 Nov', 'Fixed split, MLP study', 'MLP overfits (best epoch 14);\none validation part is noisy', GRAY),
             ('8 Nov', '5-fold CV, test used once', 'LR, SVM, MLP tie at AUROC 0.81;\ntest AUROC 0.872', BLUE),
             ('10 Nov', 'AUPRC, small MLP, losses', 'AUPRC 0.43 (guessing 0.14);\nfellow eyes change AUROC 0.004', GRAY),
             ('12 Nov', 'Second dataset: FairVLMed', '8,000 patients: AUROC 0.79;\nAsian 0.82, Black 0.77', GRAY),
             ('17-21 Nov', 'CNN vs fine-tuning', 'Fine-tuned 2 blocks: AUPRC 0.52;\nCNN from scratch below frozen', RED),
             ('Next', 'Planned', 'Fellow-eye check, fine-tune on\nFairVLMed, FairVLMed → Chákṣu', 'white')]
    W, H = 14.0, 5.4
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    ax.text(0.3, H - 0.25, 'Project timeline', fontsize=14, fontweight='bold', va='top', color=TEXT)
    cw, ch, gap = 3.15, 1.75, 0.27
    for i, (date, title, res, col) in enumerate(cards):
        r, c = divmod(i, 4)
        x = 0.3 + c * (cw + gap); y = H - 1.0 - ch - r * (ch + 0.55)
        ec = EDGE if col != 'white' else '#999999'
        ax.add_patch(FancyBboxPatch((x, y), cw, ch, boxstyle='round,pad=0,rounding_size=0.08', fc='white', ec=ec, lw=1.0,
                                    ls='-' if col != 'white' else (0, (4, 3))))
        if col != 'white':
            ax.add_patch(FancyBboxPatch((x, y + ch - 0.42), cw, 0.42, boxstyle='round,pad=0,rounding_size=0.08', fc=col, ec=ec, lw=1.0))
        ax.text(x + 0.15, y + ch - 0.21, date, fontsize=11, fontweight='bold', va='center', color='white' if col != 'white' else MUTED)
        ax.text(x + 0.15, y + ch - 0.68, title, fontsize=10.5, fontweight='bold', va='center', color=TEXT)
        ax.text(x + 0.15, y + 0.5, res, fontsize=9.5, va='center', color=TEXT, linespacing=1.35)
        if c < 3:
            ax.annotate('', xy=(x + cw + gap - 0.03, y + ch / 2), xytext=(x + cw + 0.03, y + ch / 2),
                        arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.1, mutation_scale=11))
    fig.savefig(HERE / 'timeline.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- cross-validation scheme
def cv_scheme():
    TR, VA, TE = '#7fa7c7', '#e8a07f', '#dddddd'
    W, H = 13.0, 4.6
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    ax.text(0.3, H - 0.25, 'How every model is scored on Chákṣu (the same folds for every model)', fontsize=13,
            fontweight='bold', va='top', color=TEXT)
    x0, wtv, wte, h = 1.2, 5.6, 1.0, 0.42
    for k in range(5):
        y = H - 1.25 - k * 0.56
        ax.text(x0 - 0.15, y + h / 2, f'Fold {k + 1}', ha='right', va='center', fontsize=10.5, color=TEXT)
        for j in range(5):
            ax.add_patch(FancyBboxPatch((x0 + j * wtv / 5 + 0.03, y), wtv / 5 - 0.06, h, boxstyle='round,pad=0,rounding_size=0.05',
                                        fc=VA if j == k else TR, ec='white', lw=0))
        ax.add_patch(FancyBboxPatch((x0 + wtv + 0.15, y), wte, h, boxstyle='round,pad=0,rounding_size=0.05', fc=TE, ec='white', lw=0))
    ax.text(x0 + wtv / 2, H - 1.25 - 4 * 0.56 - 0.35, '1,141 training + validation images (160 glaucoma)', ha='center',
            va='center', fontsize=10, color=MUTED)
    ax.text(x0 + wtv + 0.15 + wte / 2, H - 1.25 - 4 * 0.56 - 0.35, 'test: 204', ha='center', va='center', fontsize=10, color=MUTED)
    for i, (c, lab) in enumerate([(TR, 'train on these'), (VA, 'score these'), (TE, 'held out, used once')]):
        ax.add_patch(FancyBboxPatch((x0 + i * 2.1, 0.3), 0.35, 0.25, boxstyle='round,pad=0,rounding_size=0.04', fc=c, ec='none'))
        ax.text(x0 + i * 2.1 + 0.45, 0.425, lab, va='center', fontsize=10, color=MUTED)
    tx = 8.6
    notes = [('Optuna', 'Each trial is trained in all 5 folds;\nits score is the mean of the 5 fold scores'),
             ('Out-of-fold score', 'Every image is scored once, by a model\nthat never saw it during training'),
             ('Fair score (image models)', 'Fold k trains for the median best epoch\nof the other 4 folds: it chooses nothing')]
    for i, (t, d) in enumerate(notes):
        y = H - 1.0 - i * 1.15
        ax.text(tx, y, t, fontsize=11, fontweight='bold', va='top', color=TEXT)
        ax.text(tx, y - 0.32, d, fontsize=10, va='top', color=TEXT, linespacing=1.35)
    fig.savefig(HERE / 'cv_scheme.png', dpi=200, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- every number of the results table
def results_table_figure():
    import ast
    cp = pd.read_csv(A17 / 'confusion_percent.csv')
    def frozen_flag(ds, clf):
        r = cp[(cp['dataset'] == ds) & (cp['classifier'] == clf) & (cp['threshold rule'] == '90% sensitivity')]
        return float(r['healthy flagged %'].iloc[0]) if len(r) else np.nan
    names = {'cnn': 'CNN from scratch', 'unfreeze': '2 open blocks', 'lora': 'LoRA, all 12 blocks'}
    cols = {'cnn': '#7fa7c7', 'unfreeze': BLUE, 'lora': '#1d5f8a'}
    rows = []                                    # (group, label, dict)
    for tag, group, ds in [('chaksu', 'Chákṣu', 'Chakshu'), ('chaksu_pseudo', 'Chákṣu, fellow-eye pairs together', None),
                           ('fairvlmed', 'FairVLMed', 'FairVLMed')]:
        d = final(tag)
        if d is None:
            continue
        r0 = d.iloc[0]
        fname = {'svm': 'SVM', 'mlp_small': 'small MLP', 'logreg': 'logistic regr.'}.get(r0['best frozen-feature model'], r0['best frozen-feature model'])
        rows.append((group, f'Frozen features ({fname})', dict(ap=r0['frozen AUPRC'], rng=None, auc=r0['frozen AUROC'], auc_ci=None,
                    flag=frozen_flag(ds, r0['best frozen-feature model']) if ds else np.nan, diff=0.0, dci=None, col=GRAY, ref=True)))
        for m in ['cnn', 'unfreeze', 'lora']:
            if m in d.index:
                r = d.loc[m]
                rows.append((group, names[m], dict(ap=r['AUPRC 3 seeds mean'], rng=ast.literal_eval(r['AUPRC 3 seeds range']),
                            auc=r['AUROC seed 0'], auc_ci=ast.literal_eval(r['AUROC CI']), flag=r['healthy flagged % at 90 % sens'],
                            diff=r['this minus frozen AUPRC'], dci=ast.literal_eval(r['diff CI']), col=cols[m], ref=False)))
    # y positions with a gap between groups
    ys, y, last = [], 0, None
    for g, _, _ in rows:
        if last is not None and g != last:
            y += 0.8
        ys.append(y); y += 1; last = g
    ys = [max(ys) - v for v in ys]
    fig, axs = plt.subplots(1, 4, figsize=(16, 0.42 * len(rows) + 1.8), sharey=True,
                            gridspec_kw={'width_ratios': [1.1, 1, 0.9, 1.1]})
    for yv, (g, lab, r) in zip(ys, rows):
        a = axs[0]
        a.barh(yv, r['ap'], color=r['col'], height=0.62)
        if r['rng']:
            a.plot(r['rng'], [yv, yv], c='k', lw=1.2)
        a.text(r['ap'] + 0.015 if not r['rng'] else max(r['rng']) + 0.015, yv, f"{r['ap']:.2f}", va='center', fontsize=8.5)
        a = axs[1]
        a.plot(r['auc'], yv, 'o', color=r['col'] if not r['ref'] else '#666666', ms=7)
        if r['auc_ci']:
            a.plot(r['auc_ci'], [yv, yv], c=r['col'], lw=2)
        a.text(0.955, yv, f"{r['auc']:.2f}", va='center', fontsize=8.5)
        a = axs[2]
        if not np.isnan(r['flag']):
            a.barh(yv, r['flag'], color=r['col'], height=0.62)
            a.text(r['flag'] + 1.5, yv, f"{r['flag']:.0f} %", va='center', fontsize=8.5)
        else:
            a.text(2, yv, 'not computed', va='center', fontsize=8, color=MUTED)
        a = axs[3]
        if r['ref']:
            a.plot(0, yv, 'D', color='#666666', ms=5)
            a.text(0.012, yv, 'reference', va='center', fontsize=8, color=MUTED)
        else:
            lo, hi = r['dci']
            c = 'tab:green' if lo > 0 else 'tab:red' if hi < 0 else '#444444'
            a.plot([lo, hi], [yv, yv], c=c, lw=2); a.plot(r['diff'], yv, 'o', color=c, ms=6)
            a.text(hi + 0.01, yv, f"{r['diff']:+.2f}  " + ('better' if lo > 0 else 'worse' if hi < 0 else 'tied'), va='center', fontsize=8.5, color=c)
    axs[0].set_yticks(ys, [lab for _, lab, _ in rows])
    gy = {}
    for yv, (g, _, _) in zip(ys, rows):
        gy.setdefault(g, []).append(yv)
    for g, v in gy.items():
        axs[0].text(-0.62, max(v) + 0.55, g, transform=axs[0].get_yaxis_transform(), fontsize=10, fontweight='bold', color=TEXT)
    axs[0].set_xlim(0, 0.95); axs[0].set_title('AUPRC (bar: mean of 3 seeds; line: seed range)', fontsize=10)
    axs[1].set_xlim(0.6, 1.0); axs[1].set_title('AUROC, seed 0 (line: 95 % CI)', fontsize=10)
    axs[2].set_xlim(0, 100); axs[2].set_title('Healthy flagged at 90 % sensitivity', fontsize=10)
    axs[3].axvline(0, c=GRAY, ls='--', lw=1); axs[3].set_xlim(-0.25, 0.35)
    axs[3].set_title('AUPRC minus best frozen model (seed 0, 95 % CI)', fontsize=10)
    fig.suptitle('Every result of the fair comparison (out-of-fold; AUPRC of guessing: Chákṣu 0.14, FairVLMed 0.50)', fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.07, 0, 1, 0.97))
    fig.savefig(HERE / 'results_table.png', dpi=200, facecolor='white')
    plt.close(fig)


for f in [results_table_figure, timeline, cv_scheme, finetune_diagram, weight_confusion, approaches, block_diagram, class_weight, searches, datasets, ladder]:
    f(); print('made', f.__name__)
