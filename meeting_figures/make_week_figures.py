"""Figures for the Notion page on the work since the 10 Nov meeting (numbers from ../after_10_Nov/README.md and
../fairvlmed/README.md). Run: python make_week_figures.py  ->  since_10_nov.png, two_datasets.png"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle

HERE = Path(__file__).parent
GREY, LEARN, RESULT = '#eeeeee', '#f4ecdc', '#f6e0e0'
EDGE, TEXT, MUTED = '#444444', '#1a1a1a', '#666666'
FS = 11.5


def block_diagram():
    rows = [
        ('Left or right eye from the\nimage headers (OD / OS)',
         'Read every EXIF / XMP field of\nall 1,345 files; then the optic-\ndisc position from the expert\ndisc masks',
         'Not in the headers; on Remidio\nthe disc side separates the eyes.\n93 % of consecutive files are\nfellow eyes (effect: AUROC 0.004)'),
        ('Make the MLP smaller\n(2 layers, 32-128 neurons)',
         'New Optuna search: each layer\n32, 64, 96 or 128 neurons;\n307 trials on 8 Hamilton\nworkers',
         'Best 32 and 64 neurons:\n14,497 parameters (was\n460,289); validation loss flat\nfrom epoch 92 to 300'),
        ('Use AUPRC instead of AUROC\nas the Optuna objective',
         'All four classifiers re-tuned\nby the 5-fold mean AUPRC\n(guessing level 0.14)',
         'SVM 0.427, logistic regression\n0.415, small MLP 0.414 (tied);\nkNN 0.337'),
        ('Show the confusion matrix\nof the best model',
         'Out-of-fold confusion matrices\nfor every classifier at 90 %\nsensitivity and Youden\'s J',
         'All find 144 of 160 glaucoma;\nthe small MLP flags the fewest\nhealthy images (441 of 981)'),
        ('Try other loss functions,\nweight the rare class',
         '5 losses x 5 seeds x 5 folds:\nplain, weighted 1x and 2x,\nfocal, Optuna\'s choice',
         'The class weight is essential:\nno weight AUPRC 0.32 (unstable),\nweight 2x 0.43; focal 0.31'),
        ('Run the method on a larger\ndataset (FairVLMed)',
         '10,000 patients, SLO images,\n50 % glaucoma; frozen features;\nsame searches, comparison and\nper-group results',
         'AUROC 0.79 (Chákṣu 0.81);\nSVM and small MLP tied;\nAUROC Asian 0.82, White 0.79,\nBlack 0.77'),
    ]
    todo = ['Flip all images to one side\n(Remidio only)', 'FairVLMed: widen the MLP\nsearch (best values at the edge)',
            'Next step: LoRA or unfreezing\nRETFound-Green blocks']
    W, x0, num_w, cgap = 11.0, 0.15, 0.5, 0.34
    x1 = W - 0.15
    cx0 = x0 + num_w + 0.1
    cw = (x1 - cx0 - 2 * cgap) / 3
    cols = [cx0 + i * (cw + cgap) for i in range(3)]
    rh, rgap, th = 1.18, 0.26, 0.85
    H = 0.3 + 0.45 + 0.55 + len(rows) * (rh + rgap) + 0.45 + 0.3 + th + 0.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    def box(x, y, w, h, t, fc, ec=EDGE, lw=1.2, ls='-'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=FS, color=TEXT, linespacing=1.3)

    def arrow(a, y, b):
        ax.annotate('', xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.3, mutation_scale=13))

    top = H - 0.3
    ax.text(x0, top, 'Since the 10 November meeting', fontsize=13, fontweight='bold', va='top', color=TEXT)
    hy = top - 0.75
    for cx, t in zip(cols, ['Asked on 10 November', 'What I did', 'Result']):
        ax.text(cx + cw / 2, hy, t, ha='center', va='center', fontsize=10.5, color=MUTED, fontweight='bold')
    ry = hy - 0.3 - rh
    for k, r in enumerate(rows):
        yc = ry + rh / 2
        ax.add_patch(Circle((x0 + num_w / 2, yc), 0.19, fc='#555555', ec='#555555'))
        ax.text(x0 + num_w / 2, yc, str(k + 1), ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        for i, (cx, t, c) in enumerate(zip(cols, r, [GREY, LEARN, RESULT])):
            strong = k == len(rows) - 1 and i == 2
            box(cx, ry, cw, rh, t, c, ec='#8b2e2e' if strong else EDGE, lw=2.0 if strong else 1.2)
            if i:
                arrow(cx - cgap + 0.03, yc, cx - 0.03)
        ry -= rh + rgap
    ty = ry + rh + rgap - 0.45
    ax.plot([x0, x1], [ty + 0.22, ty + 0.22], color='#cccccc', lw=1)
    ax.text(x0, ty, 'Still to do', fontsize=13, fontweight='bold', va='top', color=TEXT)
    for cx, t in zip(cols, todo):
        box(cx, ty - 0.3 - th, cw, th, t, 'white', ls=(0, (4, 3)))
    fig.savefig(HERE / 'since_10_nov.png', dpi=220, facecolor='white')
    plt.close(fig)


def two_datasets():
    names = ['Logistic\nregression', 'SVM', 'kNN', 'Small MLP']
    ch = [0.814, 0.817, 0.747, 0.803]
    fv = [0.785, 0.791, 0.723, 0.791]
    ch_ci = [(0.779, 0.848), (0.785, 0.850), (0.708, 0.785), (0.764, 0.841)]
    fv_ci = [(0.775, 0.795), (0.781, 0.801), (0.713, 0.734), (0.782, 0.801)]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4))
    for off, vals, ci, col, lab in [(-w / 2, ch, ch_ci, '#4c72b0', 'Chákṣu: 1,141 colour photographs, 14 % glaucoma'),
                                    (w / 2, fv, fv_ci, '#dd8452', 'FairVLMed: 8,000 SLO images, 50 % glaucoma')]:
        err = np.array([[v - lo for v, (lo, hi) in zip(vals, ci)], [hi - v for v, (lo, hi) in zip(vals, ci)]])
        ax.bar(x + off, vals, w, color=col, label=lab, yerr=err, capsize=4)
        for xi, v in zip(x + off, vals):
            ax.text(xi, 0.655, f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, color='white', fontweight='bold')
    ax.set_xticks(x, names); ax.set_ylim(0.65, 0.88)
    ax.set_ylabel('out-of-fold AUROC (95 % CI)')
    ax.set_title('Frozen RETFound-Green features on both datasets (5-fold cross-validation, test sets not used)', fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / 'two_datasets.png', dpi=200); plt.close(fig)


block_diagram(); two_datasets(); print('saved')
