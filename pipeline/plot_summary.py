"""One summary figure: every model's AUROC with its 95% CI, on PAPILA, on Chakshu, and on Chakshu by camera.

Usage:
    python plot_summary.py RESULTS_DIR OUT_PNG
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID, GUESS = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
ACCENT, MUTED, BASE = '#2a78d6', '#8a8984', '#eb6834'  # main model, other models, non-image baseline


def load(results, run):
    return json.loads((Path(results) / run / 'metrics.json').read_text())


def auc(m):
    return m['auroc'], m['ci95']['auroc'][0], m['ci95']['auroc'][1]


def main(results, out):
    paper, tuned = load(results, 'frozen_lr_paper_resize'), load(results, 'frozen_lr_tuned_resize')
    head, ft = load(results, 'optuna_head_resize'), load(results, 'optuna_finetune_resize')
    fov = load(results, 'frozen_lr_tuned_fov_square')
    papila = [('Paper protocol', auc(paper['papila']['image']), MUTED),
              ('Tuned logistic regression', auc(tuned['papila']['image']), ACCENT),
              ('Optuna head', auc(head['papila']['Optuna head (frozen)']), MUTED),
              ('Optuna fine-tuning', auc(ft['papila']), MUTED),
              ('Crop + pad input', auc(fov['papila']['image']), MUTED),
              ('Age + sex only', auc(tuned['papila']['clinical (age, sex)']), BASE)]
    chaksu = [('Paper protocol', auc(paper['chaksu']['All cameras']), MUTED),
              ('Tuned logistic regression', auc(tuned['chaksu']['All cameras']), ACCENT),
              ('Optuna head', auc(head['chaksu']['All cameras']), MUTED),
              ('Optuna fine-tuning', auc(ft['chaksu']['all']), MUTED),
              ('Crop + pad input', auc(fov['chaksu']['All cameras']), MUTED)]
    cams = [(f"{c} ({tuned['chaksu'][c]['n']:,} images)", auc(tuned['chaksu'][c]), ACCENT)
            for c in ['Remidio', 'Bosch', 'Forus']]
    agree = [(f"Experts {a}", auc(tuned['chaksu_by_agreement'][f'All cameras, experts {a}']), ACCENT)
             for a in ['5-0', '4-1', '3-2']]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), facecolor=SURFACE,
                             gridspec_kw={'height_ratios': [6, 4], 'hspace': 0.55, 'wspace': 0.75})
    panels = [(axes[0, 0], papila, 'PAPILA, 5 patient-level folds'),
              (axes[0, 1], chaksu, 'Chákṣu, trained on PAPILA only'),
              (axes[1, 0], cams, 'Chákṣu by camera'),
              (axes[1, 1], agree, 'Chákṣu by expert agreement')]
    for ax, rows, title in panels:
        ax.set_facecolor(SURFACE)
        n = len(rows)
        for i, (label, (v, lo, hi), color) in enumerate(rows):
            y = n - 1 - i
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle='round')
            ax.plot(v, y, 'o', ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
            ax.text(hi + 0.012, y, f'{v:.2f} ({lo:.2f}–{hi:.2f})', va='center', ha='left', fontsize=8, color=INK)
        ax.axvline(0.5, ls='--', lw=1.2, color=GUESS)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5, color=INK)
        ax.set_xlim(0.4, 1.1)
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel('AUROC (95% CI)', fontsize=8.5, color=INK2)
        ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='bold')
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.spines['bottom'].set_bounds(0.4, 1.0)
        ax.tick_params(axis='x', colors=INK2, labelsize=8)
        ax.tick_params(axis='y', length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0, 0].text(0.505, -0.55, 'guessing', fontsize=7.5, color=GUESS, va='bottom')
    fig.suptitle('Frozen RETFound-Green for glaucoma: internal (PAPILA) and external (Chákṣu) AUROC',
                 x=0.02, ha='left', fontsize=11.5, color=INK, fontweight='bold')
    fig.text(0.02, 0.01, 'Blue: main model (frozen features + tuned logistic regression). Grey: alternatives. '
             'Orange: non-image baseline. Dashed: random guessing. 95% CIs: 2000× bootstrap by patient (PAPILA) '
             'or eye (Chákṣu).', fontsize=7.2, color=INK2, wrap=True)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(out, dpi=220, facecolor=SURFACE)
    print(out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
