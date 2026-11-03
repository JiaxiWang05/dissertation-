"""Figures for the 20 Oct follow-up.

fairness_by_camera.png  Chakshu per camera: AUROC, and sensitivity / specificity at the PAPILA Youden's J threshold
paired_differences.png  AUROC difference (paired, 95% CI) of each change: input, head layers, unfrozen blocks

Usage:
    python plot_followup.py --results R --out R/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

SURFACE, INK, INK2, GRID, GUESS = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
CAMERA_COLORS = {'Remidio': '#2a78d6', 'Bosch': '#eb6834', 'Forus': '#1baf7a'}  # as plot_curves.py
DATA_COLORS = {'PAPILA': '#2a78d6', 'Chákṣu': '#eb6834'}  # as learning_curve.py


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='bold')
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fairness(R, out):
    groups = {r['group']: r for r in json.loads((R / 'fairness_frozen_lr_tuned_resize' / 'fairness.json').read_text())[0]['by_group']}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), facecolor=SURFACE)
    for ax, (k, title, lo) in zip(axes, [('auroc', 'AUROC', 0.5), ('sensitivity', 'Sensitivity', 0.0),
                                         ('specificity', 'Specificity', 0.0)]):
        style(ax, title)
        for i, cam in enumerate(CAMERA_COLORS):
            g = groups[cam]
            ax.plot([i, i], g['ci95'][k], color=CAMERA_COLORS[cam], lw=2, solid_capstyle='round')
            ax.plot(i, g[k], 'o', ms=8, color=CAMERA_COLORS[cam], mec=SURFACE, mew=2, zorder=3)
            ax.annotate(f'{g[k]:.2f}', (i, g[k]), xytext=(9, 0), textcoords='offset points', va='center', fontsize=8,
                        color=INK)
        ax.set_xticks(range(3), [f"{c}\n{groups[c]['n_glaucoma']} / {groups[c]['n']}" for c in CAMERA_COLORS],
                      fontsize=8, color=INK)
        ax.set_xlim(-0.5, 2.6)
        ax.set_ylim(lo - 0.03 if lo else -0.03, 1.0)
        ax.grid(axis='y', color=GRID, lw=0.8)
    axes[0].axhline(0.5, ls='--', lw=1.2, color=GUESS)
    fig.suptitle('Chákṣu by camera: same ranking, different operating point', x=0.01, ha='left', fontsize=11,
                 color=INK, fontweight='bold')
    fig.text(0.01, 0.01, "Model trained on PAPILA only (full image, tuned logistic regression). Sensitivity and "
             "specificity at the Youden's J threshold chosen on PAPILA.\nUnder each camera: glaucoma / images. "
             'Lines: 95% CI, bootstrap over eyes.', fontsize=7, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out / 'fairness_by_camera.png', dpi=220, facecolor=SURFACE)


def paired_rows(R):
    """(group, label, {'PAPILA': diff, 'Chákṣu': diff}) for every change, in display order."""
    rows = []
    cmp_ = json.loads((R / 'input_comparison' / 'comparison.json').read_text())
    for r in [r for r in cmp_ if r['kind'] == 'input']:
        rows.append(('Input, vs full image', r['name'], {'PAPILA': r['papila'], 'Chákṣu': r['chaksu']['All cameras']}))
    head = json.loads((R / 'ablation_head' / 'ablation.json').read_text())
    for r in head['rows']:
        if r.get('minus_reference'):
            n = r['n_layers']
            rows.append(('Head, vs logistic regression', f"{n} hidden layer{'s' if n > 1 else ''}",
                         {'PAPILA': r['minus_reference'].get('papila'), 'Chákṣu': r['minus_reference'].get('chaksu')}))
    ft = json.loads((R / 'ablation_finetune' / 'ablation.json').read_text())
    for r in ft['rows']:
        if r.get('minus_reference'):
            rows.append(('Unfrozen blocks, vs frozen', f"{r['unfrozen_blocks']} block{'s' if r['unfrozen_blocks'] > 1 else ''}",
                         {'PAPILA': r['minus_reference'].get('papila'), 'Chákṣu': r['minus_reference'].get('chaksu')}))
    for r in [r for r in cmp_ if r['kind'] == 'model' and 'fine' in r['name']]:
        rows.append(('Unfrozen blocks, vs frozen', 'Optuna choice', {'PAPILA': r['papila'], 'Chákṣu': r['chaksu']['All cameras']}))
    return rows


def paired(R, out):
    rows = paired_rows(R)
    ypos, labels, y, prev = [], [], 0, None
    for grp, label, _ in rows:
        if grp != prev:
            y += 0.6 if prev else 0
            labels.append((y, grp, True))
            y += 1
            prev = grp
        ypos.append(y)
        labels.append((y, label, False))
        y += 1
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 0.32 * y + 1.5), facecolor=SURFACE, sharey=True, sharex=True)
    for ax, data in zip(axes, DATA_COLORS):
        style(ax, data)
        for yy, (_, _, d) in zip(ypos, rows):
            r = d.get(data)
            if not r:
                continue
            ax.plot(r['ci95'], [yy, yy], color=DATA_COLORS[data], lw=2, solid_capstyle='round')
            sig = r['ci95'][0] > 0 or r['ci95'][1] < 0
            ax.plot(r['diff'], yy, 'o', ms=7, color=DATA_COLORS[data] if sig else SURFACE, mec=DATA_COLORS[data],
                    mew=2, zorder=3)
        ax.axvline(0, color=GUESS, lw=1.2, ls='--')
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_xlabel('AUROC difference (95% CI)', fontsize=8.5, color=INK2)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks([l[0] for l in labels], [l[1] for l in labels], fontsize=8)
    for t, (_, _, is_group) in zip(axes[0].get_yticklabels(), labels):
        t.set_fontweight('bold' if is_group else 'normal')
        t.set_color(INK if is_group else INK2)
    axes[0].set_ylim(y - 0.4, -0.6)
    fig.suptitle('What each change does to AUROC (paired comparison)', x=0.01, ha='left', fontsize=11, color=INK,
                 fontweight='bold')
    fig.text(0.01, 0.01, 'Filled dot: the 95% CI excludes 0. PAPILA: out-of-fold, scores ranked within each fold. '
             'Chákṣu: trained on PAPILA only, bootstrap over eyes.\nHead rows compare with the logistic regression '
             'chosen by Optuna; other rows with the tuned frozen logistic regression on the full image.',
             fontsize=7, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out / 'paired_differences.png', dpi=220, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fairness(R, out)
    paired(R, out)


if __name__ == '__main__':
    main()
