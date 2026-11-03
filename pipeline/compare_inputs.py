"""Paired AUROC comparisons with the tuned frozen logistic regression on the squashed full image:
other inputs (same head), and the Optuna-chosen models (head on frozen features; fine-tuning).

PAPILA: out-of-fold scores ranked within each fold. Chakshu: scores of the model trained on all PAPILA, overall and
per camera. 95% CIs: paired cluster bootstrap (patients / eyes), 2000 resamples. If the interval contains 0, there
is no clear winner. AUROC does not depend on the threshold rule, so the *_sens90 runs can be used.

Usage:
    python compare_inputs.py --results /nobackup/tggq61/retfound_green/results --chaksu-labels chaksu_labels.csv \
        --out /nobackup/tggq61/retfound_green/results/input_comparison
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import rfg

BASE = 'frozen_lr_tuned_resize'
INPUTS = {'optic-disc crop': ('frozen_lr_tuned_disc_crop_sens90', 'p_image'),
          'retina crop + pad': ('frozen_lr_tuned_fov_square', 'p_image')}
MODELS = {'Optuna head (frozen)': ('optuna_head_resize', 'p_optuna_head'),
          'Optuna fine-tuning': ('optuna_finetune_resize', 'p_finetune')}
CAMERAS = ['Remidio', 'Bosch', 'Forus']


def load(folder):
    return (pd.read_csv(folder / 'papila_oof_predictions.csv').set_index('image_id'),
            pd.read_csv(folder / 'chaksu_predictions.csv').set_index('image_id'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='folder holding the run_linear_probe.py output folders')
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pb, cb = load(R / BASE)
    y, fold, g = pb['y'].to_numpy(), pb['fold'].to_numpy(), pb['patient_id'].to_numpy()
    base_rank = rfg.rank_within_fold(pb['p_image'].to_numpy(), fold)
    yc, cam, bc = cb['y'].to_numpy(), cb['camera'].to_numpy(), cb['p_glaucoma'].to_numpy()
    eyes = rfg.chaksu_external(args.chaksu_labels).set_index('image_id').loc[cb.index, 'group'].to_numpy()

    res = []
    for kind, (name, (run, col)) in [('input', i) for i in INPUTS.items()] + [('model', m) for m in MODELS.items()]:
        po, co = load(R / run)
        assert (po.loc[pb.index, 'y'].to_numpy() == y).all() and (co.loc[cb.index, 'y'].to_numpy() == yc).all()
        oc = co.loc[cb.index, 'p_glaucoma'].to_numpy()
        row = {'kind': kind, 'name': name, 'run': run, 'baseline': BASE,
               'papila': rfg.paired_bootstrap_diff(y, rfg.rank_within_fold(po.loc[pb.index, col].to_numpy(), fold),
                                                   base_rank, g),
               'chaksu': {'All cameras': rfg.paired_bootstrap_diff(yc, oc, bc, eyes)}}
        for c in CAMERAS:
            m = cam == c
            row['chaksu'][c] = rfg.paired_bootstrap_diff(yc[m], oc[m], bc[m], eyes[m])
        res.append(row)
    (out / 'comparison.json').write_text(json.dumps(res, indent=2))

    d = lambda r: f"{r['diff']:+.3f} ({r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})"  # noqa: E731
    L = [f'# Paired AUROC difference vs `{BASE}` (tuned frozen logistic regression, full image; 95% CI)\n\n'
         'PAPILA: scores ranked within each fold. Chákṣu: model trained on all PAPILA, eye-level bootstrap. '
         'If the interval contains 0 there is no clear difference.\n']
    for kind, title in [('input', 'Input (same head)'), ('model', 'Optuna-chosen model (full image)')]:
        L += [f'\n| {title} | PAPILA | Chákṣu, all | ' + ' | '.join(CAMERAS) + ' |\n',
              '|---|---|---|' + '---|' * len(CAMERAS) + '\n']
        for r in [r for r in res if r['kind'] == kind]:
            L.append(f"| {r['name']} | {d(r['papila'])} | {d(r['chaksu']['All cameras'])} | "
                     + ' | '.join(d(r['chaksu'][c]) for c in CAMERAS) + ' |\n')
    (out / 'summary.md').write_text(''.join(L))
    print(''.join(L))

if __name__ == '__main__':
    main()
