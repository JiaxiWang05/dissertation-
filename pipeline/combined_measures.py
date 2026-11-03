"""Single-number summaries of the confusion matrix at the chosen threshold: F1, balanced accuracy, Youden's J, MCC.

Recomputed from the saved predictions and thresholds of run_linear_probe.py runs (PAPILA: out-of-fold, each fold's
own threshold; Chakshu: the threshold frozen from PAPILA), with 95% cluster-bootstrap CIs (patients / eyes).
F1 combines sensitivity and precision (it ignores true negatives); balanced accuracy = (sensitivity + specificity) / 2
and Youden's J = sensitivity + specificity - 1 combine sensitivity and specificity and do not change with prevalence;
MCC uses all four cells of the confusion matrix and changes with prevalence.

Usage:
    python combined_measures.py --results R --chaksu-labels chaksu_labels.csv --out R/combined_measures
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import rfg

RUNS = {'frozen_lr_tuned_resize': "full image, Youden's J", 'frozen_lr_tuned_resize_sens90': 'full image, 90% sensitivity',
        'frozen_lr_tuned_disc_crop_sens90': 'optic-disc crop, 90% sensitivity'}
SHOWN = ['sensitivity', 'specificity', 'f1', 'balanced_acc', 'youden_j', 'mcc']
HEAD = ['Sensitivity', 'Specificity', 'F1', 'Balanced accuracy', "Youden's J", 'MCC']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--chaksu-labels', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eyes = rfg.chaksu_external(args.chaksu_labels).set_index('image_id')['group']

    res = []
    for run, label in RUNS.items():
        pap = pd.read_csv(R / run / 'papila_oof_predictions.csv')
        cha = pd.read_csv(R / run / 'chaksu_predictions.csv')
        thr = json.loads((R / run / 'metrics.json').read_text())['papila_full_fit_threshold']
        res.append({'run': run, 'label': label, 'data': 'PAPILA (out-of-fold)',
                    **rfg.evaluate(pap['y'], pap['p_image'], pap['patient_id'], thr=pap['thr_image'])})
        g = eyes.loc[cha['image_id']].to_numpy()
        for cam in ['All cameras', 'Remidio', 'Bosch', 'Forus']:
            m = (cha['camera'] == cam).to_numpy() if cam != 'All cameras' else slice(None)
            res.append({'run': run, 'label': label, 'data': f'Chákṣu, {cam}',
                        **rfg.evaluate(cha['y'][m], cha['p_glaucoma'][m], g[m], thr=thr)})
    (out / 'combined_measures.json').write_text(json.dumps(res, indent=2, default=float))

    c = lambda r, k: f"{r[k]:.2f} ({r['ci95'][k][0]:.2f}–{r['ci95'][k][1]:.2f})"  # noqa: E731
    L = ['# Single-number summaries at the chosen threshold (95% CI)\n\n',
         'F1 combines sensitivity and precision (ignores true negatives); balanced accuracy and Youden\'s J combine '
         'sensitivity and specificity and do not change with prevalence; MCC uses all four cells and changes with '
         'prevalence. Chákṣu uses the threshold frozen from PAPILA.\n\n',
         '| Input, threshold | Data | ' + ' | '.join(HEAD) + ' |\n', '|---|---|' + '---|' * len(HEAD) + '\n']
    for r in res:
        L.append(f"| {r['label']} | {r['data']} | " + ' | '.join(c(r, k) for k in SHOWN) + ' |\n')
    (out / 'summary.md').write_text(''.join(L))
    print(''.join(L))


if __name__ == '__main__':
    main()
