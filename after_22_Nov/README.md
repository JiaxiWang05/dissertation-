# 22 Nov: ideas borrowed from EndoWAM (arXiv 2608.01221)

Ch&aacute;k&#7779;u, the same 5 folds as before (1,141 training + validation images; test untouched), fair two-pass scoring
(median best epoch of the other folds), 3 seeds; paired differences on seed 0 with 2,000 grouped resamples.
All image models are LoRA RETFound-Green with its 17 Nov Optuna setting unchanged.

| File | What |
| --- | --- |
| `cache_extra.py` | expert optic-disc masks (Ch&aacute;k&#7779;u has disc masks only, no cup) and disc crops, 392 px |
| `layer_features.py` | frozen features from every block 1-12, classifiers tuned inside each fold |
| `grounding_cv.py` | LoRA variants: rerun, stronger augmentation, disc-mask head (weight 0.25 / 1), disc-crop input; camera hold-out |
| `attention_maps.py` | exact per-patch evidence maps; share of evidence inside the expert disc |
| `summarise.py`, `make_figures.py` | statistics (`grounding_summary.csv`, `camera_holdout.csv`) and figures |

## Result 1: earlier blocks are worse (`layer_features.csv`, `figures/layers.png`)
Block 12 (the features used so far): AUPRC 0.41 (logistic regression and SVM). Block 11: 0.33 / 0.40. Blocks 1-10:
0.23-0.34, all worse (paired CI below 0). Blocks 9-12 joined: 0.37 / 0.40, not better. Check: the 10 Nov SVM setting
on block 12 gives 0.427, the stored score.

## Result 2: LoRA variants (`grounding_summary.csv`, `figures/variants.png`)
| Variant | AUPRC (3 seeds) | AUROC | Healthy flagged at 90 % sens. | AUPRC vs rerun | AUROC vs rerun |
| --- | --- | --- | --- | --- | --- |
| LoRA rerun | 0.52 (0.48-0.55) | 0.87 | 39 % | reference | reference |
| Stronger augmentation | 0.56 (0.53-0.58) | 0.88 | 30 % | +0.06 (-0.02 to 0.14), tied | +0.03 (0.002 to 0.055), better |
| Disc-mask head, weight 0.25 | 0.56 (0.55-0.58) | 0.88 | 21 % | +0.06 (-0.006 to 0.12), tied | +0.04 (0.020 to 0.063), better |
| Disc-mask head, weight 1 | 0.54 (0.52-0.55) | 0.88 | 29 % | +0.03 (-0.05 to 0.12), tied | +0.02 (-0.01 to 0.05), tied |
| Disc-crop input (expert box) | 0.58 (0.56-0.60) | 0.89 | 29 % | +0.04 (-0.007 to 0.10), tied | +0.03 (0.008 to 0.060), better |

Every variant is better than the frozen SVM (AUPRC +0.09 to +0.15, all CIs above 0). The mask head learns the disc
(validation Dice 0.93-0.94 on the 28 x 28 patch grid). The LoRA rerun (0.52) is a little below the 17 Nov run (0.54),
inside the seed spread. No variant is better in AUPRC; three are better in AUROC.

## Result 3: camera hold-out (`camera_holdout.csv`, `figures/camera_holdout.png`)
Trained on two cameras, scored on the third (image models: 7 epochs, 3 seeds). Bosch held out: frozen 0.31-0.33,
LoRA 0.69-0.73; Forus: frozen 0.35-0.43, LoRA 0.55-0.61; Remidio (only 231 training images left): frozen 0.28-0.29,
LoRA 0.33-0.34. LoRA transfers to an unseen camera far better than the frozen classifiers; augmentation and the mask
head change little (Bosch 18 and Forus 15 glaucoma images: noisy).

## Result 4: where the evidence is (`attention_maps.csv`, `figures/attention_maps.png`)
Fold 1, 229 validation images; the disc covers 1.8 % of an image. Share of positive glaucoma evidence inside the disc:
frozen + logistic regression 2.1 % (mostly on the rim of the field of view), 2 open blocks 0.3 %, LoRA 0.8 %,
LoRA + disc-mask head 30.8 % (45.5 % on glaucoma images, 17 times an even spread), CNN 5.6 % (8 x 8 grid).
