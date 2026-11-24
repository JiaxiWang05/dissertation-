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

## Ground-truth disc masks (what the mask head learns from)
- Source: the masks shipped with Ch&aacute;k&#7779;u, `Chaksu/{Train|Test}/5.0_OD_OC_Mean_Median_Majority_STAPLE/<camera>/Disc/Majority/<stem>.tif`
  = the dataset's majority fusion of the experts' optic-disc segmentations (the Mean, Median and STAPLE versions are not
  used). Ch&aacute;k&#7779;u has disc masks only, no cup masks. No mask is generated here: nothing is derived from a box or from
  pixel contrast.
- Reading (`cache_extra.py`): PIL, 8-bit greyscale, values 0 / 255; each mask must have the exact pixel size of its photo
  (all 1,345 do); divided by 255, bilinear resize to 392 x 392 (the same plain resize as the photos), stored as 0-255 in
  `chaksu_mask392.npy` in the same row order as the cached photos. 0 empty masks; disc = 1.8 % of an image on average.
- Training (`grounding_cv.py`): same flip / shift (or rotation) as the photo, then 14 x 14 average pooling -> 28 x 28 grid
  matching the ViT patches; each value = share of that patch inside the disc (soft target). Loss = lambda x (BCE + soft
  Dice). Validation Dice counts a patch as disc when >= 50 % of it is inside.
- Checks (`mask_check.py`, 24 Nov): 215 of 1,345 masks have more than one connected region (Remidio 191, Bosch 21,
  Forus 3); the extra regions are tiny (median 0.002 %, max 0.8 % of the disc pixels) and are kept as they are. No photo
  has an EXIF rotation, so masks and cached photos are aligned.
- The expert disc BOX (disc-crop input only, `pipeline/disc_boxes.py`) comes from the same masks: threshold > 127, largest
  connected region, bounding box; centre = box centre, diameter = mean of box width and height; crop side = 3 diameters.

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

## Result 4: where the evidence is (`evidence_metrics.csv`, `figures/evidence_maps.png`, `evidence_figure.py`)
Exact decomposition, no gradients: logit = bias + sum of per-patch contributions c (RETFound-Green models: c = w . t / 784
on the mean-pooled patch tokens; CNN: class activation map on 8 x 8 cells). Signed maps, one symmetric colour scale per
model. Fold 1, 229 validation images (32 glaucoma).

| Model | Decision weight in disc, glaucoma / healthy (disc area 2.1 / 1.8 %) | Disc contrast (s.d.) | Positive patches | Positive / negative sum |
| --- | --- | --- | --- | --- |
| Frozen + logistic regression | 2.5 / 1.7 % | +0.45 / +0.29 | 72 % | +2.34 / -0.51 |
| 2 open blocks | 2.5 / 2.2 % | -0.14 / -1.12 | 45 % | +0.40 / -0.85 |
| LoRA | 2.0 / 1.5 % | +0.03 / +0.80 | 58 % | +0.65 / -0.61 |
| LoRA + disc-mask head | 1.7 / 0.6 % | +2.71 / +5.29 | 4 % | +0.01 / -1.54 |
| CNN (8 x 8) | 3.0 / 1.6 % | +2.02 / +0.99 | 17 % | +1.20 / -4.21 |

The mask head makes the disc stand out (2.7 s.d.) but the decision weight in the disc stays near its area: the score
comes from the whole image. Correction: the first version (`attention_maps.py` figure and csv) reported the share of
POSITIVE evidence inside the disc (30.8 % for the mask head); with the mask head almost no patch is positive, so that
share overstated the effect. The frozen model's evidence lies on the rim of the field of view.
