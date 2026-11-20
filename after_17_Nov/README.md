# After the 17 Nov meeting: CNN baseline, class weight, percentage confusion matrices, unfreezing

Status: IN PROGRESS (started 20 Nov; updated after every step). Test sets are not used anywhere in this folder.

## Tasks from the 17 Nov meeting and where they are done

| Task | Script | State |
| --- | --- | --- |
| Confusion matrices as percentages (Chakshu and FairVLMed); Chakshu split by camera | `confusion_percent.py` | done |
| Class weight: try smaller weights; let Optuna choose it with F1 or AUPRC | `weight_search.py grid` (job 19545173), `weight_search.py optuna` (array 19545174, F1 objective) | running |
| Own small CNN from scratch as the baseline (Optuna: blocks, channels, kernel, ...), both datasets | `cache_images.py`, `img_common.py`, `cnn_cv.py` (arrays 19545210 Chakshu, 19545211 FairVLMed) | queued |
| Repeat on FairVLMed (no class weight: 50 % glaucoma) | percentages done; CNN queued | partly |
| If time: open 0-3 RETFound-Green blocks (Optuna) | `unfreeze_cv.py` (array 19545212, 3 GPU workers) | queued |

## Result 1: confusion matrices in percent (`confusion_percent.csv`, `figures/fig1_confusion_percent_*.png`)

Rows normalised by the true class: of all glaucoma images, % found / missed; of all healthy images, % flagged / cleared.
Out-of-fold scores (Chakshu 1,141 images, FairVLMed 8,000 patients).

| Dataset | Classifier | 90 % sensitivity: healthy flagged | Youden's J: glaucoma found | Youden's J: healthy flagged |
| --- | --- | --- | --- | --- |
| Chakshu | Logistic regression | 48 % | 74 % | 24 % |
| Chakshu | SVM | 49 % | 75 % | 24 % |
| Chakshu | kNN | 55 % | 92 % | 55 % |
| Chakshu | Small MLP | 45 % | 80 % | 30 % |
| FairVLMed | Logistic regression | 61 % | 64 % | 20 % |
| FairVLMed | SVM | 59 % | 72 % | 28 % |
| FairVLMed | kNN | 71 % | 67 % | 33 % |
| FairVLMed | Small MLP | 60 % | 65 % | 22 % |

At the 90 % sensitivity threshold every model finds 90 % of the glaucoma images by construction; the difference is in
the healthy row: about half of the healthy images are flagged on Chakshu and about 60 % on FairVLMed.

## Result 2: Chakshu by camera (`confusion_by_camera.csv`, `figures/fig2_confusion_by_camera.png`)

Logistic regression (the chosen model), each classifier's own out-of-fold thresholds:

| Camera | Images (glaucoma) | 90 % sensitivity: glaucoma found | healthy flagged | Youden's J: glaucoma found | healthy flagged |
| --- | --- | --- | --- | --- | --- |
| Remidio | 910 (127) | 91 % | 47 % | 76 % | 23 % |
| Bosch | 125 (18) | 89 % | 55 % | 72 % | 23 % |
| Forus | 106 (15) | 80 % | 48 % | 60 % | 30 % |

- Rahele asked whether the healthy images called glaucoma come mainly from one camera: **no**. The share of healthy
  images flagged is similar on all three cameras (47-55 %), so the low specificity is not a Remidio effect.
- Forus misses the most glaucoma (80 % found at the 90 % threshold, 60 % at Youden), but it has only 15 glaucoma images.
- The same holds for the SVM and the small MLP (kNN flags 81 % of healthy Bosch images).

## Jobs (20 Nov)

- 19545173 (CPU): class-weight grid. 19545174 (8 CPU workers): small-MLP search with the F1 objective, 200 trials.
- 19545209 (CPU): image cache (Chakshu 256 and 392 px, FairVLMed 256 px, in `/nobackup/tggq61/retfound_green/cache`).
- 19545210 / 19545211 (2 GPU workers each): CNN searches, 60 trials (Chakshu) and 30 (FairVLMed).
- 19545212 (3 GPU workers): unfreezing search, 24 trials.
