# Harvard-FairVLMed: the larger dataset (from 11 Nov 2026)

Status: IN PROGRESS (updated after every step).

Rahele (10 Nov) suggested a larger dataset on Hamilton next to Chakshu; on 11 Nov she named it:
https://huggingface.co/datasets/harvardairobotics/FairVLMed (FairCLIP, CVPR 2024; Harvard Ophthalmology AI Lab).

## The dataset

- 10,000 patients, one SLO (scanning laser ophthalmoscopy) fundus image each, 664 x 512 pixels, single channel.
  Not colour fundus photographs like PAPILA and Chakshu, and RETFound-Green was trained on colour photographs:
  expect a domain gap.
- Glaucoma 5,048 / 10,000 (balanced, unlike Chakshu's 14 %). The dataset's own split: training 7,000 (3,535 glaucoma),
  validation 1,000 (490), test 2,000 (1,023).
- Per patient also: age, gender (5,631 F / 4,369 M), race (819 Asian, 1,491 Black, 7,690 White), ethnicity, language,
  marital status. The npz files hold no visual-field values (the dataset card lists md / tds, but they are not in the
  files), so nothing that leaks the label is used; only the image goes into the model.
- One image per patient: no fellow-eye problem.
- Licence CC BY-NC-ND 4.0 (non-commercial, no derivatives): images, the label table, features and per-image scores
  stay on Hamilton; only code and summary results are on GitHub (`.gitignore`).
- Download (11 Nov, approved by Alison): `Dataset/dataset.zip`, 2,668,667,218 bytes, sha256 01bbfba4...7153, to
  `/nobackup/tggq61/data/raw/FairVLMed/`; unzipped to `/nobackup/tggq61/data/FairVLMed/{Training,Validation,Test}/`
  (data_#####.npz + slo_fundus_#####.jpg per patient).
- Images are dark: over 200 training images the median 99th-percentile grey level is 64 of 255 (maximum 12-255).

## Plan (same method as Chakshu, `../after_10_Nov`)

1. `make_labels.py` -> `fairvlmed_labels.csv` (Hamilton only).
2. `extract_features.py` (GPU, `run_extract.slurm`): frozen RETFound-Green, 384 features per image, two inputs:
   `raw` (grey copied to 3 channels, official resize to 392 and mean / std 0.5) and `stretch` (per-image contrast
   stretch, 1st-99th percentile to 0-255, then the same).
3. `choose_input.py`: logistic regression by 5-fold cross-validation on training + validation (8,000) for both inputs;
   the better one is used from then on.
4. `step3_auprc.py` (copy of the Chakshu one): logistic regression, SVM, kNN and the small MLP (2 layers of 32-128,
   300 trials on 16 workers) tuned by 5-fold mean AUPRC (guessing level here 0.5).
5. `step5_compare.py`: out-of-fold comparison, confusion matrices, and AUROC per race and per gender.
6. The 2,000 test patients are not used until the model is chosen (and then only once, when Alison decides).

## Log

- 11 Nov: download, unzip, label table (job 19429685 started with the label table and feature extraction).
- Job 19429685 (one H200 slice): 10,000 images per input in 30 s; features (10,000 x 384) for `raw` and `stretch` in
  `/nobackup/tggq61/retfound_green/features/fairvlmed_rfg_{raw,stretch}.npz`.
- `choose_input.py` (8,000 training + validation patients, logistic regression, C chosen by 5-fold CV):
  raw: C 0.0117, 5-fold mean AUPRC 0.796 (folds 0.785 / 0.794 / 0.805 / 0.794 / 0.801), AUROC 0.786;
  stretch: C 0.0189, AUPRC 0.795 (0.773 / 0.789 / 0.810 / 0.799 / 0.803), AUROC 0.786.
  A tie, so the simpler official input (`raw`) is used from here on (`PREPROC=raw`).
  First result: frozen RETFound-Green features on SLO images reach AUROC about 0.79 with logistic regression,
  close to Chakshu (0.81) even though RETFound-Green never saw SLO images. AUPRC is high (0.80) because glaucoma is
  50 % here (guessing = 0.5), so AUPRC is not comparable with Chakshu's 0.41 (guessing 0.14).
- Jobs 19429774 (logistic regression, kNN, SVM searches), 19429775 (small MLP, 16 workers, journal
  `optuna_mlp_small_raw.log`), 19429776 (comparison, starts after both).
