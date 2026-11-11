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
