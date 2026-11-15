# Evidence pack: frozen foundation-model features for glaucoma (PAPILA to Chaksu)

Compiled 2026-10-12. This pack lists facts only.

**Conventions**
- [n] cites the reference list in Section E.
- (?) means not stated in the text read, or not verified.
- (calc) means arithmetic done by the reader of the record, not stated by the authors.
- (inf) means an inference by the reader of the record, not stated by the authors.
- Read status codes: F = full text read; S = supplementary material read; A = published abstract only; P = preprint full text; K = facts carried over from an earlier summary, not re-read for this pack.

---

## A. RETFound on PAPILA [1]

Sources:
- [1] Supplementary Tables 1, 3 and 4 (MOESM3 xlsx), read in full.
- Supplementary Figures PDF (MOESM1), read in full. It contains nothing on PAPILA or on how the public datasets were split.
- GitHub `rmaphoh/RETFound`: BENCHMARK.md and README.md.
- The file directory and readme.txt of the released `PAPILA.zip` and `Glaucoma_fundus.zip`. No images were downloaded.
- Methods of the Nature HTML: searched for protocol sentences only.

### A.1 Dataset and protocol

**PAPILA**
- Country: Spain. Device: Topcon TRC-NW400 non-mydriatic retinal camera (Supp. Table 1).

**Classes: 3, not binary**
- Supp. Table 1 names them No glaucoma / Suspected glaucoma / Glaucoma.
- The Methods describe both glaucoma datasets as non-glaucoma / early (suspected) / advanced.
- Folders in the released zip: `anormal`, `bsuspectglaucoma`, `cglaucoma`.

**Labels:** two experts labelled the images after a full clinical examination and a retrospective review of clinical records (Methods).

**Split counts (Supp. Table 1)**

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| No glaucoma | 212 | 54 | 67 | 333 (68.1%) |
| Suspected glaucoma | 57 | 14 | 17 | 88 (17.99%) |
| Glaucoma | 43 | 11 | 14 | 68 (13.9%) |
| Total | 312 | 79 | 98 | 489 |

- The ratio is 63.8 / 16.2 / 20.0% (calc).
- The released zip holds 488 images, not 489. Train/suspect has 56 images in the zip against 57 in the table; every other cell matches.

**Patient-level split**
- Not stated in Supp. Table 1, the Supp. Figures, BENCHMARK.md or the zip readme.
- A check of the released split shows it is not patient-level:
  - 244 patient IDs, each with exactly one OD and one OS image.
  - For 121 of 244 patients (49.6%) the two eyes are in different splits: 56 train/test, 55 train/val, 10 val/test.
  - 56 of the 98 test images (57%) have the patient's other eye in training.
  - Patients per split: train 211, val 72, test 82.
  - 7 patients have different labels for their two eyes.
- This assumes PAPILA files are named `RET<patient ID><eye>`. The PAPILA source paper was not read (?).

**Glaucoma Fundus (for context)**
- South Korea; Nidek AFC-330.
- 3 classes: No glaucoma / Early / Advanced. Two specialists had to agree, using visual fields and imaging.
- Split 861 / 218 / 465, 1,544 images in total; ratio 55.8 / 14.1 / 30.1% (calc). The zip matches the table.
- Patient-level grouping cannot be checked because files are numbered with no patient ID.

**Adaptation (Methods; Supp. Fig. 6 caption)**
- Fine-tuning of the ViT-L encoder plus an MLP head, with label smoothing.
- Batch 16; 50 epochs (10 warm-up epochs to a learning rate of 5e-4, then cosine decay to 1e-6).
- The checkpoint with the best validation AUROC is kept.
- Input is 224×224, after AutoMorph preprocessing and resizing to 256.
- Whether every encoder layer is updated: not stated in the text read; code not checked (?).
- No linear-probe or frozen-feature results.
- The current README offers an `ADAPTATION` option {finetune, lp}, added after the paper.

**Statistics**
- 5 random seeds that change only the shuffling of training data. All runs use the same fixed split.
- 95% CI = mean ± 1.96·SD/√5, with two-sided t-tests. The CIs therefore reflect seed variation only, not sampling of the test set.
- Multiclass AUROC and AUPR are computed per class and averaged.

**Not done in [1] for glaucoma**
- No external or cross-dataset test. Cross-dataset evaluation covers the three DR datasets only (Fig. 2b).
- No glaucoma data-efficiency curve. The Fig. 4 curves cover heart failure, MI, MESSIDOR-2 and IDRID.

**Comparison models in [1]** (same architecture and fine-tuning for all)
- SL-ImageNet: supervised pretraining on ImageNet-21k.
- SSL-ImageNet: self-supervised pretraining on ImageNet-1k.
- SSL-Retinal: self-supervised pretraining on retinal images from scratch.
- RETFound: SSL-ImageNet weights, then further self-supervised pretraining on retinal images.
- SSL-strategy variants: SwAV and SimCLR use ResNet-50; the others are Transformers. The contrastive variants start from ImageNet-1k weights and are trained on 1.6M retinal images.

### A.2 Results: mean (95% CI) on the internal hold-out test set, 3-class macro averages

**Sheet "Supplementary Table 3"** (pretraining comparison; table note: RETFound uses MAE by default)

| Model | PAPILA AUROC | PAPILA AUPR | Glaucoma Fundus AUROC | Glaucoma Fundus AUPR |
|---|---|---|---|---|
| RETFound | 0.855 (0.842, 0.868) | 0.748 (0.706, 0.791) | 0.943 (0.941, 0.945) | 0.863 (0.860, 0.867) |
| SSL-Retinal | 0.705 (0.685, 0.725) | 0.507 (0.475, 0.539) | 0.895 (0.892, 0.897) | 0.809 (0.807, 0.810) |
| SSL-ImageNet | 0.727 (0.709, 0.745) | 0.521 (0.488, 0.554) | 0.885 (0.881, 0.889) | 0.792 (0.784, 0.799) |
| SL-ImageNet | 0.819 (0.807, 0.830) | 0.660 (0.641, 0.679) | 0.919 (0.917, 0.922) | 0.813 (0.809, 0.817) |

**Sheet "Supplementary Table 4"** (SSL strategy inside the RETFound framework; all fine-tuned)

| Strategy | PAPILA AUROC | PAPILA AUPR | Glaucoma Fundus AUROC | Glaucoma Fundus AUPR |
|---|---|---|---|---|
| MAE (= RETFound) | 0.855 (0.842, 0.868) | 0.748 (0.706, 0.791) | 0.943 (0.941, 0.945) | 0.863 (0.860, 0.867) |
| DINO | 0.788 (0.772, 0.805) | 0.635 (0.620, 0.651) | 0.918 (0.915, 0.921) | 0.802 (0.795, 0.809) |
| MoCo-v3 | 0.797 (0.760, 0.834) | 0.627 (0.568, 0.687) | 0.915 (0.911, 0.918) | 0.839 (0.833, 0.845) |
| SimCLR | 0.805 (0.773, 0.837) | 0.637 (0.591, 0.683) | 0.897 (0.884, 0.911) | 0.765 (0.737, 0.793) |
| SwAV | 0.791 (0.774, 0.809) | 0.617 (0.589, 0.644) | 0.919 (0.908, 0.931) | 0.798 (0.768, 0.827) |

The DINO row is DINO inside the RETFound framework. It is not the later RETFound-DINOv2 checkpoint of [3] and [14].

### A.3 RETFound PAPILA AUROC as reported across papers

| Source | Checkpoint / adaptation | Split | PAPILA AUROC |
|---|---|---|---|
| [1] Supp. T3 | RETFound (MAE), full FT, 5 seeds | released 312/79/98 | 0.855 (0.842–0.868) |
| [15] | RETFound (variant not stated (?)), full FT, 5 seeds | "official" 80/20, then train split 4:1 | 0.855 |
| [15] | same model, linear probe | same | 0.620 |
| [3] | RETFound-MAE, FT, single run | 312/79/98 | 0.832 (0.736–0.899) |
| [3] | RETFound-MAE, LP (frozen encoder, gradient-trained FC head) | same | 0.748 (0.660–0.838) |
| [3] | RETFound-DINOv2, FT | same | 0.916 (0.852–0.970) |
| [3] | RETFound-DINOv2, LP | same | 0.854 (0.794–0.916) |
| [18] | RETFound, FT | 312/79/98 | 0.820 (0.788–0.854) |
| [19] | RETFound, FT | 55:15:30, 5 random splits | 0.820 (0.788–0.854) |
| [17] | RETFound-CFP, FT | 313/77/98 | 0.7436 |
| [18] | RETFound, 16-shot | — | 0.760 |
| [18] | RETFound, zero-shot | — | 0.514 |
| [9] | RETFound (described as ViT-B), FT on KULRD; PAPILA as binary external test, suspects excluded | external | 0.88 (0.84–0.91) |
| [11] | RETFound, FT on 400 SEED images; PAPILA as binary external test | external | 0.779 (0.731–0.826) |
| [10] | RETFound, FT on Glaucoma Fundus; PAPILA as external test, 3-class (?) | external | 0.558 |

### A.4 Points to watch when comparing with the student's study

- PAPILA results in [1], [3], [10], [15], [18] and [19] are 3-class macro AUROC (healthy / suspect / glaucoma). They are not directly comparable with a binary glaucoma-vs-healthy AUROC.
- [1] used one fixed image-level split that is not patient-level (A.1). [3] and [18] report the same 312/79/98 counts. Whether they used the identical split is not verified (?).
- [1] used full fine-tuning with 312 (PAPILA) and 861 (Glaucoma Fundus) training images. It has no linear probe, no glaucoma label-efficiency curve and no cross-camera glaucoma test.
- In [3], RETFound-MAE LP on PAPILA gives macro-F1 0.270 at full data. Always predicting the majority class (67/98) gives about 0.271 (calc), so those runs probably collapsed to "no glaucoma" (inf). Similar F1 values appear for DINOv2-S/B/L FT, DINOv3-S/S+/H+ and RETFound-MAE at 10–50% data.

---

## B. Evidence table

Each row is one paper that evaluates glaucoma, or a glaucoma-related optic-nerve target ([6]). Flags:
- [11]: the published full text was **not accessible**; read the published abstract plus the preprint full text.
- [8] and [22]: the publisher pages returned 403; read the author manuscript and arXiv v1 respectively.
- [9]: the TBME version was not read; read arXiv v1.
- [2]: facts carried over from an earlier summary (K).

| Paper (first author, year) [n], read status | Model(s) | Pre-training | Glaucoma data & classes | Adaptation | Labelled data | External / cross-camera test | Split / stats | Glaucoma result |
|---|---|---|---|---|---|---|---|---|
| [1] Zhou 2023, Nature. S read in full; main text K + Methods searched | RETFound ViT-L (MAE); SL-ImageNet, SSL-ImageNet, SSL-Retinal; DINO, MoCo-v3, SimCLR, SwAV (ResNet-50 for SimCLR and SwAV) | SSL-ImageNet init, then MAE on retinal images (904,170 CFP per [4], [11]); SSL variants on 1.6M retinal images | PAPILA (Spain, Topcon TRC-NW400; 489 images; no / suspect / glaucoma). Glaucoma Fundus (Korea, Nidek AFC-330; 1,544 images; no / early / advanced). Both 3-class | Full FT, ViT-L + MLP head, 50 epochs; all-layer update not stated (?); no LP | PAPILA train 312; Glaucoma Fundus train 861 | None for glaucoma (cross-dataset DR only) | Fixed image-level split; released PAPILA split not patient-level (57% of test images have the other eye in train); 5 seeds varying only shuffling; CI over seeds | Macro AUROC: PAPILA 0.855 (0.842–0.868); Glaucoma Fundus 0.943 (0.941–0.945). SL-ImageNet 0.819 / 0.919 |
| [2] Isztl 2025, arXiv. K | SwinV2-tiny, RETFound-DINOv2-CFP, RETFound-MAE-CFP; others (?) | not in facts provided (?) | AIROGS (113,893) + PAPILA (488) merged into 3 classes | Full FT only; one seed (42); best validation accuracy | 70% of the merged set (?) | None | Image-level stratified 70/15/15; single seed | Accuracy: SwinV2-tiny 93.19% vs RETFound-DINOv2-CFP 92.57%; RETFound-MAE-CFP not in the glaucoma top 5; no data-scarcity analysis |
| [3] Zhou 2025, arXiv. F + S | DINOv2 S/B/L/g; DINOv3 S/S+/B/L/H+; RETFound-MAE; RETFound-DINOv2 (all ViT) | DINOv2 LVD-142M; DINOv3 LVD-1689M; RETFound-DINOv2 on 904K AlzEye retinal images, initialised from DINOv2-L; RETFound-MAE on the same 904K | PAPILA 3-class (489); Glaucoma Fundus 3-class (1,544); Retina 4-class with a glaucoma class (601) | FT (encoder + FC) and LP (FC only, encoder frozen). LP is gradient-trained with augmentation (batch 24, 50 epochs); not LR on cached features | PAPILA 312/79/98; Glaucoma Fundus 861/218/465. Label efficiency at 10/30/50/100% (about 31/94/156/312 PAPILA images, calc), LP and FT, for 4 models | None for glaucoma; external tests only for DR and systemic (AlzEye to UKB), FT only | Methods: "patient groups" at 55:15:30, but PAPILA counts (≈64:16:20) match [1]; patient-level for PAPILA not stated (?). Single split, single run; test bootstrap n=200 | PAPILA LP: RETFound-DINOv2 0.854, RETFound-MAE 0.748, DINOv3-L 0.738, DINOv2-g 0.701, DINOv2-L 0.687. PAPILA FT: 0.916 / 0.832 / 0.855 / 0.834. Glaucoma Fundus LP: 0.903 / 0.882 / 0.893 / 0.897. At 10% LP on PAPILA: DINOv2-g 0.631 and DINOv3-L 0.637, above RETFound-DINOv2 0.613 and RETFound-MAE 0.590 |
| [4] Chuter 2025, Ophthalmol Sci. F + S | RETFound CFP (ViT-L per Table S3) | RETFound as published | DIGS/ADAGES (US, 3 sites); stereo ONH photos from several cameras incl. Nidek 3-DX. Binary GON (photo-graded) with or without VF damage. 9,787 images, 2,104 patients | FT for 5/10/20/50 epochs; layer decay 0.65 implies whole-network FT, not stated in words; no LP | Train+val subsets of 50/100/200/500/1000/2000 images; 10 runs each | None | Patient-level 70/10/20; 10 runs × 100 bootstraps | AUC at 50 epochs: 0.574 (50 images), 0.624 (100), 0.699 (200), 0.811 (500), 0.832 (1000), 0.859 (2000). At 2000: mild 0.837 vs moderate-severe 0.946 |
| [5] Bolo 2026, Ophthalmol Sci. F | RETFound (`RETFound_mae_meh`) vs VGG-19 | RETFound: ImageNet-1K + 904,170 CFP. VGG-19 pretraining not stated (?) | LA County DHS diabetic teleretinal screening (Topcon NW400/NW8, Canon CR-2 AF). Binary referable glaucoma (CDR ≥ 0.6); cropped and uncropped ONH | FT with RETFound defaults (100 epochs); all-layer update not stated (?); no LP | 12,998 images, 6,116 participants; train 8,996. Class-balanced subsets of 20/400/1000/2000/4000/6000 | Yes: USC Roski clinics (Zeiss FF450, CLARUS, Topcon NW8, Optos), 300 images, 203 patients, no retraining; per-camera results not reported (?) | Patient-level 80/20 development/test; train/val 75/25 not patient-level; DeLong CIs; repeats not described | Internal: cropped VGG-19 0.924 vs cropped RETFound 0.911 (p=0.07). External: uncropped RETFound 0.886 (0.849–0.924) vs VGG-19 0.767 cropped / 0.634 uncropped. RETFound led at 400 and 1000 images; the 2000-image result is inconsistent between Results and Discussion |
| [6] Chen 2025, Ophthalmol Sci. F | Frozen RETFound; ImageNet ViT-B/16; VGG16 | RETFound as published; ImageNet | Stanford; Zeiss FF450 plus macula-centred CFP paired with Cirrus OCT. Regression of CDR and RNFL thickness; no glaucoma label | Frozen features + linear regressors (OLS / ridge / lasso / elastic net), tuned by 5-fold CV. Baselines are labelled "fine-tuned" in the tables (inconsistent) | 776 pairs, 463 patients | None (single centre) | Patient-level 80:20; no CIs | R², CDR: 0.898 (RETFound + lasso) vs ViT 0.687 vs VGG16 0.499. Average RNFL: 0.961 vs 0.731 vs 0.600 |
| [7] Chuter 2026, Ophthalmol Sci. F | RETFound CFP and OCT encoders; multimodal fusion | RETFound as published | DIGS/ADAGES; 14,510 stereo CFPs + Spectralis OCT; binary glaucoma vs healthy | Full FT per Methods (the Limitations call it feature extraction: inconsistent); focal loss; early stopping | Train/val 898 participants (10,388 CFPs); test 200 participants (4,122 CFPs); no label efficiency | None | Participant-level train/val vs test; 95% CIs; seeds not stated (?) | AUC: CFP 0.86 (0.81–0.89); OCT 0.93; CFP + OCT 0.94 |
| [8] Xiong 2025, NEJM AI. F (author manuscript; publisher 403) | RETFound vs ImageNet-1k supervised ViT-L | RETFound as published; ImageNet-1k | Chinese health-examination cohort (Tongren); 1,512 positive + 1,576 negative. Label definition and camera in the supplement (?) | FT (details in the supplement (?)); no LP | 100/50/25/10% of the dataset (10% = 151 + 158) | No separate site | 5-fold CV; patient grouping not stated (?); "best" AUC reported | AUC, RETFound vs ViT-L: 0.863 vs 0.853 (100%); 0.838 vs 0.808 (10%); all p ≥ .3 |
| [9] Abramovich 2026 (GONet), IEEE TBME. F (arXiv v1) | ViT-B (no SSL), SwinV2, DINOv2, Mugs, MoCo-v3, RETFound (paper says all SSL models are ViT-B); GONet = DINOv2 ViT-B + multi-source-domain FT | Natural images (unnamed); RETFound MAE on 900K fundus images | KULRD (Belgium, Zeiss Visucam 500; 61,213 images, 8,203 patients). Binary GON, suspects excluded. External: HYRD, PAPILA (393 images), DRISHTI-GS, REFUGE, REFUGE2, GAMMA | Full FT only; 392 px input | KULRD split 85:5:10; no label efficiency | Yes: single-source models trained on KULRD only, tested on 6 external sets (Topcon Triton / TRC-NW400, Canon CR-2, KOWA, Zeiss) without retraining | Stratified by age, sex and label; patient-level not stated (?); 1,000 bootstrap draws; single run | PAPILA AUC (single-source): RETFound 0.88 (0.84–0.91), SwinV2 0.87, DINOv2 0.85, ViT-B 0.83. GONet 0.90. KULRD-Test: DINOv2 0.84, RETFound 0.83. CDR alone: PAPILA 0.83, REFUGE 0.96 |
| [10] Hou 2026, Ophthalmol Sci. F | DINOv2 ViT-S/B/L vs RETFound ViT-L | DINOv2 LVD-142M; RETFound ImageNet-1K then about 900K CFP + about 730K OCT (MAE) | PAPILA and Glaucoma Fundus, 3-class (non / early-suspected / late-stage); sizes in Table S1 (?) | FT replicating the RETFound protocol (MLP head, 100 epochs); backbone unfreezing implied, not stated; no LP | RETFound splits; label efficiency at 10/20/50/90/100% | Yes: FT on Glaucoma Fundus (Korea), tested on PAPILA (Spain) without retraining; cameras not named | Patient-level not stated (?); 100 bootstraps of 20% of the test set; Bonferroni-corrected t-tests | Glaucoma Fundus internal: DINOv2-B/S 0.958 vs RETFound 0.940 (P<0.001). External to PAPILA: DINOv2-L 0.625 vs RETFound 0.558. PAPILA internal only in Fig 2 / Table S3 (?). Glaucoma label efficiency: models similar |
| [11] Yew 2026, Lancet Digit Health. **A + P (published full text not accessible)** | RETFound (MAE; CFP or OCT checkpoint not stated (?)) vs ResNet50 (ImageNet-1k), ViT-Base, SwinV2 (ImageNet-21k) | RETFound: ImageNet-1k init, then 904,170 CFP + 736,442 OCT | Binary glaucoma (definition in Supp. T1 (?)). Internal SEED (Singapore). External ODIR-5K, PAPILA, GAMMA. Handling of PAPILA suspects not stated (?) | Full FT for all models; no LP | 100/50/20% of training data, plus fixed 400/200/100 images (half cases); full counts in Supp. T2 (?) | Yes: SEED-trained models tested on ODIR-5K, PAPILA and GAMMA without retraining; cameras not stated (?) | Split method not stated (?); 2,000-bootstrap CIs; Hanley-McNeil Z-tests, Bonferroni | RETFound at 400 images: SEED 0.908, ODIR-5K 0.835, PAPILA 0.779 (0.731–0.826), GAMMA 0.990; better than ResNet50 (PAPILA p=0.0032). Preprint, ≤400 images: RETFound 0.738–0.990 vs ResNet50 0.613–0.868; SwinV2 0.646–0.947, comparable to RETFound |
| [12] Zoellin 2026, CSBJ. F (appendix blocked) | 34 encoders: DINOv2 (S–G, with and without registers), CLIP, SigLIP/SigLIP2, RadDINO, Virchow/Virchow2, Prov-GigaPath, RetiZero, DINORET, UrFound, RET-CLIP, RETFound-MAE/DINOv2 variants, RETFound-Green, ViLReF, VisionFM, ResNet-50 (random / ImageNet) | As published per encoder | 39 tasks over 21 datasets. Glaucoma: Harvard FairVision glaucoma and PAPILA named in the main text; also Harvard Glaucoma Fairness, JustRAIGS, GRAPE. Definitions in the appendix (?) | kNN on frozen features; LP (frozen, linear head on the CLS token); full FT | New 70/15/15 splits; counts not given (?) | None | Stratified by participant identity where available; 3 seeds; Friedman/Nemenyi tests | Accuracy, FT: DINOv2 ViT-S Reg best on Harvard FairVision glaucoma (74.31%); Virchow 65.77% on PAPILA. All tasks pooled: FT beat LP in mean accuracy for every encoder. No per-encoder RETFound or DINOv2-L glaucoma numbers in the main text |
| [13] Restrepo 2026 (FOCUS), arXiv. F | Vision encoders: ViT-L/16, DINOv2-L, DINOv3-L, RETFound variants (unspecified (?)), VisionFM-Fundus. VLMs: CLIP, SigLIP2, MedSigLIP, EyeCLIP, RET-CLIP, FLAIR. Multimodal LLMs (Qwen3-VL, Gemma-3, LLaVA-NeXT, MedGemma) | As published | Binary GON (VCDR ≥ 0.6 or specialist annotation) on BRSET, mBRSET, PAPILA, REFUGE, G1020. Handling of PAPILA suspects not stated (?) | Frozen encoder + balanced logistic-regression probe; VLM zero-shot + the same probe; LLMs zero-shot + LoRA SFT | Each dataset's own training split; sizes not stated (?) | None for encoders (within-dataset only) | Splits, patient-level grouping and seeds not stated (?); no CIs for the base benchmark | Glaucoma mean AUROC: general vision LP 0.742; ophthalmic vision LP 0.591; general VLM LP 0.745; ophthalmic VLM LP 0.719. Finding 3 text gives different values (0.701, 0.714). PAPILA mean vision AUROC (all tasks pooled): generalist 0.766 vs ophthalmic 0.710 |
| [14] Zhou 2026, Nat Commun. F | FM-MEH and FM-SDPP, each with MAE and with DINOv2 (ViT-L) | 904,170 CFP each, from MEH (UK) or SDPP (China) | Glaucoma Fundus 3-class; Retina 4-class with a glaucoma class | FT (encoder + FC) and LP (FC only); 50 epochs | As in [1]; sizes in Supp. Data (?) | None for glaucoma | Public splits not described (?); 5 seeds; Welch t-tests, Holm-Bonferroni | Glaucoma Fundus, MAE + LP: FM-MEH > FM-SDPP (p<0.01). Other glaucoma comparisons not significant. Numbers in Supp. Data 3 (?) |
| [15] Du 2024 (RET-CLIP), MICCAI. F | RET-CLIP (ViT-B + Chinese RoBERTa). Comparators: CN-CLIP, PMC-CLIP, DINOv2 (size (?)), RETFound (variant (?)), FLAIR | 193,865 patient-level image-report triplets (Beijing Tongren) | PAPILA (488) and Glaucoma Fundus (1,544), 3-class | LP (frozen + linear head) and full FT; 50 epochs each | PAPILA: "official" 80/20, then train split 4:1; others 0.56/0.14/0.30, class-stratified; counts not given | None | Patient-level not stated (?); 5 seeds, mean; no CIs | PAPILA LP: RET-CLIP 0.775, DINOv2 0.740, RETFound 0.620. PAPILA FT: RETFound 0.855, RET-CLIP 0.853, DINOv2 0.797. Glaucoma Fundus LP: RETFound 0.899, DINOv2 0.891. Glaucoma Fundus FT: RET-CLIP 0.958, DINOv2 0.955, RETFound 0.943 |
| [16] Silva-Rodríguez 2025 (FLAIR), Med Image Anal. F | FLAIR (ResNet-50 from ImageNet + BioClinicalBERT) | 38 public datasets, 288,307 CFPs including held-out sets. PAPILA (G/N) and CHAKSU (G/noG) are in pretraining; REFUGE held out | REFUGE (1,200, binary); ACRIMA (705, binary) | Zero-shot; LP = multinomial LR (L-BFGS) on frozen features; adapters; full FT of a ResNet-50 baseline. RETFound by LP only | k = 1/5/10 per class, or 20–80% (REFUGE 80% ≈ 960 images, calc); fixed 20% test | Yes: REFUGE ↔ ACRIMA cross-dataset (LP, adapters, FT); figures only | 5 CV folds; patient-level not stated (?); no CIs | Zero-shot REFUGE AUC: FLAIR 0.920 (expert prompts) vs BiomedCLIP 0.624, CLIP 0.433. LP REFUGE balanced accuracy: FLAIR 0.843, RETFound 0.747, ImageNet RN50 0.733 |
| [17] Yu 2024 (UrFound), MICCAI. F | UrFound ViT-B (initialised from natural-image MAE) | 187,270 public CFP + OCT images; CHAKSU (284) in pretraining, PAPILA not | PAPILA (488) and Glaucoma Fundus (1,544); class definitions not stated (?) | Full FT only (encoder + linear head), 50 epochs; no LP | PAPILA 313/77/98; Glaucoma Fundus 861/218/465; PAPILA not in the data-efficiency experiment | None | RETFound ratios; patient-level and seeds not stated (?); no CIs | FT AUROC, PAPILA: UrFound 78.32, FLAIR 74.80, RETFound-CFP 74.36, MAE 62.85. Glaucoma Fundus: UrFound 95.75, RETFound-CFP 95.68 |
| [18] Shi 2025 (EyeCLIP), npj Digit Med. F + S | EyeCLIP (CLIP + MAE decoder); backbone not stated (?) | 2,777,593 images, 11 modalities, 227 Chinese hospitals, + reports | PAPILA 3-class (333/87/68); Glaucoma Fundus 3-class (788/289/467) | Zero-shot; few-shot FT (encoder frozen or not: not stated (?)); full FT (encoder frozen 5 epochs, then unfrozen); no pure LP | PAPILA 312/79/98; Glaucoma Fundus 861/218/465; few-shot 1–16 per class | Partly: zero-shot on datasets absent from pretraining (Spain/Topcon, Korea/Nidek); no cross-dataset glaucoma transfer | 55:15:30; 5 seeds; patient-level not stated (?) | Full FT, PAPILA: EyeCLIP 0.829, RETFound 0.820, BioMedCLIP 0.795. Glaucoma Fundus: 0.957 / 0.950 / 0.900. PAPILA 16-shot: EyeCLIP 0.791, RETFound 0.760. PAPILA zero-shot: 0.684 / 0.514 / 0.493 |
| [19] Shi 2024 (EyeFound), arXiv. F | EyeFound: MAE ViT-L initialised from RETFound | 2,777,593 images, 11 modalities, 227 Chinese hospitals | PAPILA (488) and Glaucoma Fundus (1,544), 3-class | FT (MLP head, 50 epochs); unfreezing not stated separately; no LP | 55:15:30; counts not given (?) | None for glaucoma (DR only) | 5 random splits/seeds; t-tests; patient-level not stated (?) | PAPILA: EyeFound 0.813 vs RETFound 0.820 (p=0.014). Glaucoma Fundus: 0.955 vs 0.943 (p=0.002) |
| [20] Yang 2024 (ViLReF), arXiv. F | ViLReF ViT-B/16 (and an RN50 version) + Chinese RoBERTa | 451,956 image-report pairs (Beijing Tongren) | REFUGE only; class definition and count not stated (?) | LP only to compare pretraining strategies; FT and zero-shot against other FMs | Not stated (?) | None | 5 runs, mean (SD); paired t-tests; split not stated (?) | REFUGE FT AUC: ViLReF-ViT 97.91, RET-CLIP 95.82, RETFound 94.51, FLAIR 92.23. LP with the same data and backbone: MAE pretraining 51.40 vs ViLReF 98.57 |
| [21] Wu 2024 (MM-Retinal / KeepFIT), MICCAI. F | KeepFIT (ResNet-50 + BioClinicalBERT) | FLAIR assembly (278,348) + atlas pairs (2,169 CFP) | REFUGE only; class definition not stated (?) | Zero-shot; FC head on a frozen encoder (LP-like); how baselines were adapted not stated (?) | 20% train / 20% test; counts not given (?) | None (REFUGE unseen only in pretraining) | 5-fold CV; patient-level not stated (?); no CIs | Zero-shot REFUGE AUC: KeepFIT 0.941, FLAIR 0.926. Accuracy (ACA) with frozen encoder + FC: KeepFIT 0.861, FLAIR 0.831, RETFound 0.809 |
| [22] Qiu 2024 (VisionFM), NEJM AI. F (arXiv v1; NEJM 403) | VisionFM; architecture not stated in v1 (?) | 3.4M images, 560,457 individuals, 8 modalities | 8-disease benchmark (glaucoma listed only in the Fig. 1b graphic); glaucoma forecasting (407 subjects; source (?)) | Forecasting: one linear layer on frozen features | Forecasting 381 train / 26 test | None for glaucoma | Subject-level 381/26; no CIs | Forecasting F1 76.9%. 8-disease mean AUC 0.993; glaucoma-specific value only in figures (?) |
| [23] Guo 2025 (ReVision), arXiv. F | CLIP ViT-L/14@336 (OpenAI init) and ViT-B variants | 485,980 CFP-report pairs from 162 hospitals | REFUGE; AIROGS (light-v2 subset); class definition (?) | Zero-shot; LP (frozen, linear head); FT | Sizes in the supplement (?); label fractions for DR only | Yes: REFUGE ↔ AIROGS under LP and FT; directions not explicit (?) | 5 runs; t-tests | No glaucoma numbers in the main text (Figs 2a, 4a, ED Fig 6; Supp. S5–S12 (?)). LP ReVision beat every fine-tuned competitor on 11 of 12 disease benchmarks |
| [24] Wang 2025 (RetiZero), Nat Commun. F | RETFound (MAE) backbone, frozen, + LoRA on Q/V; BioClinicalBERT | 341,896 image-text pairs; REFUGE, AIROGS, ACRIMA, G1020, ORIGA, PARAGUAY included; PAPILA and Chaksu not | Glaucoma is one class of EYE-15 (30,089 images, 5 Chinese hospitals; Topcon TRC-NW8, Zeiss VISUCAM-200, Topcon Triton) | Zero-shot; FT within each hospital; 5-shot per class; whether FT is full or partial not stated (?) | 5 images per class (few-shot); H1 6,942/2,284/2,288 | Multi-disease cross-domain (clinic or camera); not glaucoma-specific | Patient-level not stated (?); 3 seeds | Zero-shot, glaucoma class: Top-1 0.748, Top-3 0.929, Top-5 0.972; no binary glaucoma AUC |
| [25] Woerner 2024, MICCAI workshop. F | 16 models: BiomedCLIP ViT-B/16; CLIP ViT-B/L/H/g; DINOv2 B/L/g; supervised ViT-B/L/H; ResNet-18/50/101 | ImageNet / ImageNet-21k; DINOv2 (LVD-142M not named); CLIP 400M; LAION-2B; PMC-15M | MedIMeta "glaucoma" task = Chákṣu (1,345 images, 3 devices pooled; binary "glaucoma suspect" by majority of 5 experts) [27] | LP for all models; FT for ResNet-18/50; zero-shot for CLIP-type models | 1–30 labelled images per class; 10 query images per class; 100 task instances | None (cameras pooled) | Support and query sets subject-disjoint; mean over 100 instances; no CIs | AUROC at n=30: ResNet-18 FT 73.8, CLIP-ViT-H LP 71.2, ResNet-50 LP 70.5, BiomedCLIP LP 69.7, DINOv2-L LP 68.8. At n=10: ResNet-18 LP 63.0, BiomedCLIP 60.1, DINOv2-L 58.9 |
| [26] Zou 2025 (FusionFM), arXiv. F | RETFound, VisionFM, RetiZero, DINORET and mixture-of-experts fusions | As described by FusionFM (secondary) | Chaksu (1,345: 188 glaucoma / 1,157 not; 3 cameras pooled); PAPILA (488: 333/68/87); binary vs 3-class task not stated (?) | Frozen backbones + head stated; wording on layer-wise LR decay / "fine-tuning" conflicts (?) | 7:1:2 split (about 941 Chaksu and 342 PAPILA training images, calc) | None for glaucoma | Patient-level and number of seeds not stated (?); 1,000-bootstrap CIs | Numbers only in Fig 2 (?). Text: DINORET best single FM on PAPILA; best fusion on Chaksu +4% AUC and +18% F1 over the best single FM |

---

## C. Answers to the supervisor's five questions

### C1. Which retinal and medical foundation models exist

**Retinal, image-only self-supervised**
- RETFound-MAE: ViT-L; ImageNet-1k init, then 904,170 CFP + 736,442 OCT [1][4][11].
- RETFound-DINOv2: ViT-L; DINOv2 on 904K AlzEye images, initialised from DINOv2-L [3]. Matching cohort-controlled models appear in [14].
- EyeFound: MAE ViT-L initialised from RETFound; 2.78M images, 11 modalities, China [19].
- VisionFM: 3.4M images, 8 modalities. Architecture not stated in arXiv v1; [26] describes it as iBOT-based (secondary) [22].
- DINORET: block-expanded DINOv2, >202K CFP (secondary description in [26]; original not read).
- RETFound-Green: named only in [12].

**Retinal, vision-language**
- RET-CLIP: ViT-B, Chinese reports, 193,865 patient triplets [15].
- FLAIR: ResNet-50, 38 public datasets with expert-knowledge text [16].
- KeepFIT: ResNet-50, FLAIR data + atlas pairs [21].
- UrFound: ViT-B, CFP + OCT with label text, 187,270 images [17].
- ViLReF: ViT-B/16, 451,956 pairs [20].
- EyeCLIP: CLIP + MAE, 2.78M images; backbone not stated [18].
- RetiZero: RETFound + LoRA, 341,896 pairs [24].
- ReVision: CLIP ViT-L/14@336, 485,980 CFP-report pairs [23].

**General biomedical**
- BiomedCLIP: ViT-B/16 on PMC-15M per [25]; its own paper was not read. Used as a comparator in [16], [18] and [25].
- PMC-CLIP, PubMedCLIP and MedCLIP appear only as comparators ([15], [18], [20]). MedSigLIP and MedGemma appear in [13].
- Non-ophthalmic medical FMs used as baselines: RadDINO, Virchow/Virchow2, Prov-GigaPath [12].

**General-purpose**
- DINOv2 S/B/L/g on LVD-142M [3][10]; DINOv3 S/S+/B/L/H+ on LVD-1689M [3].
- CLIP, OpenCLIP, SigLIP/SigLIP2 [12][13][25].
- Supervised ImageNet ResNet-50, ViT and SwinV2 [9][11][25].

### C2. How each was evaluated for glaucoma

**3-class PAPILA and Glaucoma Fundus with macro AUROC (the RETFound protocol)**
- [1][3][10][15][18][19]; Glaucoma Fundus only in [14]. [17] follows the protocol but does not state classes.

**Binary glaucoma**
- Private cohorts: [4][5][7][8][9][11].
- REFUGE: [16][20][21][23], and [9] as an external set.
- Chaksu: [25][26].
- Binary GON on PAPILA, REFUGE, G1020, BRSET and mBRSET: [13].
- PAPILA as a binary external test: [9] (suspects excluded) and [11] (suspect handling not stated).

**Other metrics**
- Accuracy: [2][12]. Balanced/class accuracy (ACA): [16][21]. Zero-shot Top-k for a glaucoma class: [24]. Forecasting F1: [22]. Optic-nerve regression R²: [6].

**Zero-shot glaucoma evaluation**
- [16][18][20][21][23][24]; VLMs and LLMs in [13].
- Zero-shot numbers are confounded where the dataset was in pretraining: PAPILA and CHAKSU in FLAIR [16]; REFUGE and AIROGS in RetiZero [24][23].

### C3. Fine-tuning versus a classifier on frozen features (linear probe)

**Full fine-tuning only**
- [1][2][4][5][7][8][9][10][11][17][19].
- Whether all layers were updated is not stated in words in [1][4][5][10][19].

**Both LP and FT reported**
- [3]: LP is a gradient-trained FC head with augmentation.
- [14]: FC only, same schedule as FT.
- [15]: 50 epochs each.
- [12]: also kNN.
- [23]: glaucoma numbers not in the main text.
- [16]: LP for the FMs; FT only for the ResNet-50 baseline.
- [25]: LP for all models; FT for ResNet-18/50.
- [20]: LP only for comparing pretraining strategies.

**Classifier or regressor on cached frozen features**
- [13]: balanced logistic regression.
- [16]: L-BFGS multinomial logistic regression.
- [6]: linear regressors, regression targets only.
- [22]: one linear layer, forecasting.
- [21]: FC on a frozen encoder.
- [26]: frozen heads stated, but the wording conflicts.

**LP versus FT gap on PAPILA (3-class AUROC)**

| Model | LP | FT | Source |
|---|---|---|---|
| RETFound-MAE | 0.748 | 0.832 | [3] |
| RETFound (variant not stated) | 0.620 | 0.855 | [15] |
| RETFound-DINOv2 | 0.854 | 0.916 | [3] |
| DINOv3-L | 0.738 | 0.855 | [3] |
| DINOv2-g | 0.701 | 0.834 | [3] |
| DINOv2-L | 0.687 | 0.662 (FT below LP) | [3] |

- Across the 10 tasks of [3], FT > LP for every model (significance stars in Fig 5).
- [12], all tasks pooled: FT beat LP in mean accuracy for every encoder. Spearman correlation between FT and LP ranks was 0.58–0.62.

**Cost**
- LP took about 15 min vs about 70 min for FT per 1000 images [14].
- DINOv2-g FT needs about 5× the memory and 3× the runtime of its LP [3].

### C4. How much labelled data

**Public glaucoma dataset sizes**
- PAPILA: 488–489 images; 312 training images in the RETFound split [1][3][18].
- Glaucoma Fundus: 1,544 images; 861 training [1].
- REFUGE: 1,200 [16]. ACRIMA: 705 [16]. Chaksu: 1,345 [25][26]. Retina: 601 [3].

**Private cohorts**
- KULRD: 61,213 images [9].
- LA County DHS: 12,998 images [5].
- DIGS/ADAGES: 9,787 images [4]; 14,510 images [7].
- Chinese health-examination cohort: 3,088 images in total [8].
- Stanford: 776 CFP-OCT pairs [6].

**Label-efficiency designs**

| Paper | Training sizes | Adaptation |
|---|---|---|
| [4] | 50–2000 images | FT |
| [5] | 20–6000 images | FT |
| [11] | 100/200/400 images and 20/50/100% | FT, with external tests |
| [8] | 10–100% (10% = 309 images in total) | FT |
| [3] | 10/30/50/100% (about 31/94/156/312 PAPILA images, calc) | LP and FT |
| [10] | 10–100% | FT |
| [18] | 1–16 per class | few-shot |
| [25] | 1–30 per class | LP and FT |
| [16] | k = 1/5/10 per class and 20–80% | LP |

**Small-data facts**
- [4]: AUC 0.574 at 50 images, 0.624 at 100, 0.699 at 200 (RETFound FT).
- [11]: at ≤400 images, RETFound beat ResNet50; SwinV2 was comparable.
- [3]: at 10% LP on PAPILA, both frozen generalists (DINOv2-g 0.631, DINOv3-L 0.637) were above both frozen RETFound models (0.613, 0.590). RETFound-DINOv2 LP leads clearly from 50%.
- [25]: on Chaksu at n=10 per class, ResNet-18 LP (63.0) was above BiomedCLIP (60.1) and DINOv2-L (58.9).
- [8]: at 10% data, RETFound vs ViT-L was 0.838 vs 0.808 (p=.557).
- [5]: RETFound led VGG-19 at 400 and 1000 images.

### C5. Tests on data from a different hospital or camera

**Glaucoma cross-dataset or cross-site tests without retraining**
- [5]: LA County DHS to USC Roski (different cameras), FT.
- [9]: KULRD to 6 datasets, FT.
- [10]: Glaucoma Fundus to PAPILA, FT.
- [11]: SEED to ODIR-5K, PAPILA and GAMMA, FT.
- [16]: REFUGE ↔ ACRIMA with LP, adapters and FT; figures only.
- [23]: REFUGE ↔ AIROGS with LP and FT; numbers in a supplement not accessed.

**External results**
- [10]: DINOv2-L 0.625 vs RETFound 0.558, from about 0.96 / 0.94 internally. The authors attribute the drop to differences in resolution, field of view and glaucoma definition.
- [5]: uncropped RETFound 0.886 vs uncropped VGG-19 0.634.
- [9]: PAPILA RETFound 0.88, SwinV2 0.87, DINOv2 0.85.
- [11]: PAPILA 0.779 (RETFound, 400 images).

**Every cross-dataset glaucoma number available as text comes from a fine-tuned model** ([5][9][10][11]).

**No external glaucoma test**
- [1][2][3][4][6][7][8][12][14][15][17][19][20][21][22][25][26].
- [13] has none for its encoders.
- [18] is zero-shot only, on datasets absent from pretraining.
- [24] has cross-domain tests that are not glaucoma-specific.

**Per-camera reporting**
- No paper read reports glaucoma results broken down by camera.
- Chaksu's 3 cameras are pooled in [25] and [26].

**Label definitions differ between the datasets**
- PAPILA: two experts after a clinical examination [1].
- Glaucoma Fundus: two specialists using visual fields and imaging [1].
- Chaksu: majority vote of 5 experts on a "glaucoma suspect" label, per the MedIMeta check in the [25] record [27]; majority vote per [26].

---

## D. Gaps the project addresses

Status: **HOLDS**, **NARROWED** (part of the broad gap is contradicted), or **DROPPED** (contradicted as worded).

According to the project description, the project trains only logistic regression on frozen features. It does not fine-tune.

**G1. Binary glaucoma vs healthy on PAPILA with patient-level folds: HOLDS, with one caveat.**
- Closest papers:
  - [1]: 3-class; released split measured as not patient-level.
  - [3]: same 312/79/98 counts; Methods say "patient groups", but PAPILA's ratio does not match.
  - [15][17][18][19][10][26]: patient-level grouping not stated.
  - [2]: image-level stratified split.
  - [13]: binary GON on PAPILA; splits not stated.
  - [12]: splits stratified by participant where available. Whether this applies to PAPILA is not verified (?). Accuracy metric; PAPILA number reported only for Virchow under FT.
  - [9][11]: PAPILA used only as an external binary test set.
- What none did: no paper read states that it trained and tested a binary PAPILA classifier with patient-grouped splits.

**G2. Cross-camera external validation reported per camera, on Chaksu, without retraining: HOLDS.**
- Closest papers:
  - [10][9][11][5]: cross-site glaucoma tests, all fine-tuned.
  - [5]: several external cameras, with no per-camera breakdown in the record (?).
  - [16]: frozen LR cross-dataset test (REFUGE ↔ ACRIMA), figures only.
  - [23]: LP cross-dataset test (REFUGE ↔ AIROGS), numbers not in the main text.
  - [25][26]: Chaksu evaluated internally with cameras pooled.
- What none did:
  - No paper trains on PAPILA and tests on Chaksu.
  - None reports glaucoma AUROC per camera.
  - None gives frozen-feature external glaucoma numbers in text.
- Caveats:
  - Chaksu is in the pretraining of FLAIR [16] (1,345) and UrFound [17] (284), and probably KeepFIT [21] (inf).
  - PAPILA is in FLAIR's pretraining [16].
  - Whether either dataset is in the pretraining of RETFound, DINOv2 or DINOv3: not checked (?).

**G3. Frozen-feature label efficiency at ≤200 training images, evaluated externally: NARROWED.**
- The broad gap "no glaucoma data below ~200 images" is contradicted: [4][5][11][3][25][18][16][8].
- Closest papers and what they did not do:
  - [3]: frozen LP at about 31/94/156 PAPILA images with DINOv3-L and both RETFound models. But 3-class, one image-level split, one run per fraction, a gradient-trained head, internal test only, and no ResNet-50 or SwinV2.
  - [25]: frozen LP at 1–30 per class on Chaksu. But 20-image test sets, cameras pooled, no RETFound or DINOv3, and no external test.
  - [11]: small-data results with external tests, but fine-tuned only.
  - [16]: an LP cross-dataset test at k=10 shots (Fig S4), figures only, on REFUGE/ACRIMA.
- Residual gap: a binary, patient-level frozen LR curve at 25–200 images. The external part of this gap holds only if each training size is also tested on Chaksu, which the project description does not state (?).

**G4. One frozen LR protocol across DINOv3-L, DINOv2-L, both RETFound checkpoints, ImageNet ResNet-50 and SwinV2-tiny: NARROWED.**
- The gap as worded ("DINOv3 vs RETFound under the same frozen protocol") is dropped; see below.
- Closest papers and what they did not do:
  - [3]: DINOv2 and DINOv3 at all sizes plus both RETFound checkpoints under LP. No CNN or Swin.
  - [13]: DINOv2-L, DINOv3-L and RETFound variants with an LR probe. Per-model glaucoma numbers not in the text read.
  - [15]: DINOv2 vs RETFound LP.
  - [12]: DINOv2, RETFound variants and ResNet-50 under LP. No DINOv3; accuracy metric; per-encoder glaucoma numbers not in the main text.
  - [25]: DINOv2 and ResNet LP. No RETFound.
- SwinV2 appears only under full fine-tuning ([2][9][11]).
- Residual gap: no paper read evaluates all six models under one frozen protocol, with or without an external test.

**G5. RETFound checkpoint dependence (MAE vs DINOv2) under external or cross-camera glaucoma testing: NARROWED.**
- Evidence that checkpoint dependence exists:
  - [3]: PAPILA LP 0.854 vs 0.748; FT 0.916 vs 0.832.
  - [14]: DINOv2 > MAE overall; the Glaucoma Fundus MAE + LP result was sensitive to the pretraining cohort.
  - [2]: RETFound-DINOv2-CFP is in the glaucoma top 5; MAE is not.
  - [1] Supp. T4: DINO v1 (0.788) was below MAE (0.855) under FT.
- RETFound's PAPILA results also vary across papers: FT 0.744 [17] to 0.855 [1][15]; LP 0.620 [15] vs 0.748 [3].
- RETFound's architecture is described inconsistently: ViT-B per [9][16], ViT-L per [1][3][4][20].
- What none did: no paper compares the two checkpoints on an external glaucoma test set. [3]'s external tests are DR and systemic only, and fine-tuned.

**Dropped as worded (contradicted by the evidence)**
- "No frozen-vs-fine-tuning comparison for glaucoma": contradicted by [3][14][15][12], with partial evidence in [16][25]. The project does not fine-tune, so it cannot add a matched comparison.
- "No DINOv3 vs RETFound comparison under the same frozen protocol": contradicted by [3] (PAPILA and Glaucoma Fundus LP) and [13] (LR probe including DINOv3-L).
- "No glaucoma label-efficiency evidence below ~200 images": contradicted by [3][4][5][11][16][18][25].

---

## E. References

[1] Y. Zhou et al., "A foundation model for generalizable disease detection from retinal images," *Nature*, vol. 622, no. 7981, pp. 156–163, 2023, doi: 10.1038/s41586-023-06555-x.

[2] D. Isztl et al., "When do domain-specific foundation models justify their cost? A systematic evaluation across retinal imaging tasks," arXiv:2511.22001, 2025. https://arxiv.org/abs/2511.22001

[3] Y. Zhou et al., "Generalist versus specialist vision foundation models for ocular disease and oculomics," arXiv:2509.03421, 2025. https://arxiv.org/abs/2509.03421

[4] B. Chuter et al., "Evaluating a foundation artificial intelligence model for glaucoma detection using color fundus photographs," *Ophthalmol. Sci.*, vol. 5, no. 1, Art. 100623, 2025, doi: 10.1016/j.xops.2024.100623.

[5] K. Bolo et al., "Comparison of RETFound and a supervised convolutional neural network for detection of referable glaucoma from fundus photographs," *Ophthalmol. Sci.*, vol. 6, no. 2, Art. 101008, 2026, doi: 10.1016/j.xops.2025.101008.

[6] M. S. Chen et al., "Independent evaluation of RETFound foundation model's performance on optic nerve analysis using fundus photography," *Ophthalmol. Sci.*, vol. 5, no. 3, Art. 100720, 2025, doi: 10.1016/j.xops.2025.100720.

[7] B. Chuter, V. Joshi et al., "A novel multimodal implementation of a foundation artificial intelligence model using optic nerve head fundus photographs and OCT imaging for glaucoma detection," *Ophthalmol. Sci.*, vol. 6, no. 2, Art. 101012, 2026, doi: 10.1016/j.xops.2025.101012.

[8] Z. Xiong et al., "How generalizable are foundation models when applied to different demographic groups and settings?," *NEJM AI*, vol. 2, no. 1, 2025, doi: 10.1056/AIcs2400497.

[9] O. Abramovich et al., "GONet: A generalizable deep learning model for glaucoma detection," *IEEE Trans. Biomed. Eng.*, vol. 73, no. 1, pp. 32–39, 2026, doi: 10.1109/TBME.2025.3576688. arXiv:2502.19514.

[10] Q. Hou et al., "Can a natural image-based foundation model outperform a retina-specific model in detecting ocular and systemic diseases?," *Ophthalmol. Sci.*, vol. 6, no. 1, Art. 100923, 2026, doi: 10.1016/j.xops.2025.100923.

[11] S. M. E. Yew et al., "Performance and label efficiency of traditional deep-learning models and a retina-specific foundation model for ocular and systemic disease detection: a retrospective comparative study," *Lancet Digit. Health*, vol. 8, no. 9, Art. 101031, 2026, doi: 10.1016/j.landig.2026.101031. Preprint: arXiv:2501.12016.

[12] J. Zoellin, C. Merk et al., "Benchmarking vision encoders for image classification in ophthalmology," *Comput. Struct. Biotechnol. J.*, vol. 35, Art. 0178, 2026, doi: 10.34133/csbj.0178.

[13] D. Restrepo et al., "FOCUS: Benchmarking retinal model generalization from foundation vision encoders to multimodal LLMs," arXiv:2609.33158, 2026. https://arxiv.org/abs/2609.33158

[14] Y. Zhou, Z. Wang, Y. Wu et al., "Understanding pre-training data effects in retinal foundation models using two large fundus cohorts," *Nat. Commun.*, vol. 17, Art. 3309, 2026, doi: 10.1038/s41467-026-70077-z.

[15] J. Du et al., "RET-CLIP: A retinal image foundation model pre-trained with clinical diagnostic reports," in *Proc. MICCAI*, 2024. arXiv:2405.14137.

[16] J. Silva-Rodríguez et al., "A foundation language-image model of the retina (FLAIR): Encoding expert knowledge in text supervision," *Med. Image Anal.*, vol. 99, Art. 103357, 2025, doi: 10.1016/j.media.2024.103357. arXiv:2308.07898.

[17] K. Yu et al., "UrFound: Towards universal retinal foundation models via knowledge-guided masked modeling," in *Proc. MICCAI*, LNCS 15012, 2024. arXiv:2408.05618.

[18] D. Shi et al., "A multimodal visual–language foundation model for computational ophthalmology," *npj Digit. Med.*, vol. 8, Art. 381, 2025, doi: 10.1038/s41746-025-01772-2. arXiv:2409.06644.

[19] D. Shi et al., "EyeFound: A multimodal generalist foundation model for ophthalmic imaging," arXiv:2405.11338, 2024. https://arxiv.org/abs/2405.11338

[20] S. Yang et al., "ViLReF: An expert knowledge enabled vision-language retinal foundation model," arXiv:2408.10894, 2024. https://arxiv.org/abs/2408.10894

[21] R. Wu et al., "MM-Retinal: Knowledge-enhanced foundational pretraining with fundus image-text expertise," in *Proc. MICCAI*, LNCS, 2024, pp. 722–732, doi: 10.1007/978-3-031-72378-0_67. arXiv:2405.11793.

[22] J. Qiu et al., "Development and validation of a multimodal multitask vision foundation model for generalist ophthalmic artificial intelligence," *NEJM AI*, vol. 1, no. 12, 2024, doi: 10.1056/AIoa2300221. arXiv:2310.04992, "VisionFM: a multi-modal multi-task vision foundation model for generalist ophthalmic artificial intelligence."

[23] J. Guo et al., "Native intelligence emerges from large-scale clinical practice: A retinal foundation model with deployment efficiency," arXiv:2512.14499, 2025. https://arxiv.org/abs/2512.14499

[24] M. Wang et al., "Enhancing diagnostic accuracy in rare and common fundus diseases with a knowledge-rich vision-language model," *Nat. Commun.*, vol. 16, Art. 5528, 2025, doi: 10.1038/s41467-025-60577-9. arXiv:2406.09317.

[25] S. Woerner et al., "Navigating data scarcity using foundation models: A benchmark of few-shot and zero-shot learning approaches in medical imaging," MICCAI 2024 Workshop on Foundation Models for General Medical AI, arXiv:2408.08058, 2024. https://arxiv.org/abs/2408.08058

[26] K. Zou et al., "FusionFM: Fusing eye-specific foundational models for optimized ophthalmic diagnosis," arXiv:2508.11721, 2025. https://arxiv.org/abs/2508.11721

[27] S. Woerner, A. Jaques, C. F. Baumgartner, "A comprehensive and easy-to-use multi-domain multi-task medical imaging meta-dataset," arXiv:2404.16000, 2024. https://arxiv.org/abs/2404.16000 (Only the arXiv metadata was checked for this pack. The Chákṣu facts come from the verification in the [25] record.)

---

## F. Not verified / could not access

**Could not access**
- [11]: Lancet full text returned 403 / a bot challenge, and it is subscription-only on Europe PMC. Read the abstract and preprint only. Supp. Tables 1–2 (glaucoma definition, counts) not read.
- [8]: NEJM AI page returned 403. Supplement (label definition, camera, patient grouping, FT details) not read.
- [22]: NEJM AI version returned 403. Read arXiv v1 only. Architecture, pretraining algorithm, per-disease glaucoma AUC (figure only) and forecasting data source are unknown.
- [12]: the appendix (glaucoma task definitions, per-task accuracies) was blocked by a bot check.
- [23]: Supplementary Tables S1–S12 (all glaucoma numbers, class definitions, training sizes, transfer directions) not accessed.

**Not read**
- [2] and the main text of [1] were taken from an earlier summary.
- The RETFound fine-tuning code was not checked, so whether all layers train is unknown.
- PAPILA source paper: official counts, file naming (`RET<patient ID><eye>` is assumed) and official partition not checked.
- Chaksu source paper not read. The camera names (Remidio, Forus, Bosch) come only from the project description.
- BiomedCLIP, DINOv2 and DINOv3 original papers not read.
- DINORET (arXiv:2409.17332), the Research Square preprint rs-6080254 (RETFound-DINOv2; listed as the preprint of [14]) and RETFound-Green not read.
- The medRxiv preprint of [5] (10.1101/2025.08.21.25334170) and the VGG-19 source paper cited by [5] not read.

**Not checked**
- Whether PAPILA or Chaksu is in the pretraining data of RETFound-MAE, RETFound-DINOv2, DINOv2 (LVD-142M) or DINOv3 (LVD-1689M).
- Whether [3] and [18] use the same image split that [1] released (the counts are identical).
- Why RETFound's PAPILA AUROC differs across [1]/[15] (0.855), [18]/[19] (0.820), [3] (0.832) and [17] (0.744).

**Unverified items inside individual records**

| Paper | What remains unverified |
|---|---|
| [3] | Journal version check incomplete (Semantic Scholar rate-limited). The CI method (stated 1.96·SD/√200) does not match the tabulated widths. |
| [4] | Internal inconsistencies: 2,329 vs 2,104 participants; training pool printed as both 6,884 and 6,684. |
| [5] | Supp. Tables S1–S2; per-camera external results; the Results vs Discussion inconsistency at 2000 images. |
| [6] | Supp. Tables S1, S5, S6. |
| [7] | Table S1. |
| [9] | TBME published version not read; patient-level split. |
| [10] | Supp. Tables S1 and S3 (internal PAPILA AUROC); camera names. |
| [13] | Split ratios, patient grouping, seeds; the undefined "Shift" column; NeurIPS proceedings; the RETFound variants used. |
| [14] | Supp. Data 1–3 and 6 (all glaucoma numbers). |
| [15] | Supplementary t-tests; DINOv2 size; RETFound variant. |
| [16] | Values in the cross-dataset and label-efficiency figures; journal typeset version. |
| [17] | Values in the data-efficiency figure; the MICCAI page was not opened. |
| [18] | Vision backbone. |
| [19] | Supp. Tables 1 and 4. |
| [20] | REFUGE class definition and split; venue. |
| [21] | REFUGE class definition, split counts, how the baselines were adapted. |
| [24] | Whether glaucoma is a class in H1/H2/H3 and the cross-domain sets; binary glaucoma metrics. |
| [25] | Which MedIMeta split the few-shot tasks were drawn from. |
| [26] | Per-model glaucoma AUC/F1 (Fig. 2 only); number of seeds; whether PAPILA was trained as binary or 3-class. |

Lookups made for this pack: Crossref metadata for [1] and arXiv API metadata for [2] and [27], saved in `/private/tmp/claude-501/-Users-jiaxi-Desktop-electronics-communication-coursework/834f4611-cd03-4187-b13f-705824cb76c4/scratchpad/lit/synthesize/`.