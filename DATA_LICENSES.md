# Data and results: sources and licences

The code is under the MIT licence (`LICENSE`). The data files, results and figures in this repository are derived from
two public datasets and keep their licences:

| Dataset | Source | Licence | Files derived from it |
|---|---|---|---|
| Chákṣu IMAGE (1,345 fundus images, 3 cameras, 5 expert labels) | Scientific Data 10, 70 (2023), doi:10.1038/s41597-023-01943-4; images: figshare, doi:10.6084/m9.figshare.20123135 | CC BY 4.0 | `chaksu_labels.csv`, `pipeline/chaksu_labels.csv`; everything in `3_Nov/` and `after_3_Nov/` (splits, RETFound-Green features `*.npz`, Optuna databases, scores, predictions, test results, figures); `results/**/chaksu_*.csv` and the Chákṣu numbers in `results/` |
| PAPILA (fundus images and clinical data of both eyes) | Scientific Data 9, 291 (2022), doi:10.1038/s41597-022-01388-1; images: figshare, doi:10.6084/m9.figshare.14798004 | GPL-3.0-or-later | `papila_labels.csv`, `pipeline/papila_labels.csv`; `results/**/papila_*.csv` and the PAPILA numbers in `results/` |

- Files derived from PAPILA are distributed under the GNU General Public License, version 3 or later
  (`LICENSES/GPL-3.0.txt`).
- Files derived from Chákṣu are distributed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); credit the
  Chákṣu authors and the article above.
- Files in `results/` that combine both datasets credit both; their PAPILA parts are under GPL-3.0-or-later.
- The features (`chaksu_rfg_resize.npz`) are the frozen RETFound-Green outputs for the Chákṣu images
  (RETFound-Green: Engelmann and Bernabeu, Nature Communications, 2025).
- Not included: the fundus images themselves (download them from the figshare records above) and one QC montage of
  disc crops, because a crop shows a patient ID, exam date and medical record number burned in by the camera.
