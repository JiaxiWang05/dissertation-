"""Step 1: split the 1345 images into train / val / test = 70 / 15 / 15 and save chaksu_split.csv.
Grouped by eye (both images of an eye, and an exact copy and its original, always go to the same part).
Stratified by glaucoma label and camera, so each part has about the same glaucoma share and camera mix."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import SEED, SPLIT, load_labels

lab = load_labels()
strata = lab['label_majority'].astype(str) + '_' + lab['camera']       # 6 groups: (healthy, glaucoma) x 3 cameras

# cut the eyes into 20 folds of about 5% each; then 3 folds = test (15%), 3 folds = val (15%), 14 folds = train (70%)
# (sklearn warns that one group, Forus glaucoma, has only 18 images for 20 folds; the shares still come out even)
folds = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
fold_of = pd.Series(-1, index=lab.index)
for k, (_, rows) in enumerate(folds.split(lab, strata, groups=lab['eye'])):
    fold_of[rows] = k
lab['split'] = fold_of.map(lambda k: 'test' if k < 3 else 'val' if k < 6 else 'train')

lab[['image_id', 'split']].to_csv(SPLIT, index=False)
print('saved', SPLIT, '  images:', len(lab), '  eyes:', lab['eye'].nunique())
