"""FairVLMed versions of the data helpers; everything else (MLP, losses, thresholds, bootstrap) is reused from
../after_10_Nov/common.py so the method is the same as on Chakshu.
One image per patient, so groups = images; the dataset's own split is used: training 7,000 + validation 1,000
(5-fold cross-validation, stratified by label) and test 2,000 (kept aside)."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_spec = importlib.util.spec_from_file_location('chaksu_common', Path(__file__).parent.parent / 'after_10_Nov' / 'common.py')
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
for _name in ['SEED', 'make_mlp', 'make_loss', 'train_mlp', 'mlp_scores', 'n_trainable', 'threshold_at_recall',
              'youden_threshold', 'at_threshold', 'bootstrap_ci', 'paired_difference']:
    globals()[_name] = getattr(_base, _name)

LABELS = 'fairvlmed_labels.csv'
PREPROC = os.environ.get('PREPROC', 'stretch')          # input chosen by choose_input.py (raw or stretch)
FEATURES = f'/nobackup/tggq61/retfound_green/features/fairvlmed_rfg_{PREPROC}.npz'


def load_data(features=None):
    """Label table (one row per patient; column `eye` = image_id so the shared code's groups are one image each),
    features X (10,000 x 384), labels y (1 = glaucoma)."""
    lab = pd.read_csv(LABELS)
    lab['eye'] = lab['image_id']
    z = np.load(features or FEATURES)
    X = pd.DataFrame(z['features'], index=z['image_id']).loc[lab['image_id']].to_numpy()
    return lab, X, lab['label'].to_numpy()


def cv_split(lab, n_splits=5):
    keep = lab['part'].isin(['train', 'val']).to_numpy()
    sub = lab[keep]
    splits = list(StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(sub, sub['label']))
    return keep, splits


def cv_folds(lab, X, y, n_splits=5):
    keep, splits = cv_split(lab, n_splits)
    X, y = X[keep], y[keep]
    out = []
    for tr, va in splits:
        sc = StandardScaler().fit(X[tr])
        out.append((sc.transform(X[tr]), y[tr], sc.transform(X[va]), y[va]))
    return out


def mlp_small_storage():
    import optuna
    from optuna.storages.journal import JournalFileBackend
    return optuna.storages.JournalStorage(JournalFileBackend(f'optuna_mlp_small_{PREPROC}.log'))
