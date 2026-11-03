"""Shared code for the RETFound-Green pipeline: model loading, preprocessing, data, metrics.

Model loading follows the official example (https://github.com/justinengelmann/RETFound_Green):
timm `vit_small_patch14_reg4_dinov2`, 392x392 input, mean/std 0.5, average-pooled tokens -> 384-d.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as T

IMG_SIZE = 392
MEAN = STD = [0.5, 0.5, 0.5]
HAMILTON_DATA = '/nobackup/tggq61/data'  # prefix of the paths stored in the label CSVs
CLINICAL_COLS = ['age', 'gender']  # non-diagnostic only; IOP / visual field / CDR would leak the label


# ---------------------------------------------------------------- model

def load_retfound_green(weights, dynamic_img_size=False):
    """dynamic_img_size=True lets the model take other input sizes (position embedding interpolated);
    used only for quick smoke tests. timm >= 0.9 supports it."""
    import timm
    kw = {'dynamic_img_size': True} if dynamic_img_size else {}
    model = timm.create_model('vit_small_patch14_reg4_dinov2', img_size=(IMG_SIZE, IMG_SIZE),
                              num_classes=0, checkpoint_path=weights, **kw)
    model.global_pool = 'avg'
    return model


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ---------------------------------------------------------------- images

def fov_square(img, thresh=15):
    """Crop to the bounding box of the circular field of view, then pad to a square with black."""
    mask = img.max(0).values > thresh
    rows = torch.nonzero(mask.float().mean(1) > 0.01).flatten()
    cols = torch.nonzero(mask.float().mean(0) > 0.01).flatten()
    if len(rows) and len(cols):
        img = img[:, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    _, h, w = img.shape
    s = max(h, w)
    out = torch.zeros((3, s, s), dtype=img.dtype)
    out[:, (s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return out


def disc_crop(img, cx, cy, d, scale=3.0):
    """Square crop centred on the optic disc, side = scale x disc diameter (black padding past the image edge)."""
    _, h, w = img.shape
    half = int(round(scale * d / 2))
    out = torch.zeros((3, 2 * half, 2 * half), dtype=img.dtype)
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * half, w), min(y0 + 2 * half, h)
    out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[:, sy0:sy1, sx0:sx1]
    return out


def eval_transform(img_size=IMG_SIZE):
    return T.Compose([T.Resize((img_size, img_size), antialias=True),
                      T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


def train_transform(img_size=IMG_SIZE):
    return T.Compose([T.RandomResizedCrop((img_size, img_size), scale=(0.85, 1.0), ratio=(0.95, 1.05),
                                          antialias=True),
                      T.RandomHorizontalFlip(), T.RandomRotation(10),
                      T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                      T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


def local_paths(paths, data_root=None):
    """Map the Hamilton paths stored in the CSVs to another data root (e.g. on the laptop)."""
    if not data_root:
        return list(paths)
    return [p.replace(HAMILTON_DATA, str(data_root).rstrip('/'), 1) for p in paths]


class FundusImages(Dataset):
    """Reads each image once per call; preproc is 'resize' (official), 'fov_square' or 'disc_crop'."""

    def __init__(self, paths, transform, preproc='resize', labels=None, cache_size=None, boxes=None, crop_scale=3.0):
        self.paths, self.transform, self.preproc = list(paths), transform, preproc
        self.boxes, self.crop_scale = boxes, crop_scale  # boxes: (cx, cy, disc diameter) per image, for 'disc_crop'
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.cache_size = cache_size  # if set, keep a resized uint8 copy of every image in memory
        self.cache = {}

    def __len__(self):
        return len(self.paths)

    def _load(self, i):
        if i in self.cache:
            return self.cache[i]
        img = read_image(self.paths[i], mode=ImageReadMode.RGB)
        if self.preproc == 'fov_square':
            img = fov_square(img)
        elif self.preproc == 'disc_crop':
            img = disc_crop(img, *self.boxes[i], scale=self.crop_scale)
        if self.cache_size:
            # small margin above the network size so random crops still have detail
            img = T.functional.resize(img, [self.cache_size, self.cache_size], antialias=True)
            self.cache[i] = img
        return img

    def __getitem__(self, i):
        x = self.transform(self._load(i))
        return (x, i) if self.labels is None else (x, self.labels[i], i)


# ---------------------------------------------------------------- datasets

def papila_binary(labels_csv, suspects='exclude'):
    """PAPILA glaucoma (1) vs healthy (0). suspects: 'exclude' | 'negative' | 'positive'."""
    df = pd.read_csv(labels_csv)
    if suspects == 'exclude':
        df = df[df['diagnosis'] != 'suspect']
    pos = ['glaucoma'] + (['suspect'] if suspects == 'positive' else [])
    df = df.reset_index(drop=True)
    df['y'] = df['diagnosis'].isin(pos).astype(int)
    return df


def papila_folds(df, scheme='official', n_folds=5, seed=0):
    """Fold id per row. 'official' = PAPILA's own patient-level 'Test 2' folds (glaucoma vs healthy)."""
    if scheme == 'official':
        if df['fold_test2'].isna().any():
            raise ValueError('official folds exist only for glaucoma vs healthy; use --folds random '
                             'when suspects are included')
        return df['fold_test2'].astype(int).to_numpy() - 1
    from sklearn.model_selection import StratifiedGroupKFold
    fold = np.full(len(df), -1)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(cv.split(df, df['y'], df['patient_id'])):
        fold[te] = k
    return fold


def chaksu_external(labels_csv):
    """All Chaksu images (train and test splits; nothing is trained on Chaksu), MD5 duplicates dropped."""
    df = pd.read_csv(labels_csv)
    df['group'] = df['bosch_group'].fillna(df['image_id'])  # paired Bosch images share an eye
    # a duplicate's eye is the original's eye: merge their groups before dropping the duplicate
    group_of = df.set_index('image_id')['group']
    for dup, orig in df.loc[df['duplicate_of'].notna(), ['group', 'duplicate_of']].itertuples(index=False):
        df.loc[df['group'] == dup, 'group'] = group_of[orig]
    df = df[df['duplicate_of'].isna()].reset_index(drop=True)
    df['y'] = df['label_majority'].astype(int)
    df['agreement'] = df['n_glaucoma_votes'].map({0: '5-0', 5: '5-0', 1: '4-1', 4: '4-1', 2: '3-2', 3: '3-2'})
    return df


def load_features(df, feats_npz):
    z = np.load(feats_npz)
    order = pd.Series(np.arange(len(z['image_id'])), index=z['image_id'])
    return z['features'][order.loc[df['image_id']].to_numpy()]


# ---------------------------------------------------------------- metrics

METRICS = ['auroc', 'auprc', 'balanced_acc', 'sensitivity', 'specificity', 'precision', 'npv', 'f1', 'youden_j', 'mcc',
           'sens_at_90spec', 'spec_at_90sens']
SCREENING_PREVALENCE = 0.035  # glaucoma, ages 40-80 worldwide (Tham et al. 2014)


def _auroc(y, p):
    """Mann-Whitney AUROC with average ranks for ties (same value as sklearn's roc_auc_score)."""
    order = np.argsort(p, kind='mergesort')
    ps = p[order]
    ranks = np.empty(len(p))
    _, first, counts = np.unique(ps, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0  # 1-based average rank of each tie block
    ranks[order] = np.repeat(avg, counts)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _roc_points(y, p):
    """(fpr, tpr) at every distinct threshold, from 'flag nothing' to 'flag everything'."""
    order = np.argsort(-p, kind='mergesort')
    ys, ps = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp = np.r_[0, np.cumsum(ys)[last]]
    fp = np.r_[0, (last + 1) - np.cumsum(ys)[last]]
    return fp / max(len(y) - ys.sum(), 1), tp / max(ys.sum(), 1)


def _auprc(y, p):
    """Average precision (step-wise, same definition as sklearn's average_precision_score)."""
    order = np.argsort(-p, kind='mergesort')
    ys, ps = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]  # end of each block of tied scores
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / ys.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def ppv_at_prevalence(sens, spec, prev):
    """Chance that a positive call is right in a population with this prevalence (Bayes)."""
    denom = sens * prev + (1 - spec) * (1 - prev)
    return float(sens * prev / denom) if denom else float('nan')


def metrics(y, p, thr=0.5):
    """thr: one threshold, or one per image (e.g. each fold's own threshold).
    sens_at_90spec / spec_at_90sens are read from this data's ROC curve (operating points)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yhat = (p >= np.asarray(thr, dtype=float)).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    fp, fn = n_neg - tn, n_pos - tp
    sens = tp / n_pos if n_pos else 0.0
    spec = tn / n_neg if n_neg else 0.0
    f1 = 2 * tp / (n_pos + tp + fp) if (n_pos + tp + fp) else 0.0
    mcc_den = np.sqrt(float(tp + fp) * n_pos * n_neg * float(tn + fn))
    fpr, tpr = _roc_points(y, p)
    return {'n': int(len(y)), 'n_pos': n_pos, 'prevalence': n_pos / len(y) if len(y) else 0.0,
            'auroc': float(_auroc(y, p)), 'auprc': _auprc(y, p),
            'balanced_acc': (sens + spec) / 2, 'sensitivity': sens, 'specificity': spec,
            'precision': tp / (tp + fp) if (tp + fp) else 0.0, 'npv': tn / (tn + fn) if (tn + fn) else 0.0,
            'f1': f1, 'youden_j': sens + spec - 1, 'mcc': float((tp * tn - fp * fn) / mcc_den) if mcc_den else 0.0,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'sens_at_90spec': float(tpr[(1 - fpr) >= 0.9].max()),
            'spec_at_90sens': float((1 - fpr)[tpr >= 0.9].max()),
            'ppv_at_screening': ppv_at_prevalence(sens, spec, SCREENING_PREVALENCE)}


def metrics_sklearn(y, p, thr=0.5):
    """Reference implementation, used only to test metrics()."""
    yhat = (np.asarray(p) >= thr).astype(int)
    return {'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p),
            'balanced_acc': balanced_accuracy_score(y, yhat),
            'sensitivity': recall_score(y, yhat, pos_label=1, zero_division=0),
            'specificity': recall_score(y, yhat, pos_label=0, zero_division=0),
            'f1': f1_score(y, yhat, zero_division=0)}


def _cluster_indices(groups):
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    return [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]


def youden_threshold(y, p):
    """Threshold maximising sensitivity + specificity (Youden's J)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    cands = np.unique(p)
    j = [(p[y == 1] >= t).mean() + (p[y == 0] < t).mean() for t in cands]
    return float(cands[int(np.argmax(j))])


def sensitivity_threshold(y, p, target=0.9):
    """Highest threshold that still catches at least `target` of the positives (recall fixed on the PR curve)."""
    pos = np.sort(np.asarray(p, dtype=float)[np.asarray(y) == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


def choose_threshold(y, p, rule):
    return sensitivity_threshold(y, p, 0.9) if rule == 'sens90' else youden_threshold(y, p)


def per_fold_auroc(y, p, fold):
    """Mean and SD of the AUROC computed inside each test fold (immune to scale differences between folds)."""
    a = [metrics(np.asarray(y)[fold == k], np.asarray(p)[fold == k])['auroc'] for k in np.unique(fold)]
    return {'mean': float(np.mean(a)), 'sd': float(np.std(a, ddof=1)), 'per_fold': [float(x) for x in a]}


def rank_within_fold(p, fold):
    """Each fold's scores replaced by their percentile rank inside that fold. Pooling these avoids
    artefacts when folds' models output probabilities on different scales (e.g. different C).
    Tied scores share their average rank, so ties still count as ties in the AUROC."""
    p, fold, out = np.asarray(p, dtype=float), np.asarray(fold), np.zeros(len(p))
    for k in np.unique(fold):
        m = fold == k
        out[m] = (pd.Series(p[m]).rank(method='average').to_numpy() - 0.5) / m.sum()
    return out


def bootstrap_ci(y, p, groups, n_boot=2000, seed=0, thr=0.5):
    """95% CI by cluster bootstrap: resample whole groups (patients / eyes) with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    thr = np.broadcast_to(np.asarray(thr, dtype=float), y.shape)
    rng = np.random.default_rng(seed)
    members = _cluster_indices(groups)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if y[idx].min() != y[idx].max():
            stats.append(metrics(y[idx], p[idx], thr[idx]))
    return {k: [float(np.percentile([s[k] for s in stats], 2.5)),
                float(np.percentile([s[k] for s in stats], 97.5))] for k in METRICS}


def evaluate(y, p, groups, seed=0, thr=0.5):
    return metrics(y, p, thr) | {'ci95': bootstrap_ci(y, p, groups, seed=seed, thr=thr)}


def paired_bootstrap_diff(y, p_a, p_b, groups, metric='auroc', n_boot=2000, seed=0):
    """Bootstrap the difference metric(A) - metric(B) on the same resampled images.
    If the 95% interval contains 0 there is no clear winner."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    members = _cluster_indices(groups)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if y[idx].min() != y[idx].max():
            diffs.append(metrics(y[idx], p_a[idx])[metric] - metrics(y[idx], p_b[idx])[metric])
    diff = metrics(y, p_a)[metric] - metrics(y, p_b)[metric]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'metric': metric, 'diff': float(diff), 'ci95': [float(lo), float(hi)],
            'verdict': 'A better' if lo > 0 else 'B better' if hi < 0 else 'no clear winner (0 in CI)'}


def failure_split(y, p, votes, thr=0.5):
    """Dangerous = experts unanimous (0/5 or 5/5) but the model is wrong; safe = experts split."""
    y, p, votes = np.asarray(y), np.asarray(p), np.asarray(votes)
    wrong = (p >= np.asarray(thr, dtype=float)).astype(int) != y
    unanimous = np.isin(votes, [0, 5])
    by_votes = pd.Series(votes[wrong]).value_counts().sort_index()
    return {'n': int(len(y)), 'n_wrong': int(wrong.sum()), 'dangerous': int((wrong & unanimous).sum()),
            'safe': int((wrong & ~unanimous).sum()), 'wrong_by_votes': {int(k): int(v) for k, v in by_votes.items()}}


def fmt_row(name, m):
    """Main table: discrimination (AUROC, AUPRC vs its guessing floor) and the chosen operating point."""
    ci = m['ci95']
    c = lambda k: f"{m[k]:.3f} ({ci[k][0]:.2f}–{ci[k][1]:.2f})"  # noqa: E731
    extra = f" | {m['paper_median_auroc']:.3f}" if 'paper_median_auroc' in m else ''
    return (f"| {name} | {m['n']} ({m['n_pos']}) | {c('auroc')} | {c('auprc')} [{m['prevalence']:.2f}] | "
            f"{c('sensitivity')} | {c('specificity')} | {c('precision')} | {c('sens_at_90spec')}{extra} |\n")


def fmt_cm(name, m):
    """Confusion matrix at the chosen threshold, NPV, and PPV if used for screening (3.5% prevalence)."""
    return (f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['npv']:.3f} | "
            f"{m['precision']:.2f} at {m['prevalence']:.0%} → {m['ppv_at_screening']:.2f} at "
            f"{SCREENING_PREVALENCE:.1%} |\n")


TABLE_HEADER = ('| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) '
                '| Sens. at 90% spec. |\n|---|---|---|---|---|---|---|---|\n')
TABLE_HEADER_PAPER = ('| Setting | n (glaucoma) | AUROC | AUPRC [guessing] | Sensitivity | Specificity | Precision (PPV) '
                      '| Sens. at 90% spec. | Paper AUROC |\n|---|---|---|---|---|---|---|---|---|\n')
CM_HEADER = ('| Setting | TP | FP | FN (missed) | TN | NPV | PPV here → PPV in screening |\n'
             '|---|---|---|---|---|---|---|\n')
CM_NOTE = ('PPV in screening applies this sensitivity and specificity to a population where 3.5% have glaucoma '
           '(Tham et al. 2014): the same model raises far more false alarms when the disease is rarer.\n')
