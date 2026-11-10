"""Shared helpers for the Chakshu frozen-feature experiments after the 10 Nov meeting (copied from ../after_3_Nov/common.py).
Changes: the MLP layers may have different widths (params["widths"]); every epoch also records the validation AUPRC;
loss "bce" (no class weight) and a scale on the class weight (params["pos_weight_scale"]) were added.
Note on "eye": Chakshu has no patient or eye ID. The column called `eye` is a GROUP: the two images of a Bosch file pair
(P1_Image1 / P1_Image2) share a group, an exact copy shares its original's group, every other image is its own group."""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

FEATURES = 'chaksu_rfg_resize.npz'        # 384 RETFound-Green features per image (frozen model)
LABELS = '../pipeline/chaksu_labels.csv'   # labels: majority vote of the 5 experts
SPLIT = 'chaksu_split.csv'                 # made by step1_make_split.py: train / val / test = 70 / 15 / 15
STORAGE = 'sqlite:///optuna.db'            # Optuna saves every trial in this file
SEED = 42


def load_labels():
    """Label table, one row per image: all 1345 images (the 7 exact copies are kept), with the eye of each image."""
    lab = pd.read_csv(LABELS)
    # The eye of each image: the two Bosch images of one eye share an eye; a copy's eye is its original's eye,
    # so a copy always sits in the same part as its original.
    lab['eye'] = lab['bosch_group'].fillna(lab['image_id'])
    eye_of = lab.set_index('image_id')['eye']
    for dup_eye, original in lab.loc[lab['duplicate_of'].notna(), ['eye', 'duplicate_of']].itertuples(index=False):
        lab.loc[lab['eye'] == dup_eye, 'eye'] = eye_of[original]
    return lab


def load_data():
    """Label table (one row per image, with its eye and part), features X (1345 x 384), labels y (1 = glaucoma)."""
    lab = load_labels()
    lab['part'] = pd.read_csv(SPLIT).set_index('image_id')['split'].loc[lab['image_id']].to_numpy()
    z = np.load(FEATURES)
    X = pd.DataFrame(z['features'], index=z['image_id']).loc[lab['image_id']].to_numpy()
    y = lab['label_majority'].to_numpy()
    return lab, X, y


def part(lab, X, y, name):
    """Features and labels of one part: 'train', 'val' or 'test'."""
    m = (lab['part'] == name).to_numpy()
    return X[m], y[m]


def cv_split(lab, n_splits=5):
    """The 5 folds of training + validation (never test). The images of one eye stay in one fold;
    each fold has about the same share of glaucoma per camera (as in step1_make_split.py).
    Returns the mask of training + validation rows and, per fold, (train, val) row numbers within those rows."""
    keep = lab['part'].isin(['train', 'val']).to_numpy()
    sub = lab[keep]
    strata = sub['label_majority'].astype(str) + '_' + sub['camera']
    eyes = sub['eye'].to_numpy()
    splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(sub, strata, eyes))
    for tr, va in splits:
        assert not set(eyes[tr]) & set(eyes[va])        # no eye on both sides
    return keep, splits


def cv_folds(lab, X, y, n_splits=5):
    """Per fold of cv_split: (X_train, y_train, X_val, y_val), scaled by a StandardScaler fitted on that fold's training rows."""
    keep, splits = cv_split(lab, n_splits)
    X, y = X[keep], y[keep]
    folds = []
    for tr, va in splits:
        scaler = StandardScaler().fit(X[tr])
        folds.append((scaler.transform(X[tr]), y[tr], scaler.transform(X[va]), y[va]))
    return folds


# ---------------------------------------------------------------- MLP

def make_mlp(widths, dropout, n_inputs=384):
    """One hidden layer per entry of `widths` (ReLU + dropout), then one output: the glaucoma score.
    widths=[128, 64] = two hidden layers of 128 and 64 neurons."""
    layers, d = [], n_inputs
    for width in widths:
        layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


OPTIMIZERS = {'adam': torch.optim.Adam,              # weight_decay = L2 penalty added to the gradient
              'adamw': torch.optim.AdamW,            # weight_decay applied directly to the weights ("decoupled")
              'rmsprop': torch.optim.RMSprop,
              'sgd': lambda p, lr, weight_decay: torch.optim.SGD(p, lr=lr, weight_decay=weight_decay,
                                                                 momentum=0.9, nesterov=True)}


def make_loss(params, pos_weight):
    """Weighted binary cross-entropy (a glaucoma image counts pos_weight times), or focal loss:
    the same, but images the model already gets right with confidence count less, so training focuses on hard ones."""
    if params.get('loss') == 'bce':                       # plain cross-entropy: no extra weight on glaucoma
        pos_weight = torch.ones(1)
    else:                                                 # weighted: glaucoma counts (healthy / glaucoma) x scale times
        pos_weight = pos_weight * params.get('pos_weight_scale', 1.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    gamma = params.get('focal_gamma', 0.0) if params.get('loss') == 'focal' else 0.0

    def loss_fn(logits, y):
        per_image = bce(logits, y)
        if gamma:
            z_right = torch.where(y == 1, logits, -logits)   # large when the model is right and confident
            # (1 - p_right) ** gamma: easy images (p_right near 1) count less. Written with logsigmoid because
            # (1 - p) ** gamma has an infinite gradient at p = 1 when gamma < 1, which turned the weights into NaN.
            per_image = per_image * torch.exp(gamma * F.logsigmoid(-z_right))
        return per_image.mean()
    return loss_fn


def train_mlp(params, Xtr, ytr, Xva=None, yva=None, max_epochs=300, patience=30, seed=SEED):
    """Train on the training part; after every epoch measure the loss on training and validation.
    Keep the weights of the epoch with the lowest validation loss; stop when it has not improved for `patience` epochs.
    params: n_layers, width, dropout, lr, weight_decay, and optionally optimizer, l1, batch_size, loss, focal_gamma,
    lr_schedule ('constant' or 'plateau': halve the learning rate after 11 epochs without a lower validation loss),
    noise (standard deviation of Gaussian noise added to every training batch; the features are standardised, so
    0.3 = 30 % of one standard deviation; validation is never changed).
    Without a validation part (Xva=None; the final fit on training + validation): train exactly max_epochs and keep
    the last weights; a plateau schedule then follows the training loss."""
    torch.manual_seed(seed)
    noise = params.get('noise', 0.0)
    noise_gen = torch.Generator().manual_seed(seed)   # own random numbers: batch order and dropout stay the same as without noise
    widths = params.get('widths') or [params['width']] * params['n_layers']
    model = make_mlp(widths, params['dropout'], Xtr.shape[1])
    opt = OPTIMIZERS[params.get('optimizer', 'adam')](model.parameters(), lr=params['lr'],
                                                       weight_decay=params['weight_decay'])
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
             if params.get('lr_schedule') == 'plateau' else None)
    l1, batch_size = params.get('l1', 0.0), params.get('batch_size', 64)
    weights = [p for name, p in model.named_parameters() if name.endswith('weight')]   # L1 acts on weights, not biases
    pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32)  # about 6: glaucoma is rare
    loss_fn = make_loss(params, pos_weight)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    if Xva is not None:
        Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    history, best_loss, best_state, best_epoch, wait = [], float('inf'), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()                                   # training mode: dropout on
        order = torch.randperm(len(Xt))
        for start in range(0, len(Xt), batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            xb = Xt[idx]
            if noise:                                   # a new random version of each training image's features every time
                xb = xb + noise * torch.randn(xb.shape, generator=noise_gen)
            loss = loss_fn(model(xb).squeeze(1), yt[idx])
            if l1:
                loss = loss + l1 * sum(w.abs().sum() for w in weights)   # L1 penalty: pushes small weights to 0
            loss.backward()
            opt.step()
        model.eval()                                    # evaluation mode: dropout off
        with torch.no_grad():
            train_loss = loss_fn(model(Xt).squeeze(1), yt).item()
            if Xva is not None:
                val_score = model(Xv).squeeze(1)
                val_loss = loss_fn(val_score, yv).item()
        if Xva is None:                                 # final fit: nothing to stop on, train all max_epochs
            history.append({'epoch': epoch, 'train_loss': train_loss, 'lr': opt.param_groups[0]['lr']})
            if sched:
                sched.step(train_loss)
            continue
        if not torch.isfinite(val_score).all():         # training blew up (e.g. learning rate far too high): stop here
            break
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'val_auroc': roc_auc_score(yva, val_score.numpy()),
                        'val_auprc': average_precision_score(yva, val_score.numpy()), 'lr': opt.param_groups[0]['lr']})
        if sched:
            sched.step(val_loss)                        # smaller steps once the validation loss stops falling
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if Xva is None:
        return model, pd.DataFrame(history), max_epochs
    if best_state is not None:                          # None only if it blew up in the first epoch (best_epoch = 0)
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def mlp_scores(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- thresholds and results

def threshold_at_recall(y, scores, recall=0.9):
    """Highest threshold that still finds at least `recall` of the glaucoma images."""
    pos = np.sort(scores[y == 1])[::-1]
    return float(pos[int(np.ceil(recall * len(pos))) - 1])


def youden_threshold(y, scores):
    """Threshold with the largest sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y, scores)
    return float(thr[np.argmax(tpr - fpr)])


def at_threshold(y, scores, thr):
    """Confusion matrix and rates when every image with score >= thr is called glaucoma."""
    pred = (scores >= thr).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    fp, tn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn, 'sensitivity': sens, 'specificity': spec,
            'PPV': tp / (tp + fp) if tp + fp else float('nan'), 'NPV': tn / (tn + fn) if tn + fn else float('nan'),
            'balanced_accuracy': (sens + spec) / 2, 'MCC': matthews_corrcoef(y, pred)}


def bootstrap_auroc_ci(y, scores, eyes, n_boot=2000, seed=SEED):
    """95% CI of the AUROC: resample whole groups (see `eye` above) with replacement 2000 times."""
    rng = np.random.default_rng(seed)
    unique = np.unique(eyes)
    rows = {e: np.flatnonzero(eyes == e) for e in unique}
    values = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[e] for e in rng.choice(unique, len(unique))])
        if y[pick].min() != y[pick].max():
            values.append(roc_auc_score(y[pick], scores[pick]))
    return np.percentile(values, [2.5, 97.5])


def paired_auroc_difference(y, s1, s2, eyes, n_boot=2000, seed=SEED):
    """AUROC of s1 minus AUROC of s2 on the same images, with a 95% CI: resample whole eyes 2000 times."""
    rng = np.random.default_rng(seed)
    unique = np.unique(eyes)
    rows = {e: np.flatnonzero(eyes == e) for e in unique}
    values = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[e] for e in rng.choice(unique, len(unique))])
        if y[pick].min() != y[pick].max():
            values.append(roc_auc_score(y[pick], s1[pick]) - roc_auc_score(y[pick], s2[pick]))
    return roc_auc_score(y, s1) - roc_auc_score(y, s2), np.percentile(values, [2.5, 97.5])


def bootstrap_ci(metric, y, scores, groups, n_boot=2000, seed=SEED):
    """95% CI of any metric(y, scores): resample whole groups with replacement 2000 times."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[g] for g in rng.choice(unique, len(unique))])
        if y[pick].min() != y[pick].max():
            values.append(metric(y[pick], scores[pick]))
    return np.percentile(values, [2.5, 97.5])


def paired_difference(metric, y, s1, s2, groups, n_boot=2000, seed=SEED):
    """metric of s1 minus metric of s2 on the same images, with a 95% CI from 2000 group resamples."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[g] for g in rng.choice(unique, len(unique))])
        if y[pick].min() != y[pick].max():
            values.append(metric(y[pick], s1[pick]) - metric(y[pick], s2[pick]))
    return metric(y, s1) - metric(y, s2), np.percentile(values, [2.5, 97.5])


def mlp_small_storage():
    """The small-MLP search runs as 8 parallel workers (Hamilton job array), so it uses an Optuna journal file,
    which several processes can append to safely; the other studies stay in optuna.db."""
    import optuna
    from optuna.storages.journal import JournalFileBackend
    return optuna.storages.JournalStorage(JournalFileBackend('optuna_mlp_small.log'))
