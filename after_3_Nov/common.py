"""Shared helpers for the Chakshu frozen-feature experiments. Every step script imports from here."""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve
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


def cv_folds(lab, X, y, n_splits=5):
    """5-fold cross-validation on training + validation (never test). The images of one eye stay in one fold;
    each fold has about the same share of glaucoma per camera (as in step1_make_split.py).
    Returns, per fold, (X_train, y_train, X_val, y_val), scaled by a StandardScaler fitted on that fold's training rows."""
    keep = lab['part'].isin(['train', 'val']).to_numpy()
    X, y, sub = X[keep], y[keep], lab[keep]
    strata = sub['label_majority'].astype(str) + '_' + sub['camera']
    eyes = sub['eye'].to_numpy()
    folds = []
    for tr, va in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(X, strata, eyes):
        assert not set(eyes[tr]) & set(eyes[va])        # no eye on both sides
        scaler = StandardScaler().fit(X[tr])
        folds.append((scaler.transform(X[tr]), y[tr], scaler.transform(X[va]), y[va]))
    return folds


# ---------------------------------------------------------------- MLP

def make_mlp(n_layers, width, dropout, n_inputs=384):
    """n_layers hidden layers of `width` neurons (ReLU + dropout), then one output: the glaucoma score."""
    layers, d = [], n_inputs
    for _ in range(n_layers):
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


def train_mlp(params, Xtr, ytr, Xva, yva, max_epochs=300, patience=30, seed=SEED):
    """Train on the training part; after every epoch measure the loss on training and validation.
    Keep the weights of the epoch with the lowest validation loss; stop when it has not improved for `patience` epochs.
    params: n_layers, width, dropout, lr, weight_decay, and optionally optimizer, l1, batch_size, loss, focal_gamma,
    lr_schedule ('constant' or 'plateau': halve the learning rate after 11 epochs without a lower validation loss)."""
    torch.manual_seed(seed)
    model = make_mlp(params['n_layers'], params['width'], params['dropout'], Xtr.shape[1])
    opt = OPTIMIZERS[params.get('optimizer', 'adam')](model.parameters(), lr=params['lr'],
                                                       weight_decay=params['weight_decay'])
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
             if params.get('lr_schedule') == 'plateau' else None)
    l1, batch_size = params.get('l1', 0.0), params.get('batch_size', 64)
    weights = [p for name, p in model.named_parameters() if name.endswith('weight')]   # L1 acts on weights, not biases
    pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32)  # about 6: glaucoma is rare
    loss_fn = make_loss(params, pos_weight)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    history, best_loss, best_state, best_epoch, wait = [], float('inf'), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()                                   # training mode: dropout on
        order = torch.randperm(len(Xt))
        for start in range(0, len(Xt), batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]).squeeze(1), yt[idx])
            if l1:
                loss = loss + l1 * sum(w.abs().sum() for w in weights)   # L1 penalty: pushes small weights to 0
            loss.backward()
            opt.step()
        model.eval()                                    # evaluation mode: dropout off
        with torch.no_grad():
            train_loss = loss_fn(model(Xt).squeeze(1), yt).item()
            val_score = model(Xv).squeeze(1)
            val_loss = loss_fn(val_score, yv).item()
        if not torch.isfinite(val_score).all():         # training blew up (e.g. learning rate far too high): stop here
            break
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'val_auroc': roc_auc_score(yva, val_score.numpy()), 'lr': opt.param_groups[0]['lr']})
        if sched:
            sched.step(val_loss)                        # smaller steps once the validation loss stops falling
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
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
    """95% CI of the AUROC: resample whole eyes with replacement 2000 times."""
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
