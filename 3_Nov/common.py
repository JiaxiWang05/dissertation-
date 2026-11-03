"""Shared helpers for the Chakshu frozen-feature experiments. Every step script imports from here."""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

FEATURES = 'chaksu_rfg_resize.npz'        # 384 RETFound-Green features per image (frozen model)
LABELS = '../pipeline/chaksu_labels.csv'   # labels: majority vote of the 5 experts
SPLIT = 'chaksu_split.csv'                 # your saved split: train / val / test
STORAGE = 'sqlite:///optuna.db'            # Optuna saves every trial in this file
SEED = 42


def load_data():
    """Label table (one row per image, with its eye and part), features X (1338 x 384), labels y (1 = glaucoma)."""
    lab = pd.read_csv(LABELS)
    # The eye of each image: the two Bosch images of one eye share an eye; a duplicate's eye is its original's eye.
    lab['eye'] = lab['bosch_group'].fillna(lab['image_id'])
    eye_of = lab.set_index('image_id')['eye']
    for dup_eye, original in lab.loc[lab['duplicate_of'].notna(), ['eye', 'duplicate_of']].itertuples(index=False):
        lab.loc[lab['eye'] == dup_eye, 'eye'] = eye_of[original]
    lab = lab[lab['duplicate_of'].isna()].reset_index(drop=True)      # drop the 7 duplicate images: 1338 left
    lab['part'] = pd.read_csv(SPLIT).set_index('image_id')['split'].loc[lab['image_id']].to_numpy()
    z = np.load(FEATURES)
    X = pd.DataFrame(z['features'], index=z['image_id']).loc[lab['image_id']].to_numpy()
    y = lab['label_majority'].to_numpy()
    return lab, X, y


def part(lab, X, y, name):
    """Features and labels of one part: 'train', 'val' or 'test'."""
    m = (lab['part'] == name).to_numpy()
    return X[m], y[m]


# ---------------------------------------------------------------- MLP

def make_mlp(n_layers, width, dropout, n_inputs=384):
    """n_layers hidden layers of `width` neurons (ReLU + dropout), then one output: the glaucoma score."""
    layers, d = [], n_inputs
    for _ in range(n_layers):
        layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def train_mlp(params, Xtr, ytr, Xva, yva, max_epochs=200, patience=20, batch_size=64, seed=SEED):
    """Train on the training part; after every epoch measure the loss on training and validation.
    Keep the weights of the epoch with the lowest validation loss; stop when it has not improved for `patience` epochs."""
    torch.manual_seed(seed)
    model = make_mlp(params['n_layers'], params['width'], params['dropout'], Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32)  # about 6: glaucoma is rare
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            loss.backward()
            opt.step()
        model.eval()                                    # evaluation mode: dropout off
        with torch.no_grad():
            train_loss = loss_fn(model(Xt).squeeze(1), yt).item()
            val_score = model(Xv).squeeze(1)
            val_loss = loss_fn(val_score, yv).item()
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'val_auroc': roc_auc_score(yva, val_score.numpy())})
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
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
