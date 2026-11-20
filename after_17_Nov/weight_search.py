"""17 Nov task: is the class weight 6.1 too strong? (Chakshu, frozen RETFound-Green features; test set not used.)
Rahele: with weight 6.1 the model "only looks at glaucoma" (many healthy images called glaucoma, F1 about 0.39);
try smaller weights and let Optuna choose the weight with F1 or AUPRC.
Part 1 (python weight_search.py grid): fixed classifiers, weight w on the glaucoma class in
  w = 0.25, 0.5, 1, 2, 3, 4, 6.1 (= healthy / glaucoma), 8, 12.3; logistic regression (C 0.00725) and the small MLP
  (32-64 neurons, 10 Nov setting); 5-fold out-of-fold (MLP: 5 seeds). Reported: AUPRC, AUROC, and at the default
  threshold 0.5: sensitivity, specificity, F1; plus the best F1 over all thresholds.
Part 2 (WORKER=1 python weight_search.py optuna, job array): small-MLP search (same space as 10 Nov) with the
  weight scale 0.04-2 x (healthy / glaucoma), i.e. weight 0.25-12, and the objective = mean F1 at threshold 0.5."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, '../after_10_Nov')
from common import SEED, cv_folds, load_data, mlp_scores, train_mlp  # noqa: E402

lab, X, y = load_data()
folds = cv_folds(lab, X, y)
RATIO = 785 / 128                                         # healthy / glaucoma in a training fold, about 6.1
WEIGHTS = [0.25, 0.5, 1, 2, 3, 4, RATIO, 8, 2 * RATIO]
MLP = {'n_layers': 2, 'widths': [32, 64], 'dropout': 0.07248756818173137, 'optimizer': 'adamw',
       'lr': 0.00013648454900247372, 'lr_schedule': 'plateau', 'weight_decay': 0.0025904964212622532,
       'l1': 0.00697097574389041, 'batch_size': 32, 'loss': 'weighted_bce'}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def metrics(yv, p):
    """p = probability of glaucoma. Threshold 0.5 metrics plus threshold-free ones."""
    pred = (p >= 0.5).astype(int)
    tp, fn = ((pred == 1) & (yv == 1)).sum(), ((pred == 0) & (yv == 1)).sum()
    fp, tn = ((pred == 1) & (yv == 0)).sum(), ((pred == 0) & (yv == 0)).sum()
    pr, rc, _ = precision_recall_curve(yv, p)
    f1s = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return {'AUPRC': average_precision_score(yv, p), 'AUROC': roc_auc_score(yv, p),
            'sens@0.5': tp / (tp + fn), 'spec@0.5': tn / (tn + fp), 'F1@0.5': f1_score(yv, pred, zero_division=0),
            'best F1': f1s.max()}


def grid():
    rows = []
    for w in WEIGHTS:
        oof = np.zeros(int(lab['part'].isin(['train', 'val']).sum()))
        yy = np.zeros_like(oof)
        start = 0
        for Xtr, ytr, Xva, yva in folds:
            m = LogisticRegression(C=0.007250175039761133, class_weight={0: 1, 1: w}, max_iter=5000).fit(Xtr, ytr)
            oof[start:start + len(yva)], yy[start:start + len(yva)] = m.predict_proba(Xva)[:, 1], yva
            start += len(yva)
        rows.append({'model': 'logistic regression', 'weight': w, 'seed': 0, **metrics(yy, oof)})
        for seed in range(5):
            ps, ys = [], []
            for Xtr, ytr, Xva, yva in folds:
                params = {**MLP, 'pos_weight_scale': w / RATIO}
                model, _, _ = train_mlp(params, Xtr, ytr, Xva, yva, seed=seed)
                ps.append(sigmoid(mlp_scores(model, Xva))); ys.append(yva)
            rows.append({'model': 'small MLP', 'weight': w, 'seed': seed, **metrics(np.concatenate(ys), np.concatenate(ps))})
        print(f'weight {w:.2f} done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('weight_grid.csv', index=False)
    s = df.groupby(['model', 'weight']).mean(numeric_only=True).drop(columns='seed')
    print(s.round(3).to_string())


def optuna_search():
    import optuna
    from optuna.storages.journal import JournalFileBackend
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    storage = optuna.storages.JournalStorage(JournalFileBackend('optuna_weight_f1.log'))
    seed = SEED + int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    study = optuna.create_study(study_name='mlp_small_f1', storage=storage, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial):
        params = {'n_layers': 2,
                  'widths': [trial.suggest_categorical('width1', [32, 64, 96, 128]),
                             trial.suggest_categorical('width2', [32, 64, 96, 128])],
                  'dropout': trial.suggest_float('dropout', 0.0, 0.7),
                  'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop']),
                  'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
                  'lr_schedule': trial.suggest_categorical('lr_schedule', ['constant', 'plateau']),
                  'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True),
                  'l1': trial.suggest_float('l1', 1e-8, 1e-2, log=True),
                  'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                  'loss': 'weighted_bce',
                  'pos_weight_scale': trial.suggest_float('pos_weight_scale', 0.04, 2.0, log=True)}
        f1, ap = [], []
        for Xtr, ytr, Xva, yva in folds:
            model, _, best = train_mlp(params, Xtr, ytr, Xva, yva)
            if best == 0:
                return 0.0
            m = metrics(yva, sigmoid(mlp_scores(model, Xva)))
            f1.append(m['F1@0.5']); ap.append(m['AUPRC'])
        trial.set_user_attr('mean_auprc', float(np.mean(ap)))
        trial.set_user_attr('weight', params['pos_weight_scale'] * RATIO)
        return float(np.mean(f1))

    if os.environ.get('WORKER'):
        study.optimize(objective, callbacks=[MaxTrialsCallback(200, states=(TrialState.COMPLETE,))])
    b = study.best_trial
    print('trials:', len(study.trials), '| best mean F1 at 0.5:', round(b.value, 3), '| AUPRC', round(b.user_attrs['mean_auprc'], 3),
          '| weight', round(b.user_attrs['weight'], 2), '|', b.params)


grid() if sys.argv[1] == 'grid' else optuna_search()
