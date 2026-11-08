"""Step 3d: tune k-nearest neighbours. The test part is not used.
Each setting is scored by 5-fold cross-validation on training + validation (the same folds as step3_mlp.py).
The grid is small (50 values of k x 2 weightings x 2 distances = 200 settings), so every setting is tried."""
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from common import SEED, STORAGE, cv_folds, load_data

GRID = {'k': list(range(1, 100, 2)),              # odd k from 1 to 99: how many neighbours vote
        'weights': ['uniform', 'distance'],        # every neighbour one vote, or nearer neighbours count more
        'metric': ['euclidean', 'cosine']}         # straight-line distance, or the angle between two feature lists
lab, X, y = load_data()
folds = cv_folds(lab, X, y)                        # scaled per fold: distances need every feature on the same scale


def objective(trial):
    k = trial.suggest_categorical('k', GRID['k'])
    weights = trial.suggest_categorical('weights', GRID['weights'])
    metric = trial.suggest_categorical('metric', GRID['metric'])
    aurocs = []
    for Xtr, ytr, Xva, yva in folds:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric).fit(Xtr, ytr)   # "fit" only stores the images
        aurocs.append(roc_auc_score(yva, model.predict_proba(Xva)[:, 1]))   # score = share of the k neighbours with glaucoma
    trial.set_user_attr('fold_auroc', aurocs)
    return float(np.mean(aurocs))


try:
    optuna.delete_study(study_name='knn', storage=STORAGE)
except KeyError:
    pass
study = optuna.create_study(study_name='knn', storage=STORAGE, direction='maximize',
                            sampler=optuna.samplers.GridSampler(GRID, seed=SEED))   # grid search: every setting once
study.optimize(objective, n_trials=len(GRID['k']) * len(GRID['weights']) * len(GRID['metric']))
print('best 5-fold AUROC:', round(study.best_value, 3), '  folds:', [round(a, 3) for a in study.best_trial.user_attrs['fold_auroc']],
      '  best hyperparameters:', study.best_params)
