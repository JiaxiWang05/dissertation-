"""Step 3d: tune k-nearest neighbours. Learn on training, score on validation. The test part is not used.
The grid is small (50 values of k x 2 weightings x 2 distances = 200 settings), so every setting is tried."""
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import SEED, STORAGE, load_data, part

GRID = {'k': list(range(1, 100, 2)),              # odd k from 1 to 99: how many neighbours vote
        'weights': ['uniform', 'distance'],        # every neighbour one vote, or nearer neighbours count more
        'metric': ['euclidean', 'cosine']}         # straight-line distance, or the angle between two feature lists
lab, X, y = load_data()
Xtr, ytr = part(lab, X, y, 'train')
Xva, yva = part(lab, X, y, 'val')


def objective(trial):
    model = make_pipeline(StandardScaler(),        # distances need every feature on the same scale
                          KNeighborsClassifier(n_neighbors=trial.suggest_categorical('k', GRID['k']),
                                               weights=trial.suggest_categorical('weights', GRID['weights']),
                                               metric=trial.suggest_categorical('metric', GRID['metric'])))
    model.fit(Xtr, ytr)                                                # "fit" only stores the 802 training images
    return roc_auc_score(yva, model.predict_proba(Xva)[:, 1])          # score = share of the k neighbours with glaucoma


try:
    optuna.delete_study(study_name='knn', storage=STORAGE)
except KeyError:
    pass
study = optuna.create_study(study_name='knn', storage=STORAGE, direction='maximize',
                            sampler=optuna.samplers.GridSampler(GRID, seed=SEED))   # grid search: every setting once
study.optimize(objective, n_trials=len(GRID['k']) * len(GRID['weights']) * len(GRID['metric']))
print('best validation AUROC:', round(study.best_value, 3), '  best hyperparameters:', study.best_params)
