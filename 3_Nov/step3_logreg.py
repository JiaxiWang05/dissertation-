"""Step 3a: tune logistic regression with Optuna. Learn on training, score on validation. The test part is not used."""
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import SEED, STORAGE, load_data, part

N_TRIALS = 50
lab, X, y = load_data()
Xtr, ytr = part(lab, X, y, 'train')
Xva, yva = part(lab, X, y, 'val')


def objective(trial):
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)                 # Optuna picks a C to try
    model = make_pipeline(StandardScaler(),                           # scaler is fitted on training only
                          LogisticRegression(C=C, class_weight='balanced', max_iter=5000))
    model.fit(Xtr, ytr)
    return roc_auc_score(yva, model.predict_proba(Xva)[:, 1])          # validation AUROC = the score Optuna maximises


try:
    optuna.delete_study(study_name='logreg', storage=STORAGE)         # start fresh if you run this again
except KeyError:
    pass
study = optuna.create_study(study_name='logreg', storage=STORAGE, direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
print('best validation AUROC:', round(study.best_value, 3), '  best hyperparameters:', study.best_params)
