"""Step 3a: tune logistic regression with Optuna. The test part is not used.
Each C is scored by 5-fold cross-validation on training + validation (the same folds as step3_mlp.py),
not on the one validation part."""
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import SEED, STORAGE, cv_folds, load_data

N_TRIALS = 50
lab, X, y = load_data()
folds = cv_folds(lab, X, y)                      # 5 x (training, validation); each scaled on its own training rows


def objective(trial):
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)                 # Optuna picks a C to try
    aurocs = []
    for Xtr, ytr, Xva, yva in folds:                                  # the SAME C in all 5 folds
        model = LogisticRegression(C=C, class_weight='balanced', max_iter=5000).fit(Xtr, ytr)
        aurocs.append(roc_auc_score(yva, model.predict_proba(Xva)[:, 1]))
    trial.set_user_attr('fold_auroc', aurocs)
    return float(np.mean(aurocs))                                     # Optuna sees ONE number: the mean over the 5 folds


try:
    optuna.delete_study(study_name='logreg', storage=STORAGE)         # start fresh if you run this again
except KeyError:
    pass
study = optuna.create_study(study_name='logreg', storage=STORAGE, direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
print('best 5-fold AUROC:', round(study.best_value, 3), '  folds:', [round(a, 3) for a in study.best_trial.user_attrs['fold_auroc']],
      '  best hyperparameters:', study.best_params)
