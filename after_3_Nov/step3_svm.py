"""Step 3b: tune the SVM with Optuna. The test part is not used.
Each setting is scored by 5-fold cross-validation on training + validation (the same folds as step3_mlp.py)."""
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from common import SEED, STORAGE, cv_folds, load_data

N_TRIALS = 100
lab, X, y = load_data()
folds = cv_folds(lab, X, y)                      # 5 x (training, validation); each scaled on its own training rows


def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])     # straight or curved boundary
    C = trial.suggest_float('C', 1e-5, 1e2, log=True)                # lower end 1e-5: the first run's best C sat at 1e-3
    gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True) if kernel == 'rbf' else 'scale'
    aurocs = []
    for Xtr, ytr, Xva, yva in folds:
        model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced').fit(Xtr, ytr)
        aurocs.append(roc_auc_score(yva, model.decision_function(Xva)))   # SVM gives a score, enough for AUROC
    trial.set_user_attr('fold_auroc', aurocs)
    return float(np.mean(aurocs))


try:
    optuna.delete_study(study_name='svm', storage=STORAGE)
except KeyError:
    pass
study = optuna.create_study(study_name='svm', storage=STORAGE, direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
print('best 5-fold AUROC:', round(study.best_value, 3), '  folds:', [round(a, 3) for a in study.best_trial.user_attrs['fold_auroc']],
      '  best hyperparameters:', study.best_params)
