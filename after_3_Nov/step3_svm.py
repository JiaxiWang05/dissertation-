"""Step 3b: tune the SVM with Optuna. Learn on training, score on validation. The test part is not used."""
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import SEED, STORAGE, load_data, part

N_TRIALS = 100
lab, X, y = load_data()
Xtr, ytr = part(lab, X, y, 'train')
Xva, yva = part(lab, X, y, 'val')


def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])     # straight or curved boundary
    C = trial.suggest_float('C', 1e-5, 1e2, log=True)                # lower end 1e-5: the first run's best C sat at 1e-3
    gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True) if kernel == 'rbf' else 'scale'
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced'))
    model.fit(Xtr, ytr)
    return roc_auc_score(yva, model.decision_function(Xva))           # SVM gives a score, enough for AUROC


try:
    optuna.delete_study(study_name='svm', storage=STORAGE)
except KeyError:
    pass
study = optuna.create_study(study_name='svm', storage=STORAGE, direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
print('best validation AUROC:', round(study.best_value, 3), '  best hyperparameters:', study.best_params)
