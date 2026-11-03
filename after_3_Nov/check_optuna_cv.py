"""Check for Rahele: 5-fold cross-validation inside Optuna. Which hyperparameters does Optuna give back?
Real data, training + validation parts only. The test part is not used. Nothing is saved.
Run: cd ~/Desktop/dissertation/after_3_Nov && ~/.venvs/rfg/bin/python -B check_optuna_cv.py"""
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import SEED, load_data

N_TRIALS = 20
optuna.logging.set_verbosity(optuna.logging.WARNING)          # no log line per trial

lab, X, y = load_data()
keep = lab['part'].isin(['train', 'val']).to_numpy()          # training + validation only, never test
X, y, eyes = X[keep], y[keep], lab.loc[keep, 'eye'].to_numpy()
print('images used:', len(y), '  glaucoma:', int(y.sum()), '  eyes:', len(set(eyes)))

# 5 folds; the images of one eye stay together; each fold has about the same share of glaucoma
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups=eyes))
seen = set()
for k, (tr, va) in enumerate(folds):
    both = len(set(eyes[tr]) & set(eyes[va]))                 # eyes on both sides of this fold
    again = len(seen & set(eyes[va]))                         # eyes already in an earlier validation fold
    assert both == 0 and again == 0
    seen |= set(eyes[va])
    print(f'fold {k}: validation {len(va)} images ({int(y[va].sum())} glaucoma), training {len(tr)};'
          f' eyes on both sides {both}, eyes in an earlier validation fold {again}')
assert sum(len(va) for _, va in folds) == len(y)              # every image is validated exactly once


def fold_aurocs(C):
    """Validation AUROC of each fold for one C: fit on the other 4 folds, score this fold."""
    out = []
    for tr, va in folds:
        model = make_pipeline(StandardScaler(),                # scaler fitted on this fold's training rows only
                              LogisticRegression(C=C, class_weight='balanced', max_iter=5000))
        model.fit(X[tr], y[tr])
        out.append(roc_auc_score(y[va], model.predict_proba(X[va])[:, 1]))
    return out


def objective(trial):
    C = trial.suggest_float('C', 1e-5, 1e2, log=True)         # Optuna picks ONE C for this trial
    assert trial.suggest_float('C', 1e-5, 1e2, log=True) == C  # asking again gives the same C
    aurocs = fold_aurocs(C)                                    # the same C in all 5 folds
    trial.set_user_attr('fold_aurocs', aurocs)                 # kept only so we can print them
    return float(np.mean(aurocs))                              # Optuna sees only this number: the mean


def objective_last_fold(trial):
    """The classic bug: the same 5 fits, but only the last fold's AUROC is returned."""
    C = trial.suggest_float('C', 1e-5, 1e2, log=True)
    aurocs = fold_aurocs(C)
    trial.set_user_attr('fold_aurocs', aurocs)
    return aurocs[-1]                                          # BUG: like `return auroc` after the loop


def run(obj):
    study = optuna.create_study(direction='maximize',          # no storage: in memory, no file written
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=N_TRIALS)
    print('\ntrial         C  fold0  fold1  fold2  fold3  fold4    mean  returned')
    for t in study.trials:
        a = t.user_attrs['fold_aurocs']
        print(f'{t.number:5d} {t.params["C"]:9.3e}  ' + '  '.join(f'{v:.3f}' for v in a) +
              f'  {np.mean(a):.4f}  {t.value:.4f}')
    return study


print('\n=== 1. Correct: the objective returns the MEAN of the 5 folds ===')
study = run(objective)
by_hand = max(study.trials, key=lambda t: np.mean(t.user_attrs['fold_aurocs']))
print('Optuna best: trial', study.best_trial.number, study.best_params, 'mean AUROC', round(study.best_value, 4))
print('by hand:     trial', by_hand.number, 'has the highest mean', round(np.mean(by_hand.user_attrs['fold_aurocs']), 4))
assert study.best_trial.number == by_hand.number
print('Optuna returned ONE hyperparameter set, the one with the best 5-fold mean (not one set per fold).')

print('\n=== 2. Bug demo: the objective returns only the LAST fold ===')
bug = run(objective_last_fold)
b = bug.best_trial
print('picks trial', b.number, bug.best_params, 'last-fold AUROC', round(b.value, 4),
      'but its 5-fold mean is', round(np.mean(b.user_attrs['fold_aurocs']), 4))
print('correct study: C =', f'{study.best_params["C"]:.3g}', ' bug study: C =', f'{bug.best_params["C"]:.3g}')
