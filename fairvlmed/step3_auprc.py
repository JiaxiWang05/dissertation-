"""FairVLMed copy of ../after_10_Nov/step3_auprc.py (same search spaces and objective). The test part is not used.
Rahele (10 Nov): with 14 % glaucoma, AUROC looks good the way accuracy does; AUPRC (area under the precision-recall
curve) shows the rare class better. Also: make the MLP smaller (2 hidden layers of 32-128 neurons, widths may differ).

Run one classifier:  python step3_auprc.py logreg | svm | knn | mlp_small
Every trial is scored by 5-fold cross-validation on training + validation (the same folds as ../after_3_Nov):
one number per trial = the mean of the 5 validation AUPRCs, so Optuna returns ONE set of hyperparameters."""
import os
import sys

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from common import SEED, cv_folds, load_data, make_mlp, mlp_small_storage, n_trainable, train_mlp

STORAGE = f'sqlite:///optuna_{os.environ.get("PREPROC", "stretch")}.db'                  # this folder's own study file (../after_3_Nov/optuna.db is untouched)
NAME = sys.argv[1]
N_TRIALS = int(os.environ.get('N_TRIALS', {'logreg': 50, 'svm': 100, 'knn': 200, 'mlp_small': 300}[NAME]))
WINDOW = 5                                       # MLP: score a fold by its validation AUPRC averaged over kept epoch +- 5
lab, X, y = load_data()
folds = cv_folds(lab, X, y)                      # 5 x (training, validation), each scaled on its own training rows
KNN_GRID = {'k': list(range(1, 100, 2)), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'cosine']}


def sklearn_objective(trial):
    if NAME == 'logreg':
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        make = lambda: LogisticRegression(C=C, class_weight='balanced', max_iter=5000)
    elif NAME == 'svm':
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        C = trial.suggest_float('C', 1e-5, 1e2, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True) if kernel == 'rbf' else 'scale'
        make = lambda: SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced')
    else:
        k = trial.suggest_categorical('k', KNN_GRID['k'])
        w = trial.suggest_categorical('weights', KNN_GRID['weights'])
        m = trial.suggest_categorical('metric', KNN_GRID['metric'])
        make = lambda: KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
    auprc, auroc = [], []
    for Xtr, ytr, Xva, yva in folds:                                   # the SAME setting in all 5 folds
        model = make().fit(Xtr, ytr)
        s = model.decision_function(Xva) if NAME == 'svm' else model.predict_proba(Xva)[:, 1]
        auprc.append(average_precision_score(yva, s))
        auroc.append(roc_auc_score(yva, s))
    trial.set_user_attr('fold_auprc', auprc)
    trial.set_user_attr('mean_auroc', float(np.mean(auroc)))
    return float(np.mean(auprc))


def mlp_objective(trial):
    params = {'n_layers': 2,                                            # Rahele: two hidden layers are fine
              'widths': [trial.suggest_categorical('width1', [32, 64, 96, 128]),   # was 32-512 and equal widths;
                         trial.suggest_categorical('width2', [32, 64, 96, 128])],  # now small, and each layer its own
              'dropout': trial.suggest_float('dropout', 0.0, 0.7),
              'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop']),
              'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
              'lr_schedule': trial.suggest_categorical('lr_schedule', ['constant', 'plateau']),
              'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True),   # L2
              'l1': trial.suggest_float('l1', 1e-8, 1e-2, log=True),
              'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
              # loss functions to compare: plain cross-entropy, weighted cross-entropy, focal (weighted)
              'loss': trial.suggest_categorical('loss', ['bce', 'weighted_bce', 'focal'])}
    if params['loss'] != 'bce':                   # weight on glaucoma = (healthy / glaucoma, about 6.1) x scale
        params['pos_weight_scale'] = trial.suggest_float('pos_weight_scale', 0.5, 2.0, log=True)
    if params['loss'] == 'focal':
        params['focal_gamma'] = trial.suggest_float('focal_gamma', 0.5, 3.0)
    auprc, auroc, epochs, gaps = [], [], [], []
    for Xtr, ytr, Xva, yva in folds:
        model, history, best_epoch = train_mlp(params, Xtr, ytr, Xva, yva)    # early stopping on validation loss
        if best_epoch == 0:                                            # blew up in the first epoch
            return float(yva.mean())                                   # score it like guessing (AUPRC = prevalence)
        near = history[(history['epoch'] - best_epoch).abs() <= WINDOW]
        auprc.append(near['val_auprc'].mean())                         # one lucky epoch cannot win on its own
        auroc.append(near['val_auroc'].mean())
        kept = history.loc[best_epoch - 1]
        epochs.append(best_epoch)
        gaps.append(kept['val_loss'] - kept['train_loss'])
    trial.set_user_attr('fold_auprc', auprc)
    trial.set_user_attr('mean_auroc', float(np.mean(auroc)))
    trial.set_user_attr('fold_best_epoch', epochs)
    trial.set_user_attr('loss_gap', float(np.mean(gaps)))
    trial.set_user_attr('parameters', n_trainable(make_mlp(params['widths'], 0.0, X.shape[1])))
    return float(np.mean(auprc))


study_name = NAME if NAME == 'mlp_small' else f'{NAME}_auprc'
if NAME == 'mlp_small':
    # 8 workers (job array) share one study in a journal file; each keeps asking for trials until 300 are finished.
    # The study is created once beforehand (WORKER unset); workers (WORKER=1) only join it.
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    storage = mlp_small_storage()
    seed = SEED + int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))    # each worker its own sampler seed
    study = optuna.create_study(study_name=study_name, storage=storage, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if os.environ.get('WORKER'):
        study.optimize(mlp_objective, callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,))])
    else:
        print('study created; trials so far:', len(study.trials)); sys.exit()
elif os.environ.get('RESUME'):
    # Continue an unfinished search (the FairVLMed SVM search reached the job time limit after 48 of 100 trials):
    # workers (job array) join the saved study and stop when N_TRIALS trials are complete. Same search space and
    # objective; each worker has its own sampler seed.
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    seed = SEED + 1 + int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    study = optuna.load_study(study_name=study_name, storage=STORAGE, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(sklearn_objective, callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,))])
else:
    try:
        optuna.delete_study(study_name=study_name, storage=STORAGE)
    except KeyError:
        pass
    sampler = (optuna.samplers.GridSampler(KNN_GRID, seed=SEED) if NAME == 'knn'
               else optuna.samplers.TPESampler(seed=SEED))
    study = optuna.create_study(study_name=study_name, storage=STORAGE, direction='maximize', sampler=sampler)
    study.optimize(sklearn_objective, n_trials=N_TRIALS)
b = study.best_trial
print(f'{study_name}: best 5-fold mean AUPRC {b.value:.4f} (guessing {y[lab["part"].isin(["train", "val"])].mean():.3f}); '
      f'folds {[round(a, 3) for a in b.user_attrs["fold_auprc"]]}; mean AUROC {b.user_attrs["mean_auroc"]:.4f}')
print('best hyperparameters:', b.params)
if NAME == 'mlp_small':
    print('kept epoch per fold:', b.user_attrs['fold_best_epoch'], ' loss gap:', round(b.user_attrs['loss_gap'], 3),
          ' trainable parameters:', b.user_attrs['parameters'])
