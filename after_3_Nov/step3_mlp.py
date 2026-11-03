"""Step 3c: tune the MLP with Optuna. The test part is not used.
Rahele (3 Nov): the MLP overfits by epoch ~14, so the search also covers what fights overfitting:
optimiser, learning rate and its schedule, L2 weight decay, L1 penalty, dropout, batch size and loss function; 1-5 layers.
Each setting is scored by 5-fold cross-validation on training + validation, not on the one validation part
(28 glaucoma images): there, 300 trials picked a lucky epoch (README, "Result of the wider MLP search")."""
import os

import numpy as np
import optuna

from common import SEED, STORAGE, cv_folds, load_data, train_mlp

N_TRIALS = int(os.environ.get('N_TRIALS', 300))                        # more settings to search, so more trials
WINDOW = 5                                       # score a fold by its validation AUROC averaged over the kept epoch +- 5
lab, X, y = load_data()
folds = cv_folds(lab, X, y)                      # 5 x (training, validation); each scaled on its own training rows


def objective(trial):
    params = {'n_layers': trial.suggest_int('n_layers', 1, 5),                       # was 1-3
              'width': trial.suggest_categorical('width', [32, 64, 128, 256, 512]),  # was up to 256
              'dropout': trial.suggest_float('dropout', 0.0, 0.7),                   # was 0-0.5
              'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop']),
              'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),     # was 1e-4..1e-2; the best sat at 1.8e-4, near 1e-4
              'lr_schedule': trial.suggest_categorical('lr_schedule', ['constant', 'plateau']),
              'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True),   # L2; was up to 1e-2
              'l1': trial.suggest_float('l1', 1e-8, 1e-2, log=True),     # L1; 1e-8 is practically off; was up to 1e-3
              'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
              'loss': trial.suggest_categorical('loss', ['weighted_bce', 'focal'])}
    if params['loss'] == 'focal':
        params['focal_gamma'] = trial.suggest_float('focal_gamma', 0.5, 3.0)
    aurocs, epochs, gaps = [], [], []
    for Xtr, ytr, Xva, yva in folds:                                  # the SAME setting in all 5 folds
        model, history, best_epoch = train_mlp(params, Xtr, ytr, Xva, yva)   # early stopping on this fold's validation loss
        if best_epoch == 0:                                           # blew up in the first epoch: no usable model
            return 0.5                                                # score it like a coin flip
        near = history[(history['epoch'] - best_epoch).abs() <= WINDOW]
        aurocs.append(near['val_auroc'].mean())                       # one lucky epoch cannot win on its own
        kept = history.loc[best_epoch - 1]
        epochs.append(best_epoch)                                     # how late the overfitting starts
        gaps.append(kept['val_loss'] - kept['train_loss'])            # how far validation loss sits above training loss
    trial.set_user_attr('fold_auroc', aurocs)
    trial.set_user_attr('fold_best_epoch', epochs)
    trial.set_user_attr('loss_gap', float(np.mean(gaps)))
    return float(np.mean(aurocs))                                     # Optuna sees ONE number: the mean over the 5 folds


try:
    optuna.delete_study(study_name='mlp', storage=STORAGE)
except KeyError:
    pass
study = optuna.create_study(study_name='mlp', storage=STORAGE, direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
best = study.best_trial
print('best 5-fold score (validation AUROC, kept epoch +- 5):', round(best.value, 3), '  folds:', [round(a, 3) for a in best.user_attrs['fold_auroc']],
      '  best hyperparameters:', best.params)
print('kept epoch per fold:', best.user_attrs['fold_best_epoch'],
      '  validation minus training loss:', round(best.user_attrs['loss_gap'], 3))
