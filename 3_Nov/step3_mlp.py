"""Step 3c: tune the MLP with Optuna. Learn on training, score on validation. The test part is not used."""
import optuna
from sklearn.preprocessing import StandardScaler

from common import SEED, STORAGE, load_data, part, train_mlp

N_TRIALS = 100
lab, X, y = load_data()
Xtr, ytr = part(lab, X, y, 'train')
Xva, yva = part(lab, X, y, 'val')
scaler = StandardScaler().fit(Xtr)                                    # fitted on training only
Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)


def objective(trial):
    params = {'n_layers': trial.suggest_int('n_layers', 1, 3),
              'width': trial.suggest_categorical('width', [32, 64, 128, 256]),
              'dropout': trial.suggest_float('dropout', 0.0, 0.5),
              'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
              'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)}
    model, history, best_epoch = train_mlp(params, Xtr_s, ytr, Xva_s, yva)   # early stopping on validation loss
    trial.set_user_attr('best_epoch', best_epoch)
    return history.loc[best_epoch - 1, 'val_auroc']                   # validation AUROC at the kept epoch


try:
    optuna.delete_study(study_name='mlp', storage=STORAGE)
except KeyError:
    pass
study = optuna.create_study(study_name='mlp', storage=STORAGE, direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
print('best validation AUROC:', round(study.best_value, 3), '  best hyperparameters:', study.best_params,
      '  best epoch:', study.best_trial.user_attrs['best_epoch'])
