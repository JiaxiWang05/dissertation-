"""Check after step3_mlp.py: is the best setting of the 5-fold search really better, or just lucky?
Retrain four settings with 5 new seeds in the same 5 folds (training + validation; the test part is not used).
Per setting: AUROC of the kept model (lowest validation loss), the epoch it is kept at, how far validation loss sits
above training loss at that epoch (overfitting), and how much validation AUROC jumps from one epoch to the next (noise).
Run: python check_mlp_seeds.py   (writes check_mlp_seeds.csv)"""
import optuna
import pandas as pd

from common import STORAGE, cv_folds, load_data, train_mlp

SEEDS = [0, 1, 2, 3, 4]                          # not 42, the seed the search used
SETTINGS = {
    '3 Nov setting (Adam, 2 x 128)':
        {'n_layers': 2, 'width': 128, 'dropout': 0.1244, 'lr': 0.0001757, 'weight_decay': 1.029e-05},
    'one-validation-part search, trial 174 (its best)':
        {'n_layers': 3, 'width': 64, 'dropout': 0.6435, 'optimizer': 'rmsprop', 'lr': 0.01077, 'weight_decay': 0.001966,
         'l1': 0.0008480, 'batch_size': 64, 'loss': 'weighted_bce'},
    'one-validation-part search, trial 72':
        {'n_layers': 2, 'width': 64, 'dropout': 0.6547, 'optimizer': 'rmsprop', 'lr': 0.005501, 'weight_decay': 0.002673,
         'l1': 0.0002161, 'batch_size': 64, 'loss': 'weighted_bce'},
    '5-fold search, its best': optuna.load_study(study_name='mlp', storage=STORAGE).best_params,
}
lab, X, y = load_data()
folds = cv_folds(lab, X, y)

rows = []
for name, params in SETTINGS.items():
    for seed in SEEDS:
        for k, (Xtr, ytr, Xva, yva) in enumerate(folds):
            model, history, best_epoch = train_mlp(params, Xtr, ytr, Xva, yva, seed=seed)
            if best_epoch == 0:                  # blew up in the first epoch: score it like a coin flip
                rows.append({'setting': name, 'seed': seed, 'fold': k, 'auroc': 0.5})
                continue
            kept = history.loc[best_epoch - 1]
            rows.append({'setting': name, 'seed': seed, 'fold': k, 'auroc': kept['val_auroc'], 'kept_epoch': best_epoch,
                         'loss_gap': kept['val_loss'] - kept['train_loss'],
                         'auroc_jump': history['val_auroc'].diff().abs().mean()})
r = pd.DataFrame(rows)
r.to_csv('check_mlp_seeds.csv', index=False)

per_seed = r.groupby(['setting', 'seed'], sort=False)['auroc'].mean().groupby(level=0, sort=False)  # 5-fold mean per seed
by = r.groupby('setting', sort=False)
summary = pd.DataFrame({'kept-model AUROC, 5 folds (mean of 5 seeds)': per_seed.mean(), 'lowest seed': per_seed.min(),
                        'highest seed': per_seed.max(), 'kept epoch (median)': by['kept_epoch'].median(),
                        'val minus train loss': by['loss_gap'].mean(), 'AUROC jump per epoch': by['auroc_jump'].mean()})
pd.set_option('display.width', 250)
print(summary.round(3).to_string())
print('\n5-fold search best:', SETTINGS['5-fold search, its best'])
