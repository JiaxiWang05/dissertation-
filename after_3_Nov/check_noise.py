"""Check: does Gaussian noise on the training features reduce MLP overfitting?
During training only, every batch becomes x + noise * N(0, 1) (common.train_mlp); validation is never changed.
Part 1: three MLP settings x noise 0 - 2 x 5 seeds x the same 5 folds as step3_mlp.py (training + validation; the
test part is not used). Per run: AUROC of the kept model, kept epoch, validation minus training loss at that epoch
(overfitting), lowest validation loss.
Part 2: full 300-epoch learning curves (no early stop) on the fixed training / validation split, one per setting and
noise level, to see whether validation loss still climbs.
Run: python check_noise.py   (writes check_noise.csv and check_noise_curves.csv)"""
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import STORAGE, cv_folds, load_data, part, train_mlp

NOISE = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]     # 0 = no noise (the earlier results)
SEEDS = [0, 1, 2, 3, 4]                          # as in check_mlp_seeds.py, so noise 0 repeats its runs
SETTINGS = {
    '3 Nov setting (Adam, 2 x 128)':
        {'n_layers': 2, 'width': 128, 'dropout': 0.1244, 'lr': 0.0001757, 'weight_decay': 1.029e-05},
    'trial 72 (RMSprop, 2 x 64, dropout 0.65)':
        {'n_layers': 2, 'width': 64, 'dropout': 0.6547, 'optimizer': 'rmsprop', 'lr': 0.005501, 'weight_decay': 0.002673,
         'l1': 0.0002161, 'batch_size': 64, 'loss': 'weighted_bce'},
    'trial 112 (5-fold search best, Adam, 2 x 512)': optuna.load_study(study_name='mlp', storage=STORAGE).best_params,
}
lab, X, y = load_data()
folds = cv_folds(lab, X, y)

# ---- part 1: 5 folds x 5 seeds
rows = []
for name, params in SETTINGS.items():
    for noise in NOISE:
        for seed in SEEDS:
            for k, (Xtr, ytr, Xva, yva) in enumerate(folds):
                model, history, best_epoch = train_mlp({**params, 'noise': noise}, Xtr, ytr, Xva, yva, seed=seed)
                if best_epoch == 0:              # blew up in the first epoch: score it like a coin flip
                    rows.append({'setting': name, 'noise': noise, 'seed': seed, 'fold': k, 'auroc': 0.5})
                    continue
                kept = history.loc[best_epoch - 1]
                rows.append({'setting': name, 'noise': noise, 'seed': seed, 'fold': k, 'auroc': kept['val_auroc'],
                             'kept_epoch': best_epoch, 'loss_gap': kept['val_loss'] - kept['train_loss'],
                             'lowest_val_loss': kept['val_loss'], 'train_loss': kept['train_loss']})
        print(f'{name}, noise {noise}: done', flush=True)
r = pd.DataFrame(rows)
r.to_csv('check_noise.csv', index=False)

per_seed = r.groupby(['setting', 'noise', 'seed'], sort=False)['auroc'].mean().groupby(level=[0, 1], sort=False)
by = r.groupby(['setting', 'noise'], sort=False)
summary = pd.DataFrame({'AUROC (mean of 5 seeds)': per_seed.mean(), 'lowest seed': per_seed.min(),
                        'highest seed': per_seed.max(), 'kept epoch (median)': by['kept_epoch'].median(),
                        'val minus train loss': by['loss_gap'].mean(), 'lowest val loss': by['lowest_val_loss'].mean()})
# paired with noise 0: same setting, seed and fold; a fold counts as better if its 5-seed mean AUROC is higher
fold_mean = r.groupby(['setting', 'noise', 'fold'], sort=False)['auroc'].mean()
base = fold_mean.xs(0.0, level='noise')
diff = fold_mean - base.reindex(fold_mean.index.droplevel('noise')).to_numpy()
summary['AUROC minus noise 0'] = diff.groupby(level=[0, 1], sort=False).mean()
summary['folds better than noise 0 (of 5)'] = (diff > 0).groupby(level=[0, 1], sort=False).sum()
pd.set_option('display.width', 250)
print(summary.round(3).to_string())

# ---- part 2: full learning curves on the fixed training / validation split
Xtr, ytr = part(lab, X, y, 'train')
Xva, yva = part(lab, X, y, 'val')
scaler = StandardScaler().fit(Xtr)
curves = []
for name, params in SETTINGS.items():
    for noise in NOISE:
        # patience=300: train all 300 epochs (no early stop) so the whole curve is visible
        model, history, best_epoch = train_mlp({**params, 'noise': noise}, scaler.transform(Xtr), ytr,
                                               scaler.transform(Xva), yva, patience=300)
        curves.append(history.assign(setting=name, noise=noise, lowest_val_loss_epoch=best_epoch))
        print(f'curve {name}, noise {noise}: lowest validation loss at epoch {best_epoch}', flush=True)
pd.concat(curves).to_csv('check_noise_curves.csv', index=False)
print('saved check_noise.csv and check_noise_curves.csv')
