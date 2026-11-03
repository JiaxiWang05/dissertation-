"""Check: does a later "overfitting epoch" mean less overfitting? Train the 3 Nov MLP setting in the same 5 folds
(training + validation; the test part is not used) with its own learning rate and with a 10x smaller one.
Per fold: the kept epoch (lowest validation loss), the AUROC of the kept model, validation minus training loss
at that epoch, and the lowest validation loss.  Run: python check_lr_epoch.py"""
import pandas as pd

from common import cv_folds, load_data, train_mlp

SETTING = {'n_layers': 2, 'width': 128, 'dropout': 0.1244, 'lr': 0.0001757, 'weight_decay': 1.029e-05}   # 3 Nov
lab, X, y = load_data()
folds = cv_folds(lab, X, y)

rows = []
for factor in [1, 0.1]:
    params = dict(SETTING, lr=SETTING['lr'] * factor)
    for k, (Xtr, ytr, Xva, yva) in enumerate(folds):
        model, history, best_epoch = train_mlp(params, Xtr, ytr, Xva, yva, seed=0)
        kept = history.loc[best_epoch - 1]
        rows.append({'lr': params['lr'], 'fold': k, 'kept epoch': best_epoch, 'AUROC': kept['val_auroc'],
                     'val minus train loss': kept['val_loss'] - kept['train_loss'], 'lowest val loss': kept['val_loss']})
r = pd.DataFrame(rows)
print(r.round(4).to_string(index=False))
print(r.groupby('lr', sort=False)[['kept epoch', 'AUROC', 'val minus train loss', 'lowest val loss']].mean().round(4))
