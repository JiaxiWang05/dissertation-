"""Learning curve of the best small MLP (10 Nov): trained for all 300 epochs without early stopping on the fixed split
(939 training, 202 validation images), to show the whole curve. The test part is not used.
python step4_learning_curve.py  ->  mlp_small_learning_curve.csv"""
import optuna
from sklearn.preprocessing import StandardScaler

from common import mlp_small_storage, load_data, make_mlp, n_trainable, part, train_mlp

best = optuna.load_study(study_name='mlp_small', storage=mlp_small_storage()).best_params
params = {k: v for k, v in best.items() if k not in ('width1', 'width2')}
params.update({'n_layers': 2, 'widths': [best['width1'], best['width2']]})
lab, X, y = load_data()
Xtr, ytr = part(lab, X, y, 'train')
Xva, yva = part(lab, X, y, 'val')
scaler = StandardScaler().fit(Xtr)
_, history, best_epoch = train_mlp(params, scaler.transform(Xtr), ytr, scaler.transform(Xva), yva,
                                   max_epochs=300, patience=300)        # patience 300 = never stop early
history.to_csv('mlp_small_learning_curve.csv', index=False)
print('setting:', params)
print('trainable parameters:', n_trainable(make_mlp(params['widths'], 0.0, X.shape[1])))
print('lowest validation loss at epoch', best_epoch, '| validation loss there', round(history.loc[best_epoch - 1, 'val_loss'], 3),
      '| at epoch 300:', round(history['val_loss'].iloc[-1], 3), '| training loss at 300:', round(history['train_loss'].iloc[-1], 3))
print('validation AUPRC at the lowest loss', round(history.loc[best_epoch - 1, 'val_auprc'], 3),
      '| AUROC', round(history.loc[best_epoch - 1, 'val_auroc'], 3))
