"""Loss-function check (10 Nov): the best small MLP from step3_auprc.py mlp_small, trained with different losses.
Rahele: "play with the loss function; give the rare class extra weight". Every loss is run with 5 seeds x the same
5 folds; each run is scored by the AUPRC and AUROC of the kept model (lowest validation loss) on its validation fold.
The test part is not used.   python check_losses.py  ->  check_losses.csv"""
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import mlp_small_storage, cv_folds, load_data, mlp_scores, train_mlp

STORAGE = 'sqlite:///optuna.db'
best = optuna.load_study(study_name='mlp_small', storage=mlp_small_storage()).best_params
base = {k: v for k, v in best.items() if k not in ('width1', 'width2', 'loss', 'pos_weight_scale', 'focal_gamma')}
base.update({'n_layers': 2, 'widths': [best['width1'], best['width2']]})
LOSSES = {'plain cross-entropy (no class weight)': {'loss': 'bce'},
          'weighted cross-entropy, weight 6.1 (healthy / glaucoma)': {'loss': 'weighted_bce', 'pos_weight_scale': 1.0},
          'weighted cross-entropy, weight 12.3 (2 x)': {'loss': 'weighted_bce', 'pos_weight_scale': 2.0},
          'focal loss (gamma 2), weight 6.1': {'loss': 'focal', 'pos_weight_scale': 1.0, 'focal_gamma': 2.0},
          'as found by Optuna': {k: best[k] for k in ('loss', 'pos_weight_scale', 'focal_gamma') if k in best}}
lab, X, y = load_data()
folds = cv_folds(lab, X, y)
rows = []
for name, loss in LOSSES.items():
    params = {**base, **loss}
    for seed in range(5):
        for k, (Xtr, ytr, Xva, yva) in enumerate(folds):
            model, history, epoch = train_mlp(params, Xtr, ytr, Xva, yva, seed=seed)
            s = mlp_scores(model, Xva)
            kept = history.loc[epoch - 1]
            rows.append({'loss': name, 'seed': seed, 'fold': k, 'kept_epoch': epoch,
                         'auprc': average_precision_score(yva, s), 'auroc': roc_auc_score(yva, s),
                         'loss_gap': kept['val_loss'] - kept['train_loss']})
    print(name, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv('check_losses.csv', index=False)
per_seed = df.groupby(['loss', 'seed'])[['auprc', 'auroc']].mean()         # 5-fold mean per seed
summary = per_seed.groupby('loss').agg(['mean', 'min', 'max'])
summary[('kept_epoch', 'median')] = df.groupby('loss')['kept_epoch'].median()
summary[('loss_gap', 'mean')] = df.groupby('loss')['loss_gap'].mean()
print('best small MLP:', base)
print(summary.round(4).to_string())
