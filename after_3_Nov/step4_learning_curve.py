"""Step 4: learning curves of the best MLP: loss per epoch on training and validation (to see overfitting)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
from sklearn.preprocessing import StandardScaler

from common import STORAGE, load_data, part, train_mlp

lab, X, y = load_data()
Xtr, ytr = part(lab, X, y, 'train')
Xva, yva = part(lab, X, y, 'val')
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

params = optuna.load_study(study_name='mlp', storage=STORAGE).best_params
# patience=300: train all 300 epochs (no early stop) so the whole curve is visible, including any overfitting
model, history, best_epoch = train_mlp(params, Xtr_s, ytr, Xva_s, yva, patience=300)
history.to_csv('mlp_learning_curve.csv', index=False)
print('lowest validation loss at epoch', best_epoch)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history['epoch'], history['train_loss'], label='training loss')
ax1.plot(history['epoch'], history['val_loss'], label='validation loss')
ax1.axvline(best_epoch, color='grey', linestyle='--', label=f'lowest validation loss (epoch {best_epoch})')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss (weighted binary cross-entropy)')
ax1.legend()
ax2.plot(history['epoch'], history['val_auroc'], color='tab:orange')
ax2.axvline(best_epoch, color='grey', linestyle='--')
ax2.set_xlabel('epoch')
ax2.set_ylabel('validation AUROC')
fig.suptitle(f'MLP learning curves (best Optuna setting: {params})', fontsize=8)
fig.tight_layout()
fig.savefig('mlp_learning_curve.png', dpi=200)
print('saved mlp_learning_curve.png and mlp_learning_curve.csv')
