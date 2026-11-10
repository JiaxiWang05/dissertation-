"""Left / right eye from the optic-disc position, and likely fellow-eye pairs (10 Nov).
The image headers carry no OD / OS (laterality_check.py). But on Remidio photographs the disc is off-centre, so its
horizontal position separates the two eyes: disc in the left part or the right part of the image (disc_boxes.csv: centre
of the 5-expert majority disc mask; made by ../pipeline/disc_boxes.py). Bosch and Forus photographs are centred on the
disc, so this does not work for them.
Which side is OD depends on the camera's convention (usually the disc is on the right of the image in a right eye); it
is not confirmed for this camera, so the sides are called "disc left" and "disc right".

Then: Remidio files are numbered IMG_####. Two images with consecutive numbers and opposite disc sides are taken as the
two eyes of one patient (a pseudo patient). Pairs are formed greedily along the numbers, so an image is in one pair only.
Outputs: remidio_disc_side.csv (side per image), pseudo_patients.csv (image_id -> pseudo patient group for ALL images:
pairs share a group; Bosch pairs and copies as before; everything else alone), and printed counts.
Then the 5-fold cross-validation of logistic regression is repeated with folds grouped by pseudo patient, to see how
much the fellow eyes change the out-of-fold result. Test part not used (except the counts of where pairs fall)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from common import SEED, load_data

lab, X, y = load_data()
box = pd.read_csv('disc_boxes.csv')
box = box[box['dataset'] == 'Chaksu'].set_index('image_id')
lab['disc_x'] = (box['cx'] / box['width']).reindex(lab['image_id']).to_numpy()
rem = lab['camera'] == 'Remidio'
lab['side'] = ''
lab.loc[rem, 'side'] = np.select([lab.loc[rem, 'disc_x'] < 0.45, lab.loc[rem, 'disc_x'] > 0.55],
                                 ['disc left', 'disc right'], 'centre')
print('Remidio disc side:', lab.loc[rem, 'side'].value_counts().to_dict())
print('disc x / width, 10 bins 0-1, per camera:')
for cam, g in lab.groupby('camera'):
    print(f'  {cam}: {np.histogram(g["disc_x"], bins=10, range=(0, 1))[0].tolist()}')

r = lab[rem & (lab['duplicate_of'].isna())].copy()          # copies are handled by their original's group
r['n'] = r['stem'].astype(str).str.extract(r'(\d+)$')[0].astype(int)
r['prefix'] = r['stem'].astype(str).str.extract(r'^(\D*)')[0]
r = r.sort_values(['prefix', 'n'])
rows = r[['image_id', 'prefix', 'n', 'side', 'label_majority', 'part']].to_dict('records')
pairs, i = [], 0
while i < len(rows) - 1:                                    # greedy: each image in at most one pair
    a, b = rows[i], rows[i + 1]
    if (a['prefix'] == b['prefix'] and b['n'] - a['n'] == 1 and 'centre' not in (a['side'], b['side'])
            and a['side'] != b['side']):
        pairs.append((a, b)); i += 2
    else:
        i += 1
cons = [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)
        if rows[i]['prefix'] == rows[i + 1]['prefix'] and rows[i + 1]['n'] - rows[i]['n'] == 1
        and 'centre' not in (rows[i]['side'], rows[i + 1]['side'])]
opp = np.mean([a['side'] != b['side'] for a, b in cons])
p_left = (r['side'] == 'disc left').sum() / (r['side'] != 'centre').sum()
print(f'consecutive Remidio numbers with both sides clear: {len(cons)}; opposite sides: {opp:.0%} '
      f'(two random images: {2 * p_left * (1 - p_left):.0%})')
same = np.mean([a['label_majority'] == b['label_majority'] for a, b in pairs])
q = r['label_majority'].mean()
print(f'pseudo patients (disjoint pairs): {len(pairs)}; same label in a pair: {same:.0%} (chance {q * q + (1 - q) ** 2:.0%}); '
      f'both glaucoma: {sum(a["label_majority"] == b["label_majority"] == 1 for a, b in pairs)}')
cross = [(a['part'], b['part']) for a, b in pairs if a['part'] != b['part']]
print(f'pairs split across training / validation / test: {len(cross)} of {len(pairs)}; '
      f'with one image in test and the other in training or validation: {sum("test" in c for c in cross)}')
n_test = (lab['part'] == 'test').sum()
test_with_fellow_out = sum(1 for a, b in pairs for u, v in [(a, b), (b, a)] if u['part'] == 'test' and v['part'] != 'test')
print(f'test images whose likely fellow eye is in training or validation: {test_with_fellow_out} of {n_test}')

group = lab.set_index('image_id')['eye'].copy()             # start from the current groups (Bosch pairs, copies)
for a, b in pairs:
    g = group[a['image_id']]
    group[group == group[b['image_id']]] = g                # b's group (and its copies) joins a's group
lab['pseudo_patient'] = group.reindex(lab['image_id']).to_numpy()
lab[rem][['image_id', 'stem', 'disc_x', 'side', 'label_majority', 'part']].to_csv('remidio_disc_side.csv', index=False)
lab[['image_id', 'eye', 'pseudo_patient']].to_csv('pseudo_patients.csv', index=False)
print(f'groups: by image group {lab["eye"].nunique()}, by pseudo patient {lab["pseudo_patient"].nunique()}')

# effect on cross-validation: logistic regression (C of 8 Nov), same features, folds grouped two ways
keep = lab['part'].isin(['train', 'val']).to_numpy()
sub, Xs, ys = lab[keep], X[keep], y[keep]
strata = sub['label_majority'].astype(str) + '_' + sub['camera']
for name, col in [('image groups (as used so far)', 'eye'), ('pseudo patients (fellow eyes together)', 'pseudo_patient')]:
    res = []
    for seed in range(10):                                   # 10 different fold assignments
        oof = np.zeros(len(ys))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(sub, strata, sub[col]):
            sc = StandardScaler().fit(Xs[tr])
            m = LogisticRegression(C=0.007250175039761133, class_weight='balanced', max_iter=5000)
            oof[va] = m.fit(sc.transform(Xs[tr]), ys[tr]).predict_proba(sc.transform(Xs[va]))[:, 1]
        res.append((roc_auc_score(ys, oof), average_precision_score(ys, oof)))
    res = np.array(res)
    print(f'{name}: out-of-fold AUROC {res[:, 0].mean():.3f} (range {res[:, 0].min():.3f}-{res[:, 0].max():.3f}), '
          f'AUPRC {res[:, 1].mean():.3f} (range {res[:, 1].min():.3f}-{res[:, 1].max():.3f}) over 10 fold seeds')
