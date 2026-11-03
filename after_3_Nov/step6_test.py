"""Step 6: test the chosen classifier ONCE on the test part and report the numbers. Run this only once."""
import json
import sys
from pathlib import Path

import joblib
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from common import at_threshold, bootstrap_auroc_ci, load_data, make_mlp, mlp_scores, part

if Path('TEST_DONE').exists():
    sys.exit('The test part has already been used. Report test_results.json; do not test again.')

lab, X, y = load_data()
Xte, yte = part(lab, X, y, 'test')
eyes = lab.loc[lab['part'] == 'test', 'eye'].to_numpy()
info = json.load(open('chosen.json'))
name, hp = info['classifier'], info['hyperparameters']

if name == 'mlp':
    model = make_mlp(hp['n_layers'], hp['width'], hp['dropout'])
    model.load_state_dict(torch.load('chosen_mlp.pt'))
    s = mlp_scores(model, joblib.load('chosen_scaler.joblib').transform(Xte))
else:
    model = joblib.load('chosen_model.joblib')
    s = model.predict_proba(Xte)[:, 1] if name in ('logreg', 'knn') else model.decision_function(Xte)

auroc, (lo, hi) = roc_auc_score(yte, s), bootstrap_auroc_ci(yte, s, eyes)
results = {'classifier': name, 'test images': int(len(yte)), 'test glaucoma': int(yte.sum()),
           'AUROC': auroc, 'AUROC 95% CI': [lo, hi], 'AUPRC': average_precision_score(yte, s),
           'AUPRC of guessing': float(yte.mean()),
           'at 90% recall threshold (from validation)': at_threshold(yte, s, info['threshold_recall90']),
           "at Youden's J threshold (from validation)": at_threshold(yte, s, info['threshold_youden'])}
json.dump(results, open('test_results.json', 'w'), indent=2, default=float)
# one score per test image, so figures can be drawn later without scoring the test part again
scores = lab.loc[lab['part'] == 'test', ['image_id', 'camera', 'eye', 'n_glaucoma_votes', 'label_majority']].copy()
scores['score'] = s
scores.to_csv('test_scores.csv', index=False)
Path('TEST_DONE').write_text('test part used once\n')

r = results['at 90% recall threshold (from validation)']
print(f"{name}: test AUROC {auroc:.3f} (95% CI {lo:.3f}-{hi:.3f}), AUPRC {results['AUPRC']:.3f} "
      f"(guessing {results['AUPRC of guessing']:.3f})")
print(f"At the 90%-recall threshold: found {r['TP']} of {r['TP'] + r['FN']} glaucoma (sensitivity {r['sensitivity']:.2f}); "
      f"{r['FP']} false alarms among {r['FP'] + r['TN']} healthy (specificity {r['specificity']:.2f}); "
      f"PPV {r['PPV']:.2f}, NPV {r['NPV']:.2f}, balanced accuracy {r['balanced_accuracy']:.2f}, MCC {r['MCC']:.2f}")
r = results["at Youden's J threshold (from validation)"]
print(f"At the Youden's J threshold: found {r['TP']} of {r['TP'] + r['FN']} glaucoma (sensitivity {r['sensitivity']:.2f}); "
      f"{r['FP']} false alarms among {r['FP'] + r['TN']} healthy (specificity {r['specificity']:.2f}); "
      f"PPV {r['PPV']:.2f}, NPV {r['NPV']:.2f}, balanced accuracy {r['balanced_accuracy']:.2f}, MCC {r['MCC']:.2f}")
print('saved test_results.json and test_scores.csv')
