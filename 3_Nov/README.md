# 3 Nov 2026: Chákṣu, frozen RETFound-Green, choosing the classifier

Real Chákṣu data only. Run on Hamilton: job 19328516 (2 Nov, steps 2-5) and job 19334646 (3 Nov, SVM with the
wider C range, then step 5 again). The test part has not been used yet (step 6 not run).

| Meeting part | Code | Results |
|---|---|---|
| Data split | `common.py` (`load_data`), `step2_check_split.py` | `chaksu_split.csv`; the split table is printed in `logs/learn_chaksu_19328516.out` |
| Optuna | `step3_logreg.py`, `step3_svm.py`, `step3_mlp.py`, `step3_knn.py` | `optuna.db` (every trial); `figures/fig2_optuna.png` |
| Learning curve | `step4_learning_curve.py` | `mlp_learning_curve.csv`, `mlp_learning_curve.png`; `figures/fig3_learning_curve.png` |
| Choose the model | `step5_compare.py` | `step5_comparison.csv`, `chosen.json`, `chosen_model.joblib`; `figures/fig4_comparison.png` |
| First comparison (before the SVM range was widened) | | `step5_comparison_run1.csv`, `chosen_run1.json` |
| Test (not run yet) | `step6_test.py` | will write `test_results.json`, `test_scores.csv` |
| Figures | `make_figures.py` (draws from the saved results only) | `figures/` |
| Hamilton job | `run_learn.slurm` | `logs/` |

Inputs: `chaksu_rfg_resize.npz` (384 RETFound-Green features per image; on Hamilton a link to
/nobackup/tggq61/retfound_green/features) and `../pipeline/chaksu_labels.csv` (labels, majority of 5 experts).
