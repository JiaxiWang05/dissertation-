"""Check: does any Optuna search have its best value at the edge of its search range?
If the best value sits at a bound, the search may have been cut short: the true best could lie outside the range,
so widen the range and search again (as for the SVM's C on 3 Nov).
For every study and every numeric hyperparameter: where the best trial's value sits in its range (0 % = lower bound,
100 % = upper bound; on the log scale for log ranges), and how many of the 10 best trials sit within 5 % of the same
bound. A penalty or dropout at its lower bound is 'practically off', a natural end, not a cut-off search.
Reads only. Run: python check_search_edges.py [optuna.db]"""
import sys

import numpy as np
import optuna
import pandas as pd

EDGE = 0.05                                      # within 5 % of the range counts as 'at the edge'
NATURAL_END = {('l1', 'lower'), ('weight_decay', 'lower'), ('dropout', 'lower')}   # 'off' is the natural limit
storage = 'sqlite:///' + (sys.argv[1] if len(sys.argv) > 1 else 'optuna.db')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def position(value, dist):
    """Where value sits in its range, 0 = lower bound, 1 = upper bound (None if the choices are not numbers)."""
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        choices = dist.choices
        if not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in choices):
            return None
        lo, hi, log = min(choices), max(choices), False
    else:
        lo, hi, log = dist.low, dist.high, dist.log
    if log:
        return (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (value - lo) / (hi - lo)


rows = []
for summary in optuna.get_all_study_summaries(storage):
    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:10]
    best = top[0]
    for name, dist in best.distributions.items():
        p = position(best.params[name], dist)
        if p is None:
            continue
        side = 'lower' if p <= EDGE else 'upper' if p >= 1 - EDGE else ''
        same = sum(1 for t in top if name in t.params and position(t.params[name], t.distributions[name]) is not None
                   and ((side == 'lower' and position(t.params[name], t.distributions[name]) <= EDGE)
                        or (side == 'upper' and position(t.params[name], t.distributions[name]) >= 1 - EDGE)))
        lo, hi = ((min(dist.choices), max(dist.choices)) if isinstance(dist, optuna.distributions.CategoricalDistribution)
                  else (dist.low, dist.high))
        verdict = ('' if not side else 'natural end (off)' if (name, side) in NATURAL_END
                   else f'AT {side.upper()} EDGE: widen and search again')
        rows.append({'study': summary.study_name, 'trials': len(done), 'best score': round(best.value, 4),
                     'hyperparameter': name, 'range': f'{lo:g} - {hi:g}', 'best value': f'{best.params[name]:.3g}',
                     'position in range': f'{100 * p:.0f} %', 'top-10 at that edge': same if side else '',
                     'verdict': verdict})
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 60)
print(pd.DataFrame(rows).to_string(index=False))
