"""Audit saved fits without refitting models or changing original result files.
Run with the existing ddm-sbi environment. Writes revision_audit.json here.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / 'code'))
import ddm_core as D

fits = pd.read_csv(BASE / 'results/mle_fits.csv')
subjects = pd.read_csv(BASE / 'results/subjects.csv').set_index('Subject')
actual, optimal = [], []
for _, row in fits.iterrows():
    a = row[[f'free_p{j}' for j in range(4, 14)]].to_numpy(float).reshape(2, 5)
    opt = np.array([
        D.optimal_a(np.repeat(row[f'free_p{l}'], 5),
                    np.repeat(row[f'free_p{l+2}'], 5),
                    np.repeat(subjects.loc[row.Subject, ['iti_noload', 'iti_load'][l]], 5),
                    20, np.array([5, 10, 20, 40, 80])) for l in (0, 1)])
    actual.append(a.mean(0))
    optimal.append(opt.mean(0))
a, opt = np.array(actual), np.array(optimal)
gap = a - opt
rng = np.random.default_rng(0)
audit = {'n': len(fits), 'median_thresholds': np.median(a, axis=0).tolist(),
         'median_optima': np.median(opt, axis=0).tolist(),
         'endpoint_crossover_share': float(np.mean((gap[:, 0] > 0) & (gap[:, 4] < 0))),
         'bootstrap_resamples': 2000, 'bootstrap_seed': 0}
for k in (0, 4):
    b = np.median(gap[rng.integers(len(gap), size=(2000, len(gap))), k], axis=1)
    audit[f'gap_penalty_{[5,10,20,40,80][k]}'] = {
        'median': float(np.median(gap[:, k])), 'ci95': np.percentile(b, [2.5, 97.5]).tolist()}
cv = pd.read_csv(BASE / 'results/cv_penalty.csv')
audit['heldout_log_likelihood'] = (-cv[[c for c in cv if c.endswith('cv_nll')]].sum()).to_dict()
findings = pd.read_csv(BASE / 'results/generalize_findings.csv')
audit['existing_bootstrap_summaries'] = findings.fillna('').to_dict(orient='records')
(HERE / 'revision_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'existing_bootstrap_summaries'}, indent=2))
