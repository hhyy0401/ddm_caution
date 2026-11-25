"""Do thresholds become more stake-differentiated within a 60-s block? Free-threshold MLE on
early (first 8 trials) vs late (trial >= 8, after a correct response) trials of each block."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
from fit_mle import subject_data, RES, PENS

def fit_free(df):
    rt, acc, cell, lv = subject_data(df)
    tmin = [rt[lv == l].min() for l in (0, 1)]
    def nll(x):
        v, t0, a = x[0:2], x[2:4], x[4:]
        return -D.loglik_trials(rt, acc, v[lv], a[cell], t0[lv])
    b = [(0.01, 10)] * 2 + [(0.05, tmin[0] - 0.01), (0.05, tmin[1] - 0.01)] + [(0.2, 5)] * 10
    r = minimize(nll, np.r_[2, 1.6, 0.3, 0.33, [1.6] * 10], method="L-BFGS-B", bounds=b)
    return r.x

def one(sid, g):
    e = g[g.pos < 8]; l = g[(g.pos >= 8) & (g.prev_acc == 1)]
    if e.load.nunique() < 2 or e.Penalty.nunique() < 5: return None
    xe, xl = fit_free(e), fit_free(l)
    sl = lambda a: [np.polyfit(np.log2(PENS), a[5*k:5*k+5], 1)[0] for k in (0, 1)]
    return dict(Subject=sid, slope_early_NL=sl(xe[4:])[0], slope_early_L=sl(xe[4:])[1],
                slope_late_NL=sl(xl[4:])[0], slope_late_L=sl(xl[4:])[1],
                a_early=xe[4:].mean(), a_late=xl[4:].mean(), v_early=xe[:2].mean(), v_late=xl[:2].mean())

t = pd.read_csv(os.path.join(RES, "trials.csv"))
rows = [r for r in Parallel(n_jobs=10)(delayed(one)(s, g) for s, g in t.groupby("Subject")) if r]
o = pd.DataFrame(rows); o.to_csv(os.path.join(RES, "early_late.csv"), index=False)
print("N", len(o)); print(o.drop(columns="Subject").median().round(3).to_string())
for k in ("NL", "L"):
    d = o[f"slope_late_{k}"] - o[f"slope_early_{k}"]
    print(f"{k}: slope late > early in {(d > 0).mean():.3f}, p = {wilcoxon(d).pvalue:.2g}")
d = o.a_early - o.a_late; print(f"mean a early > late in {(d > 0).mean():.3f}, p = {wilcoxon(d).pvalue:.2g}")
