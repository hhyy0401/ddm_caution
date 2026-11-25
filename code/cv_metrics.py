"""Interpretable held-out metrics for leave-one-penalty-level-out CV.
For each held-out cell (penalty x load): predicted threshold a vs free-fit a (descriptive best),
predicted accuracy and mean RT vs observed (in-window trials)."""
import os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
import cv_penalty as C
from fit_mle import subject_data, RES

def pred_er_rt(v, a, t0):
    er = 1 / (1 + np.exp(v * a)); dt = a / (2 * v) * np.tanh(v * a / 2)
    return 1 - er, dt + t0

def run(sid, df, iti, full):
    data = subject_data(df); rt, acc, cell, lv = data
    pidx = cell % 5; afree = np.array([full[f"free_p{j}"] for j in range(4, 14)])
    rows = []
    for model in C.MODELS:
        n = 4 + len(C.SPEC[model][0]); xf = np.array([full[f"{model}_p{j}"] for j in range(n)])
        for k in range(5):
            x = C.fit(model, data, iti, pidx != k, xf, None)
            v, t0 = np.array(x[0:2]), np.array(x[2:4])
            a = C.MODELS[model].thresholds(model, x[4:], v, t0, iti)
            for l in (0, 1):
                c = 5 * l + k; m = cell == c
                if m.sum() < 5: continue
                pa, prt = pred_er_rt(v[l], a[c], t0[l])
                rows.append(dict(Subject=sid, model=model, P=[5,10,20,40,80][k], load=l, a_pred=a[c], a_free=afree[c],
                                 acc_pred=pa, acc_obs=acc[m].mean(), rt_pred=prt, rt_obs=rt[m].mean()))
    return rows

def main():
    trials = pd.read_csv(os.path.join(RES, "trials.csv")); subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    full = pd.read_csv(os.path.join(RES, "mle_fits.csv")).merge(pd.read_csv(os.path.join(RES, "alt_fits.csv")), on="Subject").set_index("Subject")
    jobs = [(s, g, subj.loc[s, ["iti_noload", "iti_load"]].to_numpy(float), full.loc[s]) for s, g in trials.groupby("Subject")]
    if len(sys.argv) > 1: jobs = jobs[: int(sys.argv[1])]
    out = pd.DataFrame([r for rs in Parallel(n_jobs=10)(delayed(run)(*j) for j in jobs) for r in rs])
    out.to_csv(os.path.join(RES, "cv_metrics.csv"), index=False)
    out["ea"] = (out.a_pred - out.a_free).abs(); out["eacc"] = (out.acc_pred - out.acc_obs).abs(); out["ert"] = (out.rt_pred - out.rt_obs).abs()
    print(out.groupby("model")[["ea", "eacc", "ert"]].mean().round(4))
    print(out.groupby(["model", "P"]).ea.mean().unstack().round(3))
    g = out.groupby(["model", "P"])[["a_pred", "a_free", "acc_pred", "acc_obs", "rt_pred", "rt_obs"]].mean().round(3); print(g)

if __name__ == "__main__": main()
