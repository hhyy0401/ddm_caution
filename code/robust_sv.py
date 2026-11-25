"""Robustness: refit the free-threshold model with across-trial drift variability sv,
and recompute the reward-rate optimum under sv (Gauss-Hermite over v ~ N(v, sv^2))."""
import os, sys, math
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
from fit_mle import subject_data, PENS, RES

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_W = GH_W / GH_W.sum()


@njit(cache=True)
def loglik_sv(rt, acc, v, a, t0, sv):
    unif = D.CONTAM * 0.5 / (D.RT_MAX - D.RT_MIN)
    ll = 0.0
    for i in range(rt.shape[0]):
        t = rt[i] - t0[i]
        f = 0.0
        if t > 0.0:
            vv = -v[i] if acc[i] == 1 else v[i]          # lower-boundary convention
            w = 0.5
            f0 = D._fpt_lower(t, 0.0, a[i], w)
            k = 1.0 + sv * sv * t
            f = f0 / math.sqrt(k) * math.exp((sv * sv * a[i] * a[i] * w * w - 2 * a[i] * vv * w - vv * vv * t) / (2 * k))
        ll += math.log((1.0 - D.CONTAM) * f + unif)
    return ll


def optimal_a_sv(v, t0, iti, P, sv):
    g = D.A_GRID[None, :]
    best = []
    for vi, ti, ii, pi in zip(v, t0, iti, P):
        vs = vi + sv * GH_X[:, None]
        va = np.clip(vs * g, -50, 50)
        er = (GH_W[:, None] / (1 + np.exp(va))).sum(0)
        dt = (GH_W[:, None] * np.where(np.abs(vs) > 1e-6, g / (2 * vs) * np.tanh(va / 2), g * g / 4)).sum(0)
        rr = (20 * (1 - er) - pi * er) / (dt + ti + ii)
        best.append(D.A_GRID[rr.argmax()])
    return np.array(best)


def fit(sid, df, iti):
    rt, acc, cell, lv = subject_data(df)
    tmin = [rt[lv == l].min() for l in (0, 1)]
    def nll(x):
        v, t0, sv, a = x[0:2], x[2:4], x[4], x[5:]
        return -loglik_sv(rt, acc, v[lv], a[cell], t0[lv], sv)
    bounds = [(0.01, 10)] * 2 + [(0.05, tmin[0] - 0.01), (0.05, tmin[1] - 0.01), (0.0, 4.0)] + [(0.2, 5)] * 10
    x0 = np.r_[2.5, 2.0, 0.3, 0.33, 1.0, [1.5] * 10]
    r = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    v, t0, sv, a = r.x[0:2], r.x[2:4], r.x[4], r.x[5:]
    l10 = np.repeat([0, 1], 5)
    ao = optimal_a_sv(v[l10], t0[l10], iti[l10], np.tile(PENS, 2), sv)
    return dict(Subject=sid, sv=sv, v0=v[0], v1=v[1], nll=r.fun, **{f"a{j}": a[j] for j in range(10)},
                **{f"ao{j}": ao[j] for j in range(10)})


def main():
    trials = pd.read_csv(os.path.join(RES, "trials.csv"))
    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    jobs = [(sid, g, subj.loc[sid, ["iti_noload", "iti_load"]].to_numpy(float)) for sid, g in trials.groupby("Subject")]
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(fit)(*j) for j in jobs))
    out.to_csv(os.path.join(RES, "robust_sv.csv"), index=False)
    m = pd.read_csv(os.path.join(RES, "mle_fits.csv"))[["Subject", "free_nll"]]
    out = out.merge(m, on="Subject")
    A = out[[f"a{j}" for j in range(10)]].to_numpy(); AO = out[[f"ao{j}" for j in range(10)]].to_numpy()
    print("median sv", out.sv.median().round(2), " dBIC(sv - no sv) summed",
          round((2 * (out.nll - out.free_nll) + np.log(500)).sum(), 1))
    print("median a  ", np.median(A, 0).round(2)); print("median a* ", np.median(AO, 0).round(2))
    print("share a>a*", (A > AO).mean(0).round(2))


if __name__ == "__main__":
    main()
