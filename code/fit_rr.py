"""Resource-rational threshold policies (exact-likelihood MLE).

Chosen threshold in cell (load l, penalty P):
    a_c = argmax_a  RR(a; P) / RR*(P)  -  c_l * (a - a0_l)^2
i.e. people trade the fraction of optimal reward rate they keep against the cost of moving the
threshold away from a default a0. Variants:
  shrink      : a0_l, c_l free                          (8 par incl. v_l, t0_l)
  shrink_c    : a0_l free, c shared   -> load moves the default only
  shrink_a0   : a0 shared, c_l free   -> load raises the cost of adjusting
  shrink_ref  : a0_l = a*_l(P = 20), c_l free           (default = optimum at the reference stake)
Results are merged with results/mle_fits.csv for model comparison.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
from fit_mle import subject_data, PENS, RES, R

LV = np.repeat([0, 1], 5)
PP = np.tile(PENS, 2)
G = D.A_GRID[None, :]


def shrink_a(v, t0, iti, P, a0, c):
    rr = D.reward_rate(G, v[:, None], t0[:, None], iti[:, None], R, P[:, None])
    J = rr / rr.max(1, keepdims=True) - c[:, None] * (G - a0[:, None]) ** 2
    k = np.clip(J.argmax(1), 1, G.shape[1] - 2)
    i = np.arange(len(k))
    y0, y1, y2 = J[i, k - 1], J[i, k], J[i, k + 1]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    return D.A_GRID[k] + np.clip(off, -1, 1) * (D.A_GRID[1] - D.A_GRID[0])


def thresholds(model, th, v, t0, iti):
    vv, tt, ii = v[LV], t0[LV], iti[LV]
    if model == "shrink":
        a0, c = np.array(th[0:2])[LV], np.exp(np.array(th[2:4]))[LV]
    elif model == "shrink_c":
        a0, c = np.array(th[0:2])[LV], np.full(10, np.exp(th[2]))
    elif model == "shrink_a0":
        a0, c = np.full(10, th[0]), np.exp(np.array(th[1:3]))[LV]
    elif model == "shrink_ref":
        a0 = D.optimal_a(v, t0, iti, R, np.array([20.0, 20.0]))[LV]
        c = np.exp(np.array(th[0:2]))[LV]
    else:
        raise ValueError(model)
    return shrink_a(vv, tt, ii, PP, a0, c)


SPEC = {
    "shrink": ([1.5, 1.8, 0.0, 0.0], [(0.3, 4.0)] * 2 + [(-6, 8)] * 2),
    "shrink_c": ([1.5, 1.8, 0.0], [(0.3, 4.0)] * 2 + [(-6, 8)]),
    "shrink_a0": ([1.6, 0.0, 0.0], [(0.3, 4.0)] + [(-6, 8)] * 2),
    "shrink_ref": ([0.0, 0.0], [(-6, 8)] * 2),
}


def negll(x, model, data, iti):
    rt, acc, cell, lv = data
    v, t0 = np.array(x[0:2]), np.array(x[2:4])
    a = thresholds(model, x[4:], v, t0, iti)
    return -D.loglik_trials(rt, acc, v[lv], a[cell], t0[lv])


def fit_subject(sid, df, iti, starts=4):
    data = subject_data(df)
    rng = np.random.default_rng(int(sid))
    tmin = [data[0][data[3] == l].min() for l in (0, 1)]
    out = {"Subject": sid}
    for model, (init, b_th) in SPEC.items():
        bounds = [(0.01, 10)] * 2 + [(0.05, tmin[0] - 0.01), (0.05, tmin[1] - 0.01)] + b_th
        best = None
        for s in range(starts):
            x0 = np.r_[2.0, 1.6, 0.32, 0.36, init]
            if s:
                x0 = x0 + rng.normal(0, 1, len(x0)) * np.r_[0.4, 0.4, 0.03, 0.03, [0.3 if b[1] <= 4 else 1.5 for b in b_th]]
            x0 = np.clip(x0, [b[0] + 1e-3 for b in bounds], [b[1] - 1e-3 for b in bounds])
            r = minimize(negll, x0, args=(model, data, iti), method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        out[f"{model}_nll"] = best.fun
        out[f"{model}_bic"] = 2 * best.fun + len(best.x) * np.log(len(data[0]))
        for j, val in enumerate(best.x):
            out[f"{model}_p{j}"] = val
    return out


def main():
    trials = pd.read_csv(os.path.join(RES, "trials.csv"))
    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    jobs = [(sid, g, subj.loc[sid, ["iti_noload", "iti_load"]].to_numpy(float)) for sid, g in trials.groupby("Subject")]
    if len(sys.argv) > 1:
        jobs = jobs[: int(sys.argv[1])]
    new = pd.DataFrame(Parallel(n_jobs=10)(delayed(fit_subject)(*j) for j in jobs))
    new.to_csv(os.path.join(RES, "rr_fits.csv"), index=False)
    old = pd.read_csv(os.path.join(RES, "mle_fits.csv"))
    m = old.merge(new, on="Subject")
    models = ["free", "reg", "compress", "avers", "opt"] + list(SPEC)
    bic = m[[f"{k}_bic" for k in models]]; bic.columns = models
    print("summed dBIC vs best:\n", (bic.sum() - bic.sum().min()).round(1).sort_values())
    print("per-subject wins:\n", bic.idxmin(1).value_counts())


if __name__ == "__main__":
    main()
