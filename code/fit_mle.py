"""Exact-likelihood MLE of static threshold-policy models, per subject.

Common parameters: drift v_l and non-decision time t0_l for l in {NoLoad, Load}.
Threshold policies (10 cells = 2 loads x 5 penalties):
  free     : a free per cell                                   (descriptive ceiling)
  reg      : log a = alpha_l + beta_l log2(P/20)               (descriptive, 8 par)
  opt      : a = a*(v_l, t0_l, ITI_l; R, P)                    (pure reward-rate optimal)
  offset   : a = a*(...; P) + delta_l                          (stake-independent extra caution)
  avers    : a = a*(...; lambda P)                             (error aversion, shared lambda)
  avers_L  : a = a*(...; lambda_l P)                           (error aversion per load)
  belief   : a = a*(kappa v_l, ...; P)                         (miscalibrated belief about own drift)
  compress : a = a*(...; lambda_l 20 (P/20)^gamma_l)           (aversion + compressed penalty sensitivity)
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PENS = np.array([5, 10, 20, 40, 80], float)
R = 20.0
CELLS = [(l, p) for l in (0, 1) for p in PENS]          # index = 5*l + k


def thresholds(model, th, v, t0, iti):
    """Return a[10] for parameters th (model-specific), v[2], t0[2], iti[2]."""
    lv = np.repeat([0, 1], 5)
    P = np.tile(PENS, 2)
    if model == "free":
        return np.asarray(th)
    if model == "reg":
        al, be = np.array(th[:2]), np.array(th[2:4])
        return np.exp(al[lv] + be[lv] * np.log2(P / 20))
    vv, tt, ii = v[lv], t0[lv], iti[lv]
    if model == "opt":
        return D.optimal_a(vv, tt, ii, R, P)
    if model == "offset":
        return np.maximum(D.optimal_a(vv, tt, ii, R, P) + np.array(th)[lv], 0.1)
    if model == "avers":
        return D.optimal_a(vv, tt, ii, R, np.exp(th[0]) * P)
    if model == "avers_L":
        return D.optimal_a(vv, tt, ii, R, np.exp(np.array(th))[lv] * P)
    if model == "belief":
        return D.optimal_a(np.exp(th[0]) * vv, tt, ii, R, P)
    if model == "compress":
        lam, gam = np.exp(np.array(th[:2])), np.array(th[2:4])
        return D.optimal_a(vv, tt, ii, R, lam[lv] * 20 * (P / 20) ** gam[lv])
    raise ValueError(model)


TH_SPEC = {  # (init, bounds) for policy parameters
    "free": ([1.4] * 10, [(0.2, 5.0)] * 10),
    "reg": ([0.3, 0.5, 0.1, 0.1], [(-2, 2)] * 2 + [(-1, 1)] * 2),
    "opt": ([], []),
    "offset": ([0.3, 0.3], [(-1.5, 3)] * 2),
    "avers": ([0.5], [(-4, 5)]),
    "avers_L": ([0.5, 0.5], [(-4, 5)] * 2),
    "belief": ([-0.3], [(-3, 1.5)]),
    "compress": ([0.5, 0.5, 0.7, 0.7], [(-4, 5)] * 2 + [(0.0, 2.5)] * 2),
}
MODELS = list(TH_SPEC)


def subject_data(df):
    d = df[df.in_win]
    cell = (5 * d.load + d.Penalty.map({5: 0, 10: 1, 20: 2, 40: 3, 80: 4})).to_numpy()
    return d.rt.to_numpy(float), d.acc.to_numpy(np.int64), cell, d.load.to_numpy()


def negll(x, model, data, iti):
    rt, acc, cell, lv = data
    v, t0, th = np.array(x[0:2]), np.array(x[2:4]), x[4:]
    a = thresholds(model, th, v, t0, iti)
    if not np.all(np.isfinite(a)):
        return 1e10
    return -D.loglik_trials(rt, acc, v[lv], a[cell], t0[lv])


def fit_subject(sid, df, iti, starts=3, seed=0):
    data = subject_data(df)
    rng = np.random.default_rng(seed + int(sid))
    tmin = [data[0][data[3] == l].min() for l in (0, 1)]
    out = {"Subject": sid, "n": len(data[0])}
    for model in MODELS:
        init_th, b_th = TH_SPEC[model]
        bounds = [(0.01, 10)] * 2 + [(0.05, tmin[0] - 0.01), (0.05, tmin[1] - 0.01)] + b_th
        best = None
        for s in range(starts):
            x0 = np.r_[3.5, 3.0, 0.35, 0.40, init_th]
            if s:
                x0 = x0 + rng.normal(0, 0.15, len(x0)) * np.r_[3, 3, 0.2, 0.2, np.ones(len(init_th))]
            x0 = np.clip(x0, [b[0] + 1e-3 for b in bounds], [b[1] - 1e-3 for b in bounds])
            r = minimize(negll, x0, args=(model, data, iti), method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        k = len(best.x)
        out[f"{model}_nll"] = best.fun
        out[f"{model}_bic"] = 2 * best.fun + k * np.log(len(data[0]))
        out[f"{model}_k"] = k
        for j, val in enumerate(best.x):
            out[f"{model}_p{j}"] = val
    return out


def main():
    trials = pd.read_csv(os.path.join(RES, "trials.csv"))
    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    subj = subj[subj.include]
    groups = [(sid, g, subj.loc[sid, ["iti_noload", "iti_load"]].to_numpy(float))
              for sid, g in trials.groupby("Subject")]
    if len(sys.argv) > 1:
        groups = groups[: int(sys.argv[1])]
    rows = Parallel(n_jobs=10, verbose=5)(delayed(fit_subject)(*g) for g in groups)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "mle_fits.csv"), index=False)
    bic = out[[f"{m}_bic" for m in MODELS]]
    bic.columns = MODELS
    print("summed dBIC vs best:\n", (bic.sum() - bic.sum().min()).round(1).sort_values())
    print("per-subject wins:\n", bic.idxmin(1).value_counts())


if __name__ == "__main__":
    main()
