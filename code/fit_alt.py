"""Alternative normative objectives and cost-form robustness (exact-likelihood MLE).

Alternative objectives from the reward-rate literature (implemented in their spirit):
  ra / ra_L         : a = argmax RR(a)/RR* - q * ER(a)                 (reward + accuracy weighting; q shared / per load)
  robust / robust_L : a = argmax min_{v' in [v/(1+u), v(1+u)]} RR(a; v')/RR*(v')   (maximin under drift uncertainty)
Cost-form robustness of the resource-rational model:
  shrink_c_log / shrink_log : cost c (log a - log a0)^2
  shrink_c_abs / shrink_abs : un-normalised objective RR(a) [points/s] - c (a - a0)^2
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
UGRID = np.linspace(-1, 1, 9)


def _argmax(J):
    k = np.clip(J.argmax(1), 1, J.shape[1] - 2); i = np.arange(len(k))
    y0, y1, y2 = J[i, k - 1], J[i, k], J[i, k + 1]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    return D.A_GRID[k] + np.clip(off, -1, 1) * (D.A_GRID[1] - D.A_GRID[0])


def _er(a, v):
    return 1.0 / (1.0 + np.exp(np.clip(v * a, -50, 50)))


def thresholds(model, th, v, t0, iti):
    vv, tt, ii = v[LV][:, None], t0[LV][:, None], iti[LV][:, None]
    P = PP[:, None]
    rr = D.reward_rate(G, vv, tt, ii, R, P)
    rel = rr / rr.max(1, keepdims=True)
    if model in ("ra", "ra_L"):
        q = np.full(10, np.exp(th[0])) if model == "ra" else np.exp(np.array(th[0:2]))[LV]
        return _argmax(rel - q[:, None] * _er(G, vv))
    if model in ("robust", "robust_L"):
        u = np.full(10, th[0]) if model == "robust" else np.array(th[0:2])[LV]
        worst = np.full_like(rel, np.inf)
        for z in UGRID:
            vz = vv * (1 + u[:, None]) ** z
            r2 = D.reward_rate(G, vz, tt, ii, R, P)
            worst = np.minimum(worst, r2 / r2.max(1, keepdims=True))
        return _argmax(worst)
    # cost-form variants
    if model.startswith("shrink_c"):
        a0, c = np.array(th[0:2])[LV], np.full(10, np.exp(th[2]))
    else:
        a0, c = np.array(th[0:2])[LV], np.exp(np.array(th[2:4]))[LV]
    if model.endswith("_log"):
        return _argmax(rel - c[:, None] * (np.log(G) - np.log(a0[:, None])) ** 2)
    if model.endswith("_abs"):
        return _argmax(rr - c[:, None] * (G - a0[:, None]) ** 2)
    raise ValueError(model)


SPEC = {
    "ra": ([-1.0], [(-8, 6)]),
    "ra_L": ([-1.0, -1.0], [(-8, 6)] * 2),
    "robust": ([0.5], [(0.0, 3.0)]),
    "robust_L": ([0.5, 0.5], [(0.0, 3.0)] * 2),
    "shrink_c_log": ([1.5, 1.8, 0.0], [(0.3, 4.0)] * 2 + [(-6, 8)]),
    "shrink_log": ([1.5, 1.8, 0.0, 0.0], [(0.3, 4.0)] * 2 + [(-6, 8)] * 2),
    "shrink_c_abs": ([1.5, 1.8, 2.0], [(0.3, 4.0)] * 2 + [(-6, 10)]),
    "shrink_abs": ([1.5, 1.8, 2.0, 2.0], [(0.3, 4.0)] * 2 + [(-6, 10)] * 2),
}


def negll(x, model, data, iti):
    rt, acc, cell, lv = data
    v, t0 = np.array(x[0:2]), np.array(x[2:4])
    a = thresholds(model, x[4:], v, t0, iti)
    return -D.loglik_trials(rt, acc, v[lv], a[cell], t0[lv])


def fit_subject(sid, df, iti, starts=3):
    data = subject_data(df)
    rng = np.random.default_rng(int(sid) + 7)
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
    new.to_csv(os.path.join(RES, "alt_fits.csv"), index=False)
    m = pd.read_csv(os.path.join(RES, "mle_fits.csv")).merge(pd.read_csv(os.path.join(RES, "rr_fits.csv")), on="Subject").merge(new, on="Subject")
    models = ["free", "reg", "compress", "avers", "opt", "shrink", "shrink_c", "shrink_a0"] + list(SPEC)
    bic = m[[f"{k}_bic" for k in models]]; bic.columns = models
    tot = (bic.sum() - bic.sum().min()).sort_values()
    print("summed dBIC vs best:\n", tot.round(1))
    print("per-subject wins:\n", bic.idxmin(1).value_counts())
    B = bic.to_numpy(); rng = np.random.default_rng(0)
    win = [models[B[rng.integers(0, len(B), len(B))].sum(0).argmin()] for _ in range(2000)]
    print("bootstrap winner share:\n", pd.Series(win).value_counts(normalize=True).round(3))
    tot.rename("dBIC").to_csv(os.path.join(RES, "model_comparison_all.csv"))


if __name__ == "__main__":
    main()
