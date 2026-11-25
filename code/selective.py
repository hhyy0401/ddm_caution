"""Selective-influence check: does the penalty change v and t0 too, and do the conclusions survive?

Per subject:
  free14 : v, t0 per load; a per cell (assumed so far)
  full30 : v, t0, a per cell (fit cell by cell)            -> LRT/BIC vs free14, slopes of v, t0 vs log2 P
  Under-adjustment recomputed with cell-specific v, t0 (full30): slope of a vs optimal a* slope.
  Policies with v, t0 linear in log2(P/20) per load (8 nuisance params), leave-one-penalty-out CV:
     cost (a0 per load, one c), compress (lam, gamma per load), reg (alpha, beta per load)
usage: python selective.py CHUNK NCHUNK
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
from fit_alt import _argmax
from fit_mle import subject_data, RES, R, PENS

LV = np.repeat([0, 1], 5); PP = np.tile(np.array(PENS, float), 2); X = np.log2(PP / 20)
G = D.A_GRID[None, :]
NTH = {"cost": 3, "compress": 4, "reg": 4}


def nuis(x):
    v = np.exp(x[0:2][LV] + x[2:4][LV] * X); t0 = x[4:6][LV] + x[6:8][LV] * X
    return v, t0


def thr(model, th, v, t0, iti):
    ii = iti[LV]
    if model == "reg":
        return np.exp(np.array(th[0:2])[LV] + np.array(th[2:4])[LV] * X)
    if model == "compress":
        lam, gam = np.exp(np.array(th[0:2]))[LV], np.array(th[2:4])[LV]
        return D.optimal_a(v, t0, ii, R, lam * 20 * (PP / 20) ** gam)
    a0, c = np.array(th[0:2])[LV], np.exp(th[2])
    rr = D.reward_rate(G, v[:, None], t0[:, None], ii[:, None], R, PP[:, None])
    return _argmax(rr - c * (G - a0[:, None]) ** 2)


def nll(x, model, data, iti, mask, tmin):
    rt, acc, cell, lv = data
    v, t0 = nuis(x)
    if np.any(t0 < 0.05) or np.any(t0 > tmin - 0.005) or np.any(v > 15):
        return 1e10
    a = thr(model, x[8:], v, t0, iti)
    if not np.all(np.isfinite(a)):
        return 1e10
    m = mask
    return -D.loglik_trials(rt[m], acc[m], v[cell[m]], a[cell[m]], t0[cell[m]])


TH0 = {"cost": [1.6, 1.9, 1.0], "compress": [0.0, 0.0, 0.5, 0.5], "reg": [0.4, 0.6, 0.1, 0.1]}
THB = {"cost": [(0.3, 4.0)] * 2 + [(-6, 10)], "compress": [(-4, 4)] * 2 + [(-1, 3)] * 2, "reg": [(-2, 2)] * 2 + [(-1, 1)] * 2}


def fit_policy(model, data, iti, mask, tmin, x0s):
    b = [(-3, 2.5)] * 2 + [(-1, 1)] * 2 + [(0.05, 1.0)] * 2 + [(-0.2, 0.2)] * 2 + THB[model]
    lo = np.array([q[0] for q in b]) + 1e-4; hi = np.array([q[1] for q in b]) - 1e-4
    best = None
    for x0 in x0s:
        r = minimize(nll, np.clip(x0, lo, hi), args=(model, data, iti, mask, tmin), method="L-BFGS-B", bounds=b)
        if best is None or r.fun < best.fun:
            best = r
    return best


def cell_fit(rt, acc, a0):
    tmin = rt.min()
    f = lambda p: -D.loglik_trials(rt, acc, np.full(len(rt), p[0]), np.full(len(rt), p[1]), np.full(len(rt), p[2]))
    best = None
    for s in ([1.8, a0, 0.3], [1.2, a0 * 1.2, 0.25]):
        r = minimize(f, np.clip(s, [0.02, 0.3, 0.06], [9, 4.9, tmin - 0.02]), method="L-BFGS-B",
                     bounds=[(0.01, 10), (0.2, 5.0), (0.05, tmin - 0.01)])
        if best is None or r.fun < best.fun:
            best = r
    return best


def run(sid, df, iti, mle):
    data = subject_data(df); rt, acc, cell, lv = data
    out = {"Subject": sid, "free14_nll": mle["free_nll"]}
    # full30
    tot = 0; vv = np.zeros(10); tt = np.zeros(10); aa = np.zeros(10)
    for c in range(10):
        m = cell == c
        r = cell_fit(rt[m], acc[m], mle[f"free_p{4 + c}"]); tot += r.fun; vv[c], aa[c], tt[c] = r.x
    out["full30_nll"] = tot
    ao = D.optimal_a(vv, tt, iti[LV], R, PP)
    for c in range(10):
        out[f"v_c{c}"], out[f"t0_c{c}"], out[f"a_c{c}"], out[f"aopt_c{c}"] = vv[c], tt[c], aa[c], ao[c]
    sl = lambda y: np.mean([np.polyfit(X[:5], y[l * 5:(l + 1) * 5], 1)[0] for l in (0, 1)])
    out.update(v_slope=sl(np.log(vv)), t0_slope=sl(tt), a_slope=sl(aa), aopt_slope=sl(ao))
    # policies with linear-in-log-penalty nuisance, CV
    tmin = np.array([rt[cell == c].min() for c in range(10)])
    v0 = np.array([mle["free_p0"], mle["free_p1"]]); t00 = np.array([mle["free_p2"], mle["free_p3"]])
    base = np.r_[np.log(v0), 0, 0, t00, 0, 0]
    allm = np.ones(len(rt), bool)
    for model in NTH:
        r = fit_policy(model, data, iti, allm, tmin, [np.r_[base, TH0[model]]])
        out[f"{model}_nll"] = r.fun
        cv = 0
        for k in range(5):
            tr = (cell % 5) != k
            rk = fit_policy(model, data, iti, tr, tmin, [r.x])
            h = nll(rk.x, model, data, iti, ~tr, tmin); cv += h; out[f"{model}_heldout_P{k}"] = h
        out[f"{model}_cv"] = cv
    return out


def main():
    k, n = int(sys.argv[1]), int(sys.argv[2])
    trials = pd.read_csv(os.path.join(RES, "trials.csv")); subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    mle = pd.read_csv(os.path.join(RES, "mle_fits.csv")).set_index("Subject")
    sids = sorted(trials.Subject.unique())[k::n]
    jobs = [(s, trials[trials.Subject == s], subj.loc[s, ["iti_noload", "iti_load"]].to_numpy(float), mle.loc[s]) for s in sids]
    out = pd.DataFrame(Parallel(n_jobs=int(os.environ.get("NJOBS", 8)))(delayed(run)(*j) for j in jobs))
    os.makedirs(os.path.join(RES, "selective"), exist_ok=True)
    out.to_csv(os.path.join(RES, "selective", f"chunk_{k}.csv"), index=False)


if __name__ == "__main__":
    main()
