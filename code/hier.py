"""Hierarchical (empirical-Bayes) fitting of resource-rational models: EM with Laplace approximation
(Huys et al., 2011 style). Group prior phi_i ~ N(mu, diag(s2)) on transformed parameters.

phi layout: [log v0, log v1, t00, t01, <threshold params>]
  shrink    : log a0_0, log a0_1, logc0, logc1
  shrink_c  : log a0_0, log a0_1, logc
  shrink_a0 : log a0, logc0, logc1
Model evidence: integrated BIC, iBIC = -2 sum_i log p(D_i | mu, s2) + 2 d log(N_trials).

usage: python hier.py real          fit the three models to the 358 participants
       python hier.py recover       simulate from fitted shrink_c and shrink_a0 groups, refit both
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
import fit_rr as F
import fit_alt as A
from fit_mle import subject_data, RES

MODELS = {"shrink": 4, "shrink_c": 3, "shrink_a0": 3, "shrink_c_abs": 3}
LOGA0 = (np.log(0.3), np.log(4.0))


def to_x(model, phi):
    v = np.exp(phi[0:2]); t0 = phi[2:4]; p = phi[4:]
    if model == "shrink":
        th = [np.exp(p[0]), np.exp(p[1]), p[2], p[3]]
    elif model in ("shrink_c", "shrink_c_abs"):
        th = [np.exp(p[0]), np.exp(p[1]), p[2]]
    else:
        th = [np.exp(p[0]), p[1], p[2]]
    return v, t0, th


def bounds(model, tmin):
    b = [(np.log(0.05), np.log(10))] * 2 + [(0.05, tmin[0] - 0.01), (0.05, tmin[1] - 0.01)]
    k = MODELS[model]
    na0 = 2 if model in ("shrink", "shrink_c", "shrink_c_abs") else 1
    hi = 10 if model.endswith("_abs") else 8
    return b + [LOGA0] * na0 + [(-6, hi)] * (k - na0)


def nll(phi, model, data, iti):
    rt, acc, cell, lv = data
    v, t0, th = to_x(model, phi)
    a = (A if model.endswith("_abs") else F).thresholds(model, th, v, t0, iti)
    return -D.loglik_trials(rt, acc, v[lv], a[cell], t0[lv])


def map_fit(model, data, iti, mu, s2, x0):
    tmin = [data[0][data[3] == l].min() for l in (0, 1)]
    bnd = bounds(model, tmin)
    obj = lambda p: nll(p, model, data, iti) + 0.5 * np.sum((p - mu) ** 2 / s2)
    x0 = np.clip(x0, [b[0] + 1e-4 for b in bnd], [b[1] - 1e-4 for b in bnd])
    best = None
    for start in (x0, np.clip(mu, [b[0] + 1e-4 for b in bnd], [b[1] - 1e-4 for b in bnd])):
        r = minimize(obj, start, method="L-BFGS-B", bounds=bnd)
        if best is None or r.fun < best.fun:
            best = r
    m = best.x
    d = len(m); h = 1e-3 * np.maximum(1, np.abs(m)); H = np.zeros((d, d)); f0 = best.fun
    for i in range(d):
        for j in range(i, d):
            ei = np.zeros(d); ej = np.zeros(d); ei[i] = h[i]; ej[j] = h[j]
            H[i, j] = H[j, i] = (obj(m + ei + ej) - obj(m + ei - ej) - obj(m - ei + ej) + obj(m - ei - ej)) / (4 * h[i] * h[j])
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, 1.0 / s2.max() * 0.5)          # keep the Hessian positive definite
    Hpd = (V * w) @ V.T
    S = np.linalg.inv(Hpd)
    logev = -f0 + 0.5 * d * np.log(2 * np.pi) - 0.5 * np.sum(np.log(w)) - 0.5 * np.sum(np.log(2 * np.pi * s2))
    return m, np.diag(S), logev


def fit_hier(model, datasets, phi0, n_iter=12, tol=1e-3, verbose=True):
    """datasets: list of (sid, data_tuple, iti). phi0: (N, d) initial per-subject values."""
    mu = phi0.mean(0); s2 = phi0.var(0) + 0.05
    m = phi0.copy()
    for it in range(n_iter):
        res = Parallel(n_jobs=int(os.environ.get("NJOBS", 10)))(delayed(map_fit)(model, dat, iti, mu, s2, m[i])
                                  for i, (_, dat, iti) in enumerate(datasets))
        m = np.array([r[0] for r in res]); Sd = np.array([r[1] for r in res]); ev = np.array([r[2] for r in res])
        mu_new = m.mean(0); s2_new = (m ** 2 + Sd).mean(0) - mu_new ** 2
        s2_new = np.maximum(s2_new, 1e-4)
        delta = np.max(np.abs(mu_new - mu))
        mu, s2 = mu_new, s2_new
        if verbose:
            print(f"  [{model}] iter {it}: sum log ev {ev.sum():.1f}  max d mu {delta:.4f}", flush=True)
        if delta < tol and it > 2:
            break
    ntr = sum(len(dat[0]) for _, dat, _ in datasets)
    ibic = -2 * ev.sum() + 2 * len(mu) * np.log(ntr)
    return dict(mu=mu, s2=s2, m=m, Sd=Sd, ev=ev, ibic=ibic)


def init_from_mle(model, sids):
    f = pd.read_csv(os.path.join(RES, "rr_fits.csv")).set_index("Subject").loc[sids]
    p = lambda j: f[f"{model}_p{j}"].to_numpy()
    cols = [np.log(p(0)), np.log(p(1)), p(2), p(3)]
    if model == "shrink":
        cols += [np.log(p(4)), np.log(p(5)), p(6), p(7)]
    elif model == "shrink_c":
        cols += [np.log(p(4)), np.log(p(5)), p(6)]
    else:
        cols += [np.log(p(4)), p(5), p(6)]
    return np.clip(np.column_stack(cols), -6, 8)


def load_real():
    trials = pd.read_csv(os.path.join(RES, "trials.csv"))
    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    return [(sid, subject_data(g), subj.loc[sid, ["iti_noload", "iti_load"]].to_numpy(float))
            for sid, g in trials.groupby("Subject")]


def real():
    ds = load_real(); sids = [d[0] for d in ds]
    out = {}
    for model in MODELS:
        t = time.time()
        r = fit_hier(model, ds, init_from_mle(model, sids))
        out[model] = r
        np.savez(os.path.join(RES, f"hier_{model}.npz"), sids=np.array(sids), **{k: v for k, v in r.items()})
        print(f"{model}: iBIC {r['ibic']:.1f}  ({time.time() - t:.0f}s)  mu {np.round(r['mu'], 3)}  sd {np.round(np.sqrt(r['s2']), 3)}", flush=True)
    best = min(out, key=lambda k: out[k]["ibic"])
    print("dIBIC:", {k: round(v["ibic"] - out[best]["ibic"], 1) for k, v in out.items()})
    r = out["shrink"]
    dc = r["m"][:, 7] - r["m"][:, 6]
    print(f"[shrink] group logc NoLoad {r['mu'][6]:.3f}, Load {r['mu'][7]:.3f}; subjects with c_Load > c_NoLoad: {(dc > 0).mean():.3f}")


def recover():
    """Simulate a synthetic cohort from each fitted group model (per-subject posterior means), refit both."""
    from rr_analysis import sim_trials
    ds = load_real(); sids = [d[0] for d in ds]; itis = [d[2] for d in ds]
    rows = []
    for gen in ("shrink_c", "shrink_a0"):
        g = np.load(os.path.join(RES, f"hier_{gen}.npz"))
        truth = g["m"]
        sims = []
        for i, sid in enumerate(sids):
            v, t0, th = to_x(gen, truth[i])
            a = F.thresholds(gen, th, v, t0, itis[i])
            df = sim_trials(v, t0, a, itis[i], 5000 + i)
            sims.append((sid, subject_data(df), itis[i]))
        fits = {}
        for fm in ("shrink_c", "shrink_a0"):
            init = init_from_mle(fm, sids)
            fits[fm] = fit_hier(fm, sims, init, verbose=False)
        win = min(fits, key=lambda k: fits[k]["ibic"])
        rec = fits[gen]["m"]
        r_par = [np.corrcoef(truth[:, j], rec[:, j])[0, 1] for j in range(truth.shape[1])]
        # subject-level choice by Laplace evidence
        subj_win = (fits[gen]["ev"] > fits["shrink_a0" if gen == "shrink_c" else "shrink_c"]["ev"]).mean()
        rows.append(dict(generator=gen, group_winner=win,
                         dIBIC=fits["shrink_a0"]["ibic"] - fits["shrink_c"]["ibic"],
                         subject_level_correct=subj_win, **{f"r_p{j}": r for j, r in enumerate(r_par)}))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "hier_recovery.csv"), index=False)


if __name__ == "__main__":
    {"real": real, "recover": recover}[sys.argv[1]]()
