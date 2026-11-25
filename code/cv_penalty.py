"""Leave-one-penalty-level-out cross-validation.

For each subject and each penalty level P (5 folds), fit a model on the trials of the other four
penalty levels (both load conditions) and score the held-out trials by their log-likelihood.
Models: shrink_c_abs (adjustment cost, 7 par), shrink_abs (8 par), compress (penalty compression, 8 par),
reg (log-linear, 8 par).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
import fit_mle as M
import fit_alt as A
from fit_mle import subject_data, RES

MODELS = {"shrink_c_abs": A, "shrink_abs": A, "compress": M, "reg": M}
SPEC = {"shrink_c_abs": A.SPEC["shrink_c_abs"], "shrink_abs": A.SPEC["shrink_abs"],
        "compress": M.TH_SPEC["compress"], "reg": M.TH_SPEC["reg"]}


def nll_masked(x, model, data, iti, mask):
    rt, acc, cell, lv = data
    v, t0 = np.array(x[0:2]), np.array(x[2:4])
    a = MODELS[model].thresholds(model, x[4:], v, t0, iti)
    if not np.all(np.isfinite(a)):
        return 1e10
    return -D.loglik_trials(rt[mask], acc[mask], v[lv[mask]], a[cell[mask]], t0[lv[mask]])


def fit(model, data, iti, mask, x_init, rng):
    tmin = [data[0][data[3] == l].min() for l in (0, 1)]
    init, b_th = SPEC[model]
    bounds = [(0.01, 10)] * 2 + [(0.05, tmin[0] - 0.01), (0.05, tmin[1] - 0.01)] + b_th
    lo = np.array([b[0] + 1e-3 for b in bounds]); hi = np.array([b[1] - 1e-3 for b in bounds])
    best = None
    starts = [np.clip(x_init, lo, hi), np.clip(np.r_[2.0, 1.6, 0.32, 0.36, init], lo, hi)]
    for x0 in starts:
        r = minimize(nll_masked, x0, args=(model, data, iti, mask), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def run_subject(sid, df, iti, full):
    data = subject_data(df)
    pidx = data[2] % 5
    rng = np.random.default_rng(int(sid))
    out = {"Subject": sid}
    for model in MODELS:
        n = 4 + len(SPEC[model][0])
        x_full = np.array([full[f"{model}_p{j}"] for j in range(n)])
        tot = 0.0
        for k in range(5):
            train = pidx != k
            x = fit(model, data, iti, train, x_full, rng)
            tot += nll_masked(x, model, data, iti, ~train)
            out[f"{model}_heldout_P{k}"] = nll_masked(x, model, data, iti, ~train)
        out[f"{model}_cv_nll"] = tot
    return out


def main():
    trials = pd.read_csv(os.path.join(RES, "trials.csv"))
    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    full = pd.read_csv(os.path.join(RES, "mle_fits.csv")).merge(pd.read_csv(os.path.join(RES, "alt_fits.csv")), on="Subject").set_index("Subject")
    jobs = [(sid, g, subj.loc[sid, ["iti_noload", "iti_load"]].to_numpy(float), full.loc[sid]) for sid, g in trials.groupby("Subject")]
    if len(sys.argv) > 1:
        jobs = jobs[: int(sys.argv[1])]
    out = pd.DataFrame(Parallel(n_jobs=10)(delayed(run_subject)(*j) for j in jobs))
    out.to_csv(os.path.join(RES, "cv_penalty.csv"), index=False)
    cv = out[[f"{m}_cv_nll" for m in MODELS]]; cv.columns = list(MODELS)
    print("held-out log-likelihood, summed (higher = better predictions):")
    print((-cv.sum()).round(1).sort_values(ascending=False))
    print("per-subject best:", cv.idxmin(1).value_counts().to_dict())
    d = cv["compress"] - cv["shrink_c_abs"]
    bs = [d.sample(len(d), replace=True, random_state=i).sum() for i in range(2000)]
    print(f"cost minus compress (held-out lnL): {d.sum():.1f}, bootstrap P(cost better) = {np.mean(np.array(bs) > 0):.3f}")
    for k, P in enumerate([5, 10, 20, 40, 80]):
        h = {m: -out[f"{m}_heldout_P{k}"].sum() for m in MODELS}
        print(f"  held-out P={P}:", {m: round(v, 1) for m, v in h.items()})


if __name__ == "__main__":
    main()
