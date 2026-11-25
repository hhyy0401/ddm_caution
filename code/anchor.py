"""What anchors threshold adjustment: a stable default or the previous block's threshold?

Blocks are processed in the order each participant did them. For block b with penalty P_b, load l_b:
    anchor_b = w * a_{b-1} + (1 - w) * a0_{l_b}            (block 1: anchor = a0_{l_1})
    a_b      = argmax_a  RR(a; P_b, v_l, t0_l, ITI_l) - c * (a - anchor_b)^2
Models: default (w = 0; = shrink_c_abs), previous (w = 1, Grahek et al.-style reconfiguration cost),
mixed (w free). Compared on full-data fit and leave-one-penalty-level-out CV.
Also a model-free check: does the previous block's penalty shift the current block's RT/accuracy?
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
from fit_mle import RES

G = D.A_GRID
PMAP = {5: 0, 10: 1, 20: 2, 40: 3, 80: 4}
MODELS = {"default": 0.0, "previous": 1.0, "mixed": None}


def block_data(df):
    blocks = df.groupby("TaskBlock").agg(P=("Penalty", "first"), load=("load", "first")).sort_index()
    order = {b: i for i, b in enumerate(blocks.index)}
    d = df[df.in_win]
    return dict(rt=d.rt.to_numpy(float), acc=d.acc.to_numpy(np.int64), bi=d.TaskBlock.map(order).to_numpy(),
                lv=d.load.to_numpy(), pidx=d.Penalty.map(PMAP).to_numpy(),
                bP=blocks.P.to_numpy(float), bL=blocks.load.to_numpy())


def block_thresholds(x, model, dat, iti):
    v, t0 = x[0:2], x[2:4]
    a0 = np.array(x[4:6]); c = np.exp(x[6])
    w = MODELS[model] if MODELS[model] is not None else x[7]
    out = np.empty(len(dat["bP"])); prev = None
    for b, (P, l) in enumerate(zip(dat["bP"], dat["bL"])):
        anchor = a0[l] if prev is None else w * prev + (1 - w) * a0[l]
        J = D.reward_rate(G, v[l], t0[l], iti[l], 20.0, P) - c * (G - anchor) ** 2
        k = int(np.clip(J.argmax(), 1, len(G) - 2))
        y0, y1, y2 = J[k - 1], J[k], J[k + 1]; den = y0 - 2 * y1 + y2
        out[b] = G[k] + (np.clip(0.5 * (y0 - y2) / den, -1, 1) if abs(den) > 1e-12 else 0) * (G[1] - G[0])
        prev = out[b]
    return out


def nll(x, model, dat, iti, mask):
    a_b = block_thresholds(x, model, dat, iti)
    v, t0 = np.array(x[0:2]), np.array(x[2:4])
    lv = dat["lv"][mask]
    return -D.loglik_trials(dat["rt"][mask], dat["acc"][mask], v[lv], a_b[dat["bi"][mask]], t0[lv])


def fit(model, dat, iti, mask, x0):
    tmin = [dat["rt"][dat["lv"] == l].min() for l in (0, 1)]
    bounds = [(0.01, 10)] * 2 + [(0.05, tmin[0] - 0.01), (0.05, tmin[1] - 0.01)] + [(0.3, 4.0)] * 2 + [(-6, 10)]
    if model == "mixed":
        bounds = bounds + [(0.0, 1.0)]
    lo = np.array([b[0] + 1e-4 for b in bounds]); hi = np.array([b[1] - 1e-4 for b in bounds])
    best = None
    for start in x0:
        s = np.clip(np.r_[start, 0.3][:len(bounds)] if model == "mixed" and len(start) == 7 else start[:len(bounds)], lo, hi)
        r = minimize(nll, s, args=(model, dat, iti, mask), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best


def run_subject(sid, df, iti, init):
    dat = block_data(df)
    allm = np.ones(len(dat["rt"]), bool)
    out = {"Subject": sid, "n": len(dat["rt"])}
    starts = [init, np.r_[init[:6], init[6] + 1.0], np.r_[init[:6], init[6] - 1.0]]
    for model in MODELS:
        st = [np.r_[s, w0] for s in starts[:2] for w0 in (0.2, 0.7)] if model == "mixed" else starts
        r = fit(model, dat, iti, allm, st)
        out[f"{model}_nll"] = r.fun
        out[f"{model}_k"] = len(r.x)
        if model == "mixed":
            out["mixed_w"] = r.x[7]
        cv = 0.0
        for k in range(5):
            train = dat["pidx"] != k
            rk = fit(model, dat, iti, train, [r.x])
            cv += nll(rk.x, model, dat, iti, ~train)
        out[f"{model}_cv"] = cv
    return out


def model_free(trials):
    b = trials[trials.in_win].groupby(["Subject", "TaskBlock"]).agg(
        P=("Penalty", "first"), load=("load", "first"), lrt=("rt", lambda x: np.log(x).median()),
        acc=("acc", "mean")).reset_index().sort_values(["Subject", "TaskBlock"])
    b["prevP"] = b.groupby("Subject").P.shift()
    b = b.dropna()
    b["cur"] = np.log2(b.P / 20); b["prev"] = np.log2(b.prevP / 20)
    rows = []
    for y in ("lrt", "acc"):
        d = b.copy()
        # within-subject: remove subject x current-condition means (so only previous-block effect remains)
        d[y + "_r"] = d[y] - d.groupby(["Subject", "P", "load"])[y].transform("mean")
        d["prev_r"] = d.prev - d.groupby(["Subject", "P", "load"]).prev.transform("mean")
        d["pos_r"] = d.TaskBlock - d.groupby(["Subject", "P", "load"]).TaskBlock.transform("mean")
        X = np.c_[d.prev_r, d.pos_r]; beta = np.linalg.lstsq(X, d[y + "_r"], rcond=None)[0]
        # subject-level bootstrap for the previous-penalty slope
        subs = d.Subject.unique(); rng = np.random.default_rng(0); bs = []
        g = {s: x for s, x in d.groupby("Subject")}
        for _ in range(1000):
            dd = pd.concat([g[s] for s in rng.choice(subs, len(subs))])
            bs.append(np.linalg.lstsq(np.c_[dd.prev_r, dd.pos_r], dd[y + "_r"], rcond=None)[0][0])
        rows.append(dict(outcome=y, prev_slope=beta[0], ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    # reference: effect of current penalty (between conditions) on the same scale
    cur = b.groupby("P")[["lrt", "acc"]].mean()
    return pd.DataFrame(rows), cur


def main():
    trials = pd.read_csv(os.path.join(RES, "trials.csv"))
    mf, cur = model_free(trials)
    print("model-free: effect of previous block's log2 penalty on current block (within subject x condition)")
    print(mf.round(4).to_string(index=False))
    print("for scale, mean log RT / acc by current penalty:\n", cur.round(4).T)

    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    alt = pd.read_csv(os.path.join(RES, "alt_fits.csv")).set_index("Subject")
    jobs = []
    for sid, g in trials.groupby("Subject"):
        init = np.array([alt.loc[sid, f"shrink_c_abs_p{j}"] for j in range(7)])
        jobs.append((sid, g, subj.loc[sid, ["iti_noload", "iti_load"]].to_numpy(float), init))
    if len(sys.argv) > 1:
        jobs = jobs[: int(sys.argv[1])]
    out = pd.DataFrame(Parallel(n_jobs=10)(delayed(run_subject)(*j) for j in jobs))
    out.to_csv(os.path.join(RES, "anchor_fits.csv"), index=False)
    for m in MODELS:
        out[f"{m}_bic"] = 2 * out[f"{m}_nll"] + out[f"{m}_k"] * np.log(out.n)
    print("\nfull-data: sum lnL", {m: round(-out[f"{m}_nll"].sum(), 1) for m in MODELS},
          "\nsum BIC", {m: round(out[f"{m}_bic"].sum(), 1) for m in MODELS})
    cv = pd.DataFrame({m: -out[f"{m}_cv"] for m in MODELS})
    print("held-out lnL (higher better):", cv.sum().round(1).to_dict())
    print("per-subject best (CV):", cv.idxmax(1).value_counts().to_dict())
    d = cv["default"] - cv["previous"]; dm = cv["mixed"] - cv["default"]
    bs = lambda x: np.mean([x.sample(len(x), replace=True, random_state=i).sum() > 0 for i in range(2000)])
    print(f"default - previous: {d.sum():.1f} (P>0 {bs(d):.3f});  mixed - default: {dm.sum():.1f} (P>0 {bs(dm):.3f})")
    print("mixed w: median %.3f, IQR %.3f-%.3f, share w>0.5 %.3f" % (out.mixed_w.median(), out.mixed_w.quantile(.25),
                                                                  out.mixed_w.quantile(.75), (out.mixed_w > .5).mean()))


if __name__ == "__main__":
    main()
