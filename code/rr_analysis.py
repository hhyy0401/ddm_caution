"""Analyses of the winning resource-rational model (shrink_c: one cost c, load-specific default a0_l).

1. parameters + whether load moves a0
2. does the flatter reward landscape under load reproduce the load-induced compression?
   (predicted slope of a vs log2 P, per load, from fitted shrink_c; and RR curvature at the optimum)
3. how much reward is sacrificed (RR fraction) vs how much it would cost to be optimal
4. parameter recovery: simulate trials from fitted parameters, refit
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
import sim
import fit_rr as F
from fit_mle import RES, R, PENS

LV = np.repeat([0, 1], 5)
PP = np.tile(PENS, 2)


def predicted(row, iti):
    v = np.array([row.shrink_c_p0, row.shrink_c_p1]); t0 = np.array([row.shrink_c_p2, row.shrink_c_p3])
    th = [row.shrink_c_p4, row.shrink_c_p5, row.shrink_c_p6]
    a = F.thresholds("shrink_c", th, v, t0, iti)
    ao = D.optimal_a(v[LV], t0[LV], iti[LV], R, PP)
    # relative curvature of RR at the optimum: -(d2 RR / da2) / RR*  (finite differences)
    h = 0.01
    f = lambda x: D.reward_rate(x, v[LV], t0[LV], iti[LV], R, PP)
    curv = -(f(ao + h) - 2 * f(ao) + f(ao - h)) / h ** 2 / f(ao)
    return a, ao, curv, v, t0


def sim_trials(v, t0, a_cells, iti, seed):
    """Trial-level simulation of one session with static thresholds (12 x 60 s blocks)."""
    th = np.zeros(13); th[0:2] = v; th[2:4] = t0; th[12] = -20.0
    rt = np.zeros(sim.MAXT); acc = np.zeros(sim.MAXT, np.int64); cell = np.zeros(sim.MAXT, np.int64)
    blk = np.zeros(sim.MAXT, np.int64); posn = np.zeros(sim.MAXT); prev = np.zeros(sim.MAXT, np.int64)
    n = sim._simulate_one(th, np.ascontiguousarray(a_cells), 0.001, seed, rt, acc, cell, blk, posn, prev)
    inv = {0: 5, 1: 10, 2: 20, 3: 40, 4: 80}
    df = pd.DataFrame({"rt": rt[:n], "acc": acc[:n], "load": cell[:n] // 5,
                       "Penalty": [inv[c % 5] for c in cell[:n]], "TaskBlock": blk[:n]})
    df["in_win"] = df.rt.between(D.RT_MIN, D.RT_MAX)
    return df


def recover_one(i, row, iti):
    a, _, _, v, t0 = predicted(row, iti)
    df = sim_trials(v, t0, a, iti, 1000 + i)
    out = F.fit_subject(int(row.Subject), df, iti, starts=3)
    return dict(true_v0=v[0], true_v1=v[1], true_a00=row.shrink_c_p4, true_a01=row.shrink_c_p5, true_logc=row.shrink_c_p6,
                rec_v0=out["shrink_c_p0"], rec_v1=out["shrink_c_p1"], rec_a00=out["shrink_c_p4"], rec_a01=out["shrink_c_p5"],
                rec_logc=out["shrink_c_p6"],
                dbic_c_vs_a0=out["shrink_c_bic"] - out["shrink_a0_bic"])


def main():
    f = pd.read_csv(os.path.join(RES, "rr_fits.csv"))
    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    rows = []
    for _, r in f.iterrows():
        iti = subj.loc[r.Subject, ["iti_noload", "iti_load"]].to_numpy(float)
        a, ao, curv, v, t0 = predicted(r, iti)
        sl = [np.polyfit(np.log2(PENS), a[5 * l:5 * l + 5], 1)[0] for l in (0, 1)]
        slo = [np.polyfit(np.log2(PENS), ao[5 * l:5 * l + 5], 1)[0] for l in (0, 1)]
        rr = D.reward_rate(a, v[LV], t0[LV], iti[LV], R, PP) / D.reward_rate(ao, v[LV], t0[LV], iti[LV], R, PP)
        rows.append(dict(Subject=r.Subject, a0_noload=r.shrink_c_p4, a0_load=r.shrink_c_p5, logc=r.shrink_c_p6,
                         slope_noload=sl[0], slope_load=sl[1], optslope_noload=slo[0], optslope_load=slo[1],
                         curv_noload=curv[:5].mean(), curv_load=curv[5:].mean(), rr_frac=rr.mean(),
                         rr_frac_noload=rr[:5].mean(), rr_frac_load=rr[5:].mean()))
    o = pd.DataFrame(rows); o.to_csv(os.path.join(RES, "rr_subject.csv"), index=False)
    print(o.drop(columns="Subject").median().round(3).to_string())
    print("a0 Load > NoLoad:", (o.a0_load > o.a0_noload).mean().round(3), "p =", wilcoxon(o.a0_load, o.a0_noload).pvalue)
    print("predicted slope Load < NoLoad (same c):", (o.slope_load < o.slope_noload).mean().round(3),
          "p =", wilcoxon(o.slope_load, o.slope_noload).pvalue)
    print("RR curvature Load < NoLoad:", (o.curv_load < o.curv_noload).mean().round(3))
    o["shrink_ratio_noload"] = o.slope_noload / o.optslope_noload
    o["shrink_ratio_load"] = o.slope_load / o.optslope_load
    print("slope / optimal slope median:", o.shrink_ratio_noload.median().round(3), o.shrink_ratio_load.median().round(3))
    # MLE free-model slopes for comparison (data-driven)
    oc = pd.read_csv(os.path.join(RES, "overcaution_subject.csv"))
    A = oc[[f"a{j}" for j in range(10)]].to_numpy()
    fs = np.array([[np.polyfit(np.log2(PENS), A[i, 5 * l:5 * l + 5], 1)[0] for l in (0, 1)] for i in range(len(A))])
    m = o.merge(pd.DataFrame({"Subject": oc.Subject, "free_slope_noload": fs[:, 0], "free_slope_load": fs[:, 1]}), on="Subject")
    print("free-model slopes median:", np.median(fs, 0).round(3),
          " corr(pred, free) NoLoad", round(spearmanr(m.slope_noload, m.free_slope_noload).statistic, 3),
          "Load", round(spearmanr(m.slope_load, m.free_slope_load).statistic, 3))

    # recovery on 60 subjects
    sel = f.sample(60, random_state=0)
    rec = pd.DataFrame(Parallel(n_jobs=10)(delayed(recover_one)(i, r, subj.loc[r.Subject, ["iti_noload", "iti_load"]].to_numpy(float))
                                           for i, (_, r) in enumerate(sel.iterrows())))
    rec.to_csv(os.path.join(RES, "rr_recovery.csv"), index=False)
    for p in ["v0", "v1", "a00", "a01", "logc"]:
        print(f"recovery {p}: r = {np.corrcoef(rec['true_' + p], rec['rec_' + p])[0, 1]:.3f}")
    print("model recovery: shrink_c preferred over shrink_a0 in", (rec.dbic_c_vs_a0 < 0).mean().round(3), "of simulated subjects")


if __name__ == "__main__":
    main()
