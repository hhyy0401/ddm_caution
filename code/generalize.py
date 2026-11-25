"""Do the findings generalize across the 358 participants?

Per-subject quantities are built from the saved per-subject fits; then for every finding:
  - subject bootstrap 95% CI (2000 resamples)
  - random split-half replication over subjects (1000 splits): share of splits where both halves
    show the same-signed effect (and, for correlations, both p < .05)
  - extreme groups: top vs bottom 25% on WMC, gF, AC, SuS_AC, task score, drift v.
    Does each finding hold inside both groups, and does its size differ (Mann-Whitney)?
Outputs results/generalize_findings.csv, generalize_groups.csv, generalize_subjects.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, rankdata, pearsonr
sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
from fit_mle import RES, R, PENS

LP = np.log2(np.array(PENS) / 20.0)
rng = np.random.default_rng(0)


def per_subject():
    mle = pd.read_csv(os.path.join(RES, "mle_fits.csv")).set_index("Subject")
    alt = pd.read_csv(os.path.join(RES, "alt_fits.csv")).set_index("Subject")
    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    cv = pd.read_csv(os.path.join(RES, "cv_penalty.csv")).set_index("Subject")
    cm = pd.read_csv(os.path.join(RES, "cv_metrics.csv"))
    an = pd.read_csv(os.path.join(RES, "anchor_fits.csv")).set_index("Subject")
    tr = pd.read_csv(os.path.join(RES, "trials.csv"))
    comp = pd.read_csv(os.path.join(RES, "composites.csv")).set_index("Subject")
    rows = []
    for s in mle.index:
        v = mle.loc[s, ["free_p0", "free_p1"]].to_numpy(float); t0 = mle.loc[s, ["free_p2", "free_p3"]].to_numpy(float)
        a = mle.loc[s, [f"free_p{j}" for j in range(4, 14)]].to_numpy(float).reshape(2, 5)
        iti = subj.loc[s, ["iti_noload", "iti_load"]].to_numpy(float)
        ao = np.array([D.optimal_a(np.full(5, v[l]), np.full(5, t0[l]), np.full(5, iti[l]), R, np.array(PENS, float)) for l in (0, 1)])
        rr = np.array([D.reward_rate(a[l], v[l], t0[l], iti[l], R, np.array(PENS, float)) for l in (0, 1)])
        rro = np.array([D.reward_rate(ao[l], v[l], t0[l], iti[l], R, np.array(PENS, float)) for l in (0, 1)])
        sl = np.polyfit(LP, a.T, 1)[0]; slo = np.polyfit(LP, ao.T, 1)[0]
        rows.append(dict(Subject=s, v=v.mean(), slope=sl.mean(), slope_opt=slo.mean(), adj_ratio=sl.mean() / slo.mean(),
                         rr_frac=(rr / rro).mean(), free_load_shift=a[1].mean() - a[0].mean(),
                         a0_noload=alt.loc[s, "shrink_c_abs_p4"], a0_load=alt.loc[s, "shrink_c_abs_p5"],
                         load_shift=alt.loc[s, "shrink_c_abs_p5"] - alt.loc[s, "shrink_c_abs_p4"], logc=alt.loc[s, "shrink_c_abs_p6"],
                         cv_cost_vs_compress=cv.loc[s, "compress_cv_nll"] - cv.loc[s, "shrink_c_abs_cv_nll"],
                         cv_cost_vs_reg=cv.loc[s, "reg_cv_nll"] - cv.loc[s, "shrink_c_abs_cv_nll"],
                         cv_cost_vs_costperload=cv.loc[s, "shrink_abs_cv_nll"] - cv.loc[s, "shrink_c_abs_cv_nll"],
                         anchor_default_vs_prev=an.loc[s, "previous_cv"] - an.loc[s, "default_cv"]))
    S = pd.DataFrame(rows).set_index("Subject")
    cm["ea"] = (cm.a_pred - cm.a_free).abs()
    e = cm.pivot_table(index="Subject", columns="model", values="ea")
    S["aerr_compress_minus_cost"] = e.compress - e.shrink_c_abs
    S["aerr_reg_minus_cost"] = e.reg - e.shrink_c_abs
    pts = tr.assign(p=np.where(tr.acc == 1, 20, -tr.Penalty)).groupby("Subject").p.sum()
    S["score"] = pts
    return S.join(comp)


# ---- findings: each is a function of a subject subset -> scalar (positive = finding holds) ----
def pcor(x, y, z):
    x, y = rankdata(x), rankdata(y); Z = np.c_[np.ones(len(x)), rankdata(z)]
    res = lambda u: u - Z @ np.linalg.lstsq(Z, u, rcond=None)[0]
    return pearsonr(res(x), res(y))

MEAN_FINDINGS = {  # name: (column, statistic, expected sign, description)
    "F1 under-adjustment (1 - slope/optimal slope)": ("adj_ratio", lambda x: 1 - np.median(x), "people raise a less than optimal"),
    "F2 points kept (rr_frac - 0.9)": ("rr_frac", lambda x: np.mean(x) - 0.9, "keep >90% of max reward rate"),
    "F3 CV: cost > compression (sum heldout lnL)": ("cv_cost_vs_compress", np.sum, ""),
    "F3 CV: cost > log-linear (sum heldout lnL)": ("cv_cost_vs_reg", np.sum, ""),
    "F3 CV: one cost > cost per load": ("cv_cost_vs_costperload", np.sum, ""),
    "F3 a-error: compression - cost": ("aerr_compress_minus_cost", np.mean, ""),
    "F4 load raises default a0": ("load_shift", np.mean, ""),
    "F4 load raises a (model-free)": ("free_load_shift", np.mean, ""),
    "F5 anchor: default > previous (CV)": ("anchor_default_vs_prev", np.sum, ""),
}
COR_FINDINGS = {"F6 logc ~ SuS_AC (neg)": ("logc", "SuS_AC", -1), "F6 logc ~ WMC (pos)": ("logc", "WMC", 1),
                "F6 a0_noload ~ AC (neg)": ("a0_noload", "AC", -1), "F6 a0_noload ~ WMC (neg)": ("a0_noload", "WMC", -1)}


def evaluate(S, B=2000, SPL=1000):
    out = []
    n = len(S); idx = np.arange(n)
    for name, (col, f, _) in MEAN_FINDINGS.items():
        x = S[col].dropna().to_numpy(); m = len(x)
        bs = np.array([f(x[rng.integers(0, m, m)]) for _ in range(B)])
        rep = []
        for _ in range(SPL):
            p = rng.permutation(m); h = m // 2
            rep.append(f(x[p[:h]]) > 0 and f(x[p[h:]]) > 0)
        share = np.mean(x > 0) if col not in ("adj_ratio", "rr_frac") else (np.mean(x < 1) if col == "adj_ratio" else np.mean(x > .9))
        out.append(dict(finding=name, estimate=f(x), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                        p_boot_le0=np.mean(bs <= 0), split_half_both=np.mean(rep), share_subjects_consistent=share, n=m))
    for name, (p, c, sgn) in COR_FINDINGS.items():
        d = S[[p, c, "v"]].dropna(); m = len(d)
        r, pv = pcor(d[p], d[c], d.v)
        bs = []
        for _ in range(B):
            b = d.iloc[rng.integers(0, m, m)]; bs.append(pcor(b[p], b[c], b.v)[0])
        same, sig = [], []
        for _ in range(SPL):
            q = rng.permutation(m); h = m // 2
            ra = pcor(d.iloc[q[:h]][p], d.iloc[q[:h]][c], d.iloc[q[:h]].v); rb = pcor(d.iloc[q[h:]][p], d.iloc[q[h:]][c], d.iloc[q[h:]].v)
            same.append(np.sign(ra[0]) == sgn and np.sign(rb[0]) == sgn); sig.append(same[-1] and ra[1] < .05 and rb[1] < .05)
        out.append(dict(finding=name + " [individual est.]", estimate=r, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                        p_boot_le0=np.mean(np.sign(bs) != sgn), split_half_both=np.mean(same), split_half_both_sig=np.mean(sig), n=m))
    return pd.DataFrame(out)


GROUP_VARS = ["WMC", "gF", "AC", "SuS_AC", "score", "v"]
GROUP_METRICS = ["adj_ratio", "rr_frac", "cv_cost_vs_compress", "cv_cost_vs_reg", "aerr_compress_minus_cost",
                 "load_shift", "free_load_shift", "anchor_default_vs_prev", "logc", "a0_noload", "a0_load", "slope"]


def groups(S):
    rows = []
    for g in GROUP_VARS:
        d = S.dropna(subset=[g]); lo, hi = d[g].quantile([.25, .75])
        L, H = d[d[g] <= lo], d[d[g] >= hi]
        for mtr in GROUP_METRICS:
            a, b = H[mtr].dropna(), L[mtr].dropna()
            agg = np.sum if mtr.startswith(("cv_", "anchor")) else np.median
            rows.append(dict(group_var=g, metric=mtr, n_top=len(a), n_bottom=len(b), top=agg(a), bottom=agg(b),
                             top_share_pos=np.mean(a > 0) if not mtr in ("adj_ratio",) else np.mean(a < 1),
                             bottom_share_pos=np.mean(b > 0) if not mtr in ("adj_ratio",) else np.mean(b < 1),
                             mw_p=mannwhitneyu(a, b).pvalue))
    t = pd.DataFrame(rows)
    p = t.mw_p.to_numpy(); o = np.argsort(p); m = len(p)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; t.loc[t.index[o], "q"] = np.minimum(q, 1)
    return t


def main():
    S = per_subject(); S.to_csv(os.path.join(RES, "generalize_subjects.csv"))
    F = evaluate(S); F.to_csv(os.path.join(RES, "generalize_findings.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(F.round(3).to_string(index=False))
    G = groups(S); G.to_csv(os.path.join(RES, "generalize_groups.csv"), index=False)
    print(G.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
