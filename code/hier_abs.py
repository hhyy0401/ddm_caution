"""Hierarchical (EM-Laplace) fit of the final adjustment-cost model (shrink_c_abs) and
individual differences with the shrunk estimates.

phi = [ln v0, ln v1, t00, t01, ln a0_noload, ln a0_load, ln c]
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
import hier as H
import fit_alt as A
from fit_mle import RES

MODEL = "shrink_c_abs"


def init(sids):
    f = pd.read_csv(os.path.join(RES, "alt_fits.csv")).set_index("Subject").loc[sids]
    p = lambda j: f[f"{MODEL}_p{j}"].to_numpy()
    return np.clip(np.column_stack([np.log(p(0)), np.log(p(1)), p(2), p(3), np.log(p(4)), np.log(p(5)), p(6)]), -6, 10)


def pcor(x, y, z):
    x, y = rankdata(x), rankdata(y)
    Z = np.c_[np.ones(len(x)), rankdata(z)]
    res = lambda u: u - Z @ np.linalg.lstsq(Z, u, rcond=None)[0]
    return pearsonr(res(x), res(y))


def bh(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1); return out


def main():
    ds = H.load_real(); sids = [d[0] for d in ds]
    r = H.fit_hier(MODEL, ds, init(sids))
    np.savez(os.path.join(RES, "hier_shrink_c_abs.npz"), sids=np.array(sids), **r)
    print("group mean:", dict(zip(["v0", "v1", "t00", "t01", "a0_NL", "a0_L", "c"],
                                   np.round([np.exp(r["mu"][0]), np.exp(r["mu"][1]), r["mu"][2], r["mu"][3],
                                             np.exp(r["mu"][4]), np.exp(r["mu"][5]), np.exp(r["mu"][6])], 3))))
    m = r["m"]
    hier = pd.DataFrame({"Subject": sids, "v": (m[:, 0] + m[:, 1]) / 2, "a0_noload": np.exp(m[:, 4]),
                         "a0_load": np.exp(m[:, 5]), "load_shift": np.exp(m[:, 5]) - np.exp(m[:, 4]), "logc": m[:, 6]})
    hier.to_csv(os.path.join(RES, "hier_abs_subjects.csv"), index=False)
    print(f"a0 Load > NoLoad: {(hier.load_shift > 0).mean():.3f}")

    ind = pd.read_csv(os.path.join(RES, "alt_fits.csv"))[["Subject", "shrink_c_abs_p0", "shrink_c_abs_p1", "shrink_c_abs_p4",
                                                          "shrink_c_abs_p5", "shrink_c_abs_p6"]]
    ind = pd.DataFrame({"Subject": ind.Subject, "v": np.log(ind.shrink_c_abs_p0 * ind.shrink_c_abs_p1) / 2,
                        "a0_noload": ind.shrink_c_abs_p4, "a0_load": ind.shrink_c_abs_p5,
                        "load_shift": ind.shrink_c_abs_p5 - ind.shrink_c_abs_p4, "logc": ind.shrink_c_abs_p6})
    comp = pd.read_csv(os.path.join(os.path.dirname(RES), "data", "all_Scores_filtered_composites.csv"))
    for name, est in (("individual", ind), ("hierarchical", hier)):
        d = est.merge(comp, on="Subject")
        rows = []
        for p in ["logc", "a0_noload", "a0_load", "load_shift"]:
            for c in ["WMC", "gF", "AC", "SuS_AC"]:
                rr, pv = pcor(d[p], d[c], d.v)
                rows.append(dict(param=p, comp=c, r=rr, p=pv))
        t = pd.DataFrame(rows); t["q"] = bh(t.p)
        t.to_csv(os.path.join(RES, f"inddiff_{name}_abs.csv"), index=False)
        print(f"\n[{name}] N = {len(d)}  partial Spearman r (controlling v):")
        print(t.pivot(index="param", columns="comp", values="r").round(3))
        print("FDR q < .05:\n", t[t.q < .05].round(4).to_string(index=False) if (t.q < .05).any() else "  none")


if __name__ == "__main__":
    main()
