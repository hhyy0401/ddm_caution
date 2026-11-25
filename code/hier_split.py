"""Split-half replication of the hierarchical individual-difference results.
Task id k in 0..9: split seed = k // 2, half = k % 2. Fits shrink_c_abs hierarchically on that half alone
(its own group prior) and reports partial Spearman (controlling v) with the composites on that half."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import hier as H
from hier_abs import init, pcor
from fit_mle import RES

k = int(sys.argv[1]); seed, half = k // 2, k % 2
ds = H.load_real(); sids = np.array([d[0] for d in ds])
perm = np.random.default_rng(100 + seed).permutation(len(ds)); h = len(ds) // 2
take = np.sort(perm[:h] if half == 0 else perm[h:])
sub = [ds[i] for i in take]; ss = sids[take]
r = H.fit_hier("shrink_c_abs", sub, init(list(ss)), verbose=True)
m = r["m"]
est = pd.DataFrame({"Subject": ss, "v": (m[:, 0] + m[:, 1]) / 2, "a0_noload": np.exp(m[:, 4]), "a0_load": np.exp(m[:, 5]),
                    "load_shift": np.exp(m[:, 5]) - np.exp(m[:, 4]), "logc": m[:, 6]})
comp = pd.read_csv(os.path.join(RES, "composites.csv"))
d = est.merge(comp, on="Subject")
rows = []
for p in ["logc", "a0_noload", "a0_load", "load_shift"]:
    for c in ["WMC", "gF", "AC", "SuS_AC"]:
        rr, pv = pcor(d[p], d[c], d.v); rows.append(dict(seed=seed, half=half, param=p, comp=c, r=rr, p=pv, n=len(d)))
rows.append(dict(seed=seed, half=half, param="group_load_shift", comp="", r=np.exp(r["mu"][5]) - np.exp(r["mu"][4]),
                 p=np.mean(est.load_shift > 0), n=len(d)))
os.makedirs(os.path.join(RES, "hier_split"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "hier_split", f"split_{k}.csv"), index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
