"""Summaries for selective.py and recover_final.py chunk outputs."""
import os, glob, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
from fit_mle import RES
rng = np.random.default_rng(0)
bs = lambda d: np.mean([d[rng.integers(0, len(d), len(d))].sum() > 0 for _ in range(2000)])

S = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(RES, "selective", "chunk_*.csv")))
print(f"[selective] N = {len(S)}")
ntr = pd.read_csv(os.path.join(RES, "trials.csv")).query("in_win").groupby("Subject").size()
n = S.Subject.map(ntr)
d = S.free14_nll - S.full30_nll
print(f"full30 vs free14: summed dlnL {d.sum():.1f}; sum dBIC (positive favors full30) {(2*d - 16*np.log(n)).sum():.1f}; "
      f"subjects favoring full30 by BIC {((2*d - 16*np.log(n)) > 0).mean():.3f}")
for c in ("v_slope", "t0_slope", "a_slope"):
    print(f"  {c}: median {S[c].median():.4f}  IQR {S[c].quantile(.25):.4f}..{S[c].quantile(.75):.4f}  Wilcoxon p {wilcoxon(S[c]).pvalue:.2g}")
r = S.a_slope / S.aopt_slope
print(f"under-adjustment with cell-specific v,t0: median slope ratio {r.median():.3f}, share < 1 {(r < 1).mean():.3f}")
A = S[[f"a_c{c}" for c in range(10)]].to_numpy().reshape(-1, 2, 5).mean(1); O = S[[f"aopt_c{c}" for c in range(10)]].to_numpy().reshape(-1, 2, 5).mean(1)
print("  median a by P:", np.median(A, 0).round(2), " optimal:", np.median(O, 0).round(2))
cv = pd.DataFrame({m: -S[f"{m}_cv"] for m in ("cost", "compress", "reg")})
print("held-out lnL (v,t0 linear in log P):", cv.sum().round(1).to_dict(), " best per subject:", cv.idxmax(1).value_counts().to_dict())
for m in ("compress", "reg"):
    dd = (cv.cost - cv[m]).to_numpy(); print(f"  cost - {m}: {dd.sum():.1f}  P(boot>0) {bs(dd):.3f}  share {np.mean(dd > 0):.3f}")
S.to_csv(os.path.join(RES, "selective_all.csv"), index=False)

R = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(RES, "recover_final", "chunk_*.csv")))
print(f"\n[recovery] rows {len(R)}")
names = ["v0", "v1", "t00", "t01", "a0_NL", "a0_L", "logc"]
c = R[R.gen == "shrink_c_abs"]
print("cost-model parameter recovery r:", {nm: round(np.corrcoef(c[f"true_p{j}"], c[f"rec_p{j}"])[0, 1], 3) for j, nm in enumerate(names)})
print("  rank r logc:", round(c.true_p6.rank().corr(c.rec_p6.rank()), 3), " load shift r:",
      round(np.corrcoef(c.true_p5 - c.true_p4, c.rec_p5 - c.rec_p4)[0, 1], 3))
for gen, g in R.groupby("gen"):
    cvg = pd.DataFrame({m: -g[f"{m}_cv"] for m in ("shrink_c_abs", "compress", "reg")})
    dd = (cvg.shrink_c_abs - cvg.compress).to_numpy()
    print(f"gen={gen}: held-out sums {cvg.sum().round(1).to_dict()}; cost-compress {dd.sum():.1f} (P boot>0 {bs(dd):.3f}); "
          f"subject-level best {cvg.idxmax(1).value_counts(normalize=True).round(3).to_dict()}")
R.to_csv(os.path.join(RES, "recover_final_all.csv"), index=False)
