"""Parameter and model recovery for the final adjustment-cost policy (shrink_c_abs) vs penalty compression.

For each real subject, simulate a synthetic session (12 x 60-s blocks) from
  gen = cost      : that subject's shrink_c_abs fit
  gen = compress  : that subject's compress fit
then on each synthetic dataset fit shrink_c_abs, compress, reg (full data: parameter recovery)
and run leave-one-penalty-out CV (model recovery with the same criterion as the real data).
usage: python recover_final.py CHUNK NCHUNK
"""
import os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
import cv_penalty as C
from rr_analysis import sim_trials
from fit_mle import subject_data, RES

FITM = ["shrink_c_abs", "compress", "reg"]


def params(full, model):
    n = 4 + len(C.SPEC[model][0]); return np.array([full[f"{model}_p{j}"] for j in range(n)])


def run(i, sid, iti, full):
    rows = []
    for g, gen in enumerate(("shrink_c_abs", "compress")):
        x = params(full, gen); v, t0 = x[0:2], x[2:4]
        a = C.MODELS[gen].thresholds(gen, x[4:], v, t0, iti)
        df = sim_trials(v, t0, np.ascontiguousarray(a, dtype=float), iti, 70000 + 10 * i + g)
        data = subject_data(df); pidx = data[2] % 5
        out = {"Subject": sid, "gen": gen, "n": len(data[0])}
        for j, val in enumerate(x):
            out[f"true_p{j}"] = val
        for m in FITM:
            xf = C.fit(m, data, iti, np.ones(len(data[0]), bool), params(full, m), None)
            out[f"{m}_nll"] = C.nll_masked(xf, m, data, iti, np.ones(len(data[0]), bool))
            if m == gen:
                for j, val in enumerate(xf):
                    out[f"rec_p{j}"] = val
            cv = 0
            for k in range(5):
                tr = pidx != k
                xk = C.fit(m, data, iti, tr, xf, None); cv += C.nll_masked(xk, m, data, iti, ~tr)
            out[f"{m}_cv"] = cv
        rows.append(out)
    return rows


def main():
    k, n = int(sys.argv[1]), int(sys.argv[2])
    subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
    full = pd.read_csv(os.path.join(RES, "mle_fits.csv")).merge(pd.read_csv(os.path.join(RES, "alt_fits.csv")), on="Subject").set_index("Subject")
    sids = sorted(full.index)
    jobs = [(i, s, subj.loc[s, ["iti_noload", "iti_load"]].to_numpy(float), full.loc[s]) for i, s in enumerate(sids)][k::n]
    res = Parallel(n_jobs=int(os.environ.get("NJOBS", 8)))(delayed(run)(*j) for j in jobs)
    os.makedirs(os.path.join(RES, "recover_final"), exist_ok=True)
    pd.DataFrame([r for rs in res for r in rs]).to_csv(os.path.join(RES, "recover_final", f"chunk_{k}.csv"), index=False)


if __name__ == "__main__":
    main()
