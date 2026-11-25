"""Figures for the CogSci paper."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import ddm_core as D
import fit_alt as A
from fit_mle import RES, R, PENS
OUT = os.path.join(os.path.dirname(RES), "paper", "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
P = np.array(PENS, float); x = np.log2(P)
mle = pd.read_csv(os.path.join(RES, "mle_fits.csv")).set_index("Subject")
alt = pd.read_csv(os.path.join(RES, "alt_fits.csv")).set_index("Subject")
subj = pd.read_csv(os.path.join(RES, "subjects.csv"), index_col=0)
S = pd.read_csv(os.path.join(RES, "generalize_subjects.csv")).set_index("Subject")
act, opt, cost = [], [], []
for s in mle.index:
    v = mle.loc[s, ["free_p0", "free_p1"]].to_numpy(float); t0 = mle.loc[s, ["free_p2", "free_p3"]].to_numpy(float)
    iti = subj.loc[s, ["iti_noload", "iti_load"]].to_numpy(float)
    act.append(mle.loc[s, [f"free_p{j}" for j in range(4, 14)]].to_numpy(float).reshape(2, 5))
    opt.append([D.optimal_a(np.full(5, v[l]), np.full(5, t0[l]), np.full(5, iti[l]), R, P) for l in (0, 1)])
    th = alt.loc[s, [f"shrink_c_abs_p{j}" for j in range(7)]].to_numpy(float)
    cost.append(A.thresholds("shrink_c_abs", th[4:], th[0:2], th[2:4], iti).reshape(2, 5))
act, opt, cost = map(np.array, (act, opt, cost))
se = lambda m: m.std(0) / np.sqrt(len(m))

# Fig 1: optimal vs actual vs cost model
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.0), sharey=True)
for l, ax in enumerate(axs):
    ax.errorbar(x, act[:, l].mean(0), se(act[:, l]), fmt="o", color="k", ms=3, label="Observed", zorder=3)
    ax.plot(x, opt[:, l].mean(0), "--", color="C3", label="Optimal $a^*$")
    ax.plot(x, cost[:, l].mean(0), "-", color="C0", label="Adjustment cost")
    ax.axhline(np.exp(np.log(alt.shrink_c_abs_p4 if l == 0 else alt.shrink_c_abs_p5)).mean(), color="C0", lw=.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([int(p) for p in P]); ax.set_xlabel("Error penalty (points)")
    ax.set_title(["No load", "Memory load"][l], fontsize=8)
axs[0].set_ylabel("Boundary separation $a$")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=6, loc="upper center", ncol=3)
fig.tight_layout(pad=.3, rect=(0, 0, 1, .9)); fig.savefig(os.path.join(OUT, "fig1_threshold.pdf")); plt.close(fig)

# Fig 2: held-out log-likelihood relative to adjustment cost, per held-out penalty
cv = pd.read_csv(os.path.join(RES, "cv_penalty.csv"))
names = {"compress": "Penalty compression", "reg": "Log-linear", "shrink_abs": "Cost per load"}
fig, ax = plt.subplots(figsize=(3.3, 1.7))
rng = np.random.default_rng(0); w = .25
for i, (m, lab) in enumerate(names.items()):
    d = np.array([(cv[f"{m}_heldout_P{k}"] - cv[f"shrink_c_abs_heldout_P{k}"]).to_numpy() for k in range(5)])  # >0: cost better
    tot = d.sum(1); bs = np.array([d[:, rng.integers(0, d.shape[1], d.shape[1])].sum(1) for _ in range(1000)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    ax.bar(np.arange(5) + (i - 1) * w, tot, w, yerr=[tot - lo, hi - tot], label=lab, error_kw=dict(lw=.6), color=f"C{i+1}")
ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(5)); ax.set_xticklabels([int(p) for p in P])
ax.set_xlabel("Held-out penalty level"); ax.set_ylabel("$\\Delta$ held-out log-lik.\n(adj. cost advantage)")
ax.legend(frameon=False, fontsize=6); fig.tight_layout(pad=.3); fig.savefig(os.path.join(OUT, "fig2_cv.pdf")); plt.close(fig)

# Fig 3: (a) default a0 by load; (b) observed a by sustained attention extremes
fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.7))
a0n, a0l = alt.shrink_c_abs_p4, alt.shrink_c_abs_p5
ax = axs[0]; ax.scatter(a0n, a0l, s=3, alpha=.4, color="C0", lw=0); m = [0.5, 3.5]; ax.plot(m, m, "k", lw=.6)
ax.set_xlim(m); ax.set_ylim(m); ax.set_xlabel("Default $a_0$, no load"); ax.set_ylabel("Default $a_0$, load"); ax.set_title("A", loc="left", fontsize=8)
ax = axs[1]; d = S.dropna(subset=["SuS_AC"]); lo, hi = d.SuS_AC.quantile([.25, .75])
for g, lab, c in ((d.index[d.SuS_AC >= hi], "High sustained attn.", "C2"), (d.index[d.SuS_AC <= lo], "Low sustained attn.", "C1")):
    idx = np.isin(mle.index, g); a = act[idx].mean(1)
    ax.errorbar(x, a.mean(0), se(a), fmt="o-", ms=2.5, color=c, label=lab, lw=.8)
ax.plot(x, opt.mean(1).mean(0), "--", color="C3", lw=.8, label="Optimal (all)")
ax.set_xticks(x); ax.set_xticklabels([int(p) for p in P]); ax.set_xlabel("Error penalty"); ax.set_ylabel("Observed $a$")
ax.legend(frameon=False, fontsize=5.5, loc="lower right"); ax.set_title("B", loc="left", fontsize=8)
fig.tight_layout(pad=.3); fig.savefig(os.path.join(OUT, "fig3_load_inddiff.pdf")); plt.close(fig)
print("group means: act", act.mean(0).round(2), "\nopt", opt.mean(0).round(2), "\ncost", cost.mean(0).round(2))
