"""Fixed-duration-block simulator + summary statistics (shared by simulated and real data).

Session = 12 blocks of 60 s: per load, penalties (5, 10, 20, 20, 40, 80).
Trial i in a block consumes RT_i + ITI_load seconds; the block ends when 60 s are used.

Threshold on trial i of a block in cell c:
    a_i = A_c * exp(x_i),  A_c = a*(v_l, t0_l, ITI_l; R, lambda_l 20 (P/20)^gamma_l)   (normative target)
    x_0 = delta0,  x_{i+1} = rho * x_i + eta * error_i + sigma * eps_i                   (caution dynamics)
    v_i = v_l * (1 - zeta * error_{i-1})                                                (post-error drift dip)
Static model = delta0 = eta = zeta = 0, sigma -> 0.

theta layout (13): v0, v1, t00, t01, loglam0, loglam1, gam0, gam1, delta0, eta, zeta, rho, logsigma
"""
import math
import numpy as np
from numba import njit, prange

import ddm_core as D

PENS = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
BLOCK_PENS = np.array([0, 1, 2, 2, 3, 4])          # penalty index per block within a load
ITI = np.array([0.537, 0.567])                      # median per-trial overhead (NoLoad, Load), from prep.py
R = 20.0
BLOCK_SEC = 60.0
MAXT = 1400
N_STATS = 10 * 8 + 2 * 5


def targets(theta):
    """Normative target thresholds A[n, 10] for theta[n, >=6]."""
    n = theta.shape[0]
    lv = np.repeat([0, 1], 5)
    P = np.tile(PENS, 2)
    v = theta[:, 0:2][:, lv].ravel()
    t0 = theta[:, 2:4][:, lv].ravel()
    lam = np.exp(theta[:, 4:6])[:, lv].ravel()
    gam = theta[:, 6:8][:, lv].ravel()
    iti = np.tile(ITI[lv], n)
    Peff = lam * 20.0 * (np.tile(P, n) / 20.0) ** gam
    return D.optimal_a(v, t0, iti, R, Peff).reshape(n, 10)


@njit(cache=True)
def _simulate_one(th, A, dt, seed, rt, acc, cell, blk, posn, prev):
    np.random.seed(seed)
    v0, v1, t00, t01 = th[0], th[1], th[2], th[3]
    delta0, eta, zeta, rho, sig = th[8], th[9], th[10], th[11], math.exp(th[12])
    n = 0
    b = 0
    for l in range(2):
        v = v0 if l == 0 else v1
        t0 = t00 if l == 0 else t01
        for kb in range(6):
            c = 5 * l + BLOCK_PENS[kb]
            used = 0.0
            x = delta0
            start = n
            last = -1
            while used < BLOCK_SEC and n < MAXT:
                a = A[c] * math.exp(x)
                if a < 0.05:
                    a = 0.05
                vi = v * (1.0 - zeta) if last == 0 else v
                r, k = D._one_trial(vi, a, t0, dt)
                if np.random.random() < D.CONTAM:
                    r = D.RT_MIN + np.random.random() * (D.RT_MAX - D.RT_MIN)
                    k = 1 if np.random.random() < 0.5 else 0
                rt[n] = r
                acc[n] = k
                cell[n] = c
                blk[n] = b
                prev[n] = last
                last = k
                used += r + ITI[l]
                x = rho * x + eta * (1 - k) + sig * np.random.standard_normal()
                n += 1
            for j in range(start, n):
                posn[j] = (j - start) / max(n - start, 1)
            b += 1
    return n


@njit(cache=True)
def _quant(sorted_x, q):
    m = sorted_x.shape[0]
    h = (m - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, m - 1)
    return sorted_x[lo] + (h - lo) * (sorted_x[hi] - sorted_x[lo])


@njit(cache=True)
def summarize(rt, acc, cell, blk, posn, prev, n):
    """Summary statistics, identical for simulated and observed data."""
    s = np.zeros(N_STATS)
    inw = np.zeros(n, np.bool_)
    for i in range(n):
        inw[i] = rt[i] >= D.RT_MIN and rt[i] <= D.RT_MAX
    # per-cell log-RT mean (for residuals)
    cm = np.zeros(10)
    cn = np.zeros(10)
    for i in range(n):
        if inw[i]:
            cm[cell[i]] += math.log(rt[i])
            cn[cell[i]] += 1
    for c in range(10):
        cm[c] = cm[c] / cn[c] if cn[c] > 0 else 0.0
    nblocks_c = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0])
    for c in range(10):
        cor = np.empty(n)
        nc = 0
        err = np.empty(n)
        ne = 0
        na = 0
        ntot = 0
        for i in range(n):
            if cell[i] != c:
                continue
            ntot += 1
            if not inw[i]:
                continue
            na += 1
            if acc[i] == 1:
                cor[nc] = rt[i]
                nc += 1
            else:
                err[ne] = rt[i]
                ne += 1
        o = 8 * c
        s[o] = nc / max(na, 1)
        if nc >= 2:
            srt = np.sort(cor[:nc])
            for j, q in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
                s[o + 1 + j] = _quant(srt, q)
            med = s[o + 3]
        else:
            med = 0.7
            for j in range(5):
                s[o + 1 + j] = med
        s[o + 6] = (_quant(np.sort(err[:ne]), 0.5) - med) if ne >= 2 else 0.0
        s[o + 7] = math.log(max(ntot, 1) / nblocks_c[c])
    # dynamics per load
    for l in range(2):
        pe_s = 0.0; pe_n = 0; pc_s = 0.0; pc_n = 0
        pe_a = 0.0; pc_a = 0.0
        e_s = 0.0; e_n = 0; L_s = 0.0; L_n = 0; e_a = 0.0; L_a = 0.0
        ac_xy = 0.0; ac_xx = 0.0
        for i in range(n):
            if cell[i] // 5 != l or not inw[i]:
                continue
            r = math.log(rt[i]) - cm[cell[i]]
            if prev[i] == 0:
                pe_s += r; pe_a += acc[i]; pe_n += 1
            elif prev[i] == 1:
                pc_s += r; pc_a += acc[i]; pc_n += 1
            if posn[i] < 0.25:
                e_s += r; e_a += acc[i]; e_n += 1
            elif posn[i] >= 0.75:
                L_s += r; L_a += acc[i]; L_n += 1
            if i > 0 and blk[i - 1] == blk[i] and inw[i - 1]:
                r0 = math.log(rt[i - 1]) - cm[cell[i - 1]]
                ac_xy += r * r0
                ac_xx += r0 * r0
        o = 80 + 5 * l
        if pe_n > 0 and pc_n > 0:
            s[o] = pe_s / pe_n - pc_s / pc_n
            s[o + 1] = pe_a / pe_n - pc_a / pc_n
        if e_n > 0 and L_n > 0:
            s[o + 2] = e_s / e_n - L_s / L_n
            s[o + 4] = e_a / e_n - L_a / L_n
        s[o + 3] = ac_xy / ac_xx if ac_xx > 0 else 0.0
    return s


@njit(parallel=True, cache=True)
def _simulate_batch(theta, A, dt, seeds):
    N = theta.shape[0]
    out = np.zeros((N, N_STATS))
    for j in prange(N):
        rt = np.zeros(MAXT); acc = np.zeros(MAXT, np.int64); cell = np.zeros(MAXT, np.int64)
        blk = np.zeros(MAXT, np.int64); posn = np.zeros(MAXT); prev = np.zeros(MAXT, np.int64)
        n = _simulate_one(theta[j], A[j], dt, seeds[j], rt, acc, cell, blk, posn, prev)
        out[j] = summarize(rt, acc, cell, blk, posn, prev, n)
    return out


def simulate(theta, dt=0.001, seed=0):
    """theta: (N, 13) array (static model: delta0=eta=zeta=rho=0, logsigma=-20)."""
    theta = np.ascontiguousarray(theta, float)
    A = targets(theta)
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, len(theta))
    return _simulate_batch(theta, np.ascontiguousarray(A), dt, seeds)


def observed_stats(df):
    """Summary stats for one subject's trial DataFrame (from results/trials.csv)."""
    df = df.copy()
    pidx = df.Penalty.map({5: 0, 10: 1, 20: 2, 40: 3, 80: 4}).to_numpy()
    cell = (5 * df.load.to_numpy() + pidx).astype(np.int64)
    blk = df.TaskBlock.to_numpy().astype(np.int64)
    n_in_blk = df.groupby("TaskBlock").pos.transform("size").to_numpy()
    posn = df.pos.to_numpy() / n_in_blk
    prev = df.prev_acc.fillna(-1).to_numpy().astype(np.int64)
    return summarize(df.rt.to_numpy(float), df.acc.to_numpy(np.int64), cell, blk, posn.astype(float), prev, len(df))
