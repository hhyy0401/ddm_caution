"""Core DDM utilities: exact Wiener density, reward-rate optimal threshold, simulators.

Conventions: accuracy-coded DDM, diffusion noise s = 1, unbiased start z = a/2,
correct responses hit the upper boundary.
"""
import math
import numpy as np
from numba import njit, prange

EPS_TERMS = 1e-9
CONTAM = 0.02            # uniform contaminant mixture weight
RT_MIN, RT_MAX = 0.2, 3.0


# --------------------------------------------------------------------------- density
@njit(cache=True)
def _fpt_lower(t, v, a, w):
    """Navarro & Fuss (2009) density of hitting the LOWER boundary at time t."""
    if t <= 0.0:
        return 0.0
    u = t / (a * a)
    # number of terms
    if math.pi * u * EPS_TERMS < 1.0:
        kl = math.sqrt(-2.0 * math.log(math.pi * u * EPS_TERMS) / (math.pi ** 2 * u))
        kl = max(kl, 1.0 / (math.pi * math.sqrt(u)))
    else:
        kl = 1.0 / (math.pi * math.sqrt(u))
    if 2.0 * math.sqrt(2.0 * math.pi * u) * EPS_TERMS < 1.0:
        ks = 2.0 + math.sqrt(-2.0 * u * math.log(2.0 * math.sqrt(2.0 * math.pi * u) * EPS_TERMS))
        ks = max(ks, math.sqrt(u) + 1.0)
    else:
        ks = 2.0
    p = 0.0
    if ks < kl:
        K = int(math.ceil(ks))
        lo = -((K - 1) // 2)
        hi = (K - 1) - ((K - 1) // 2)
        for k in range(lo, hi + 1):
            p += (w + 2 * k) * math.exp(-((w + 2 * k) ** 2) / (2.0 * u))
        p /= math.sqrt(2.0 * math.pi * u ** 3)
    else:
        K = int(math.ceil(kl))
        for k in range(1, K + 1):
            p += k * math.exp(-(k ** 2) * (math.pi ** 2) * u / 2.0) * math.sin(k * math.pi * w)
        p *= math.pi
    return p * math.exp(-v * a * w - (v * v) * t / 2.0) / (a * a)


@njit(cache=True)
def loglik_trials(rt, acc, v, a, t0):
    """Sum log-likelihood. rt, acc, v, a, t0 are per-trial arrays (same length)."""
    unif = CONTAM * 0.5 / (RT_MAX - RT_MIN)
    ll = 0.0
    for i in range(rt.shape[0]):
        dt = rt[i] - t0[i]
        f = 0.0
        if dt > 0.0:
            if acc[i] == 1:
                f = _fpt_lower(dt, -v[i], a[i], 0.5)
            else:
                f = _fpt_lower(dt, v[i], a[i], 0.5)
        ll += math.log((1.0 - CONTAM) * f + unif)
    return ll


# --------------------------------------------------------------------------- normative threshold
A_GRID = np.linspace(0.05, 5.0, 1200)


def reward_rate(a, v, t0, iti, R, P):
    """Bogacz et al. (2006) reward rate for an unbiased DDM with s = 1."""
    va = np.clip(v * a, -50, 50)
    er = 1.0 / (1.0 + np.exp(va))
    dt = np.where(np.abs(v) > 1e-6, a / (2.0 * v) * np.tanh(va / 2.0), a * a / 4.0)
    return (R * (1.0 - er) - P * er) / (dt + t0 + iti)


def optimal_a(v, t0, iti, R, P):
    """Reward-rate maximising threshold, vectorised over conditions (grid + parabolic refine)."""
    v = np.atleast_1d(np.asarray(v, float))[:, None]
    t0 = np.atleast_1d(np.asarray(t0, float))[:, None]
    iti = np.atleast_1d(np.asarray(iti, float))[:, None]
    P = np.atleast_1d(np.asarray(P, float))[:, None]
    g = A_GRID[None, :]
    rr = reward_rate(g, v, t0, iti, R, P)
    k = np.clip(rr.argmax(1), 1, len(A_GRID) - 2)
    idx = np.arange(rr.shape[0])
    y0, y1, y2 = rr[idx, k - 1], rr[idx, k], rr[idx, k + 1]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    h = A_GRID[1] - A_GRID[0]
    return A_GRID[k] + np.clip(off, -1, 1) * h


# --------------------------------------------------------------------------- simulator
@njit(cache=True)
def _one_trial(v, a, t0, dt):
    x = a / 2.0
    t = 0.0
    sq = math.sqrt(dt)
    while True:
        x += v * dt + sq * np.random.standard_normal()
        t += dt
        if x >= a:
            return t + t0, 1
        if x <= 0.0:
            return t + t0, 0
        if t > RT_MAX + 1.0:
            return t + t0, 1 if x > a / 2.0 else 0
