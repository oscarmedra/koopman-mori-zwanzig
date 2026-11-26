"""The Mori–Zwanzig memory, estimated from data.

In a basis of the dictionary V that is orthonormal for the empirical measure of the initial states,
let C(k) = E[Φ(x0)ᵀ Φ(x_k)] be the correlations. The discrete Mori–Zwanzig equation reads
    C(k+1) = Σ_{l=0..k} C(k−l) Ω_l,
which gives the memory kernels Ω_l recursively (Cor. 4, method of Lin et al. 2021).

The "hybrid" model uses M_k = C(k) (exact) while the correlations are known (data window), then the
recursion truncated to p memory terms beyond.
"""
import numpy as np
from .linalg import rms


def memory_kernels(C):
    """Ω_k = C(k+1) − Σ_{l<k} C(k−l) Ω_l (orthonormal basis: C(0) = I)."""
    Om = []
    for k in range(len(C) - 1):
        Om.append(C[k + 1] - sum(C[k - l] @ Om[l] for l in range(k)))
    return Om


def companion_radius(Om, p):
    """Spectral radius of the recursion X_{k+1} = Σ_{l≤p} X_{k−l} Ω_l (stable if ≤ 1)."""
    n = Om[0].shape[0]; Cm = np.zeros((n * (p + 1), n * (p + 1)))
    for l in range(p + 1):
        Cm[:n, l * n:(l + 1) * n] = Om[l].T
    if p > 0:
        Cm[n:, :-n] = np.eye(n * p)
    return float(max(abs(np.linalg.eigvals(Cm))))


def hybrid(C, Om, p, K, seed=None):
    """M_k, k = 0..K: exact (= C(k)) for k ≤ seed, then the recursion truncated to p.
    seed = None: the whole known window; seed = 0: pure recursion (Markov if p = 0)."""
    Hs = len(C) - 1 if seed is None else seed
    M = [C[k] for k in range(min(Hs, K) + 1)]
    for k in range(len(M) - 1, K):
        M.append(sum(M[k - l] @ Om[l] for l in range(min(k, p) + 1)))
    return M


def choose_p(C, Om, E_val, cpsi, y_val, H_target, grid=(0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120)):
    """Choice of the memory length p by temporal validation.

    The Ω are re-estimated on the half window; each p extrapolates up to the full window, and the error
    is measured on the validation trajectories. Only the p whose recursion is stable are kept (companion
    radius ≤ 1 + tol, tol = growth of at most ×2 over the target horizon), then the best extrapolator is
    chosen. If no p is stable, the smallest radius is taken and this is reported (``stable = False``)."""
    H_mem = len(C) - 1; Hs = H_mem // 2
    p_max = min(Hs - 1, 60)                 # beyond, the companion matrix gets expensive and nothing changes
    gr = sorted(set([p for p in grid if p <= p_max] + [max(p_max, 0)]))
    H_extra = max(H_target - H_mem, H_mem, 1)
    tol = 2 ** (1 / H_extra) - 1
    tab = []
    for p in gr:
        M = hybrid(C[:Hs + 1], Om[:Hs], p, H_mem)
        e = np.array([rms(E_val @ (M[k] @ cpsi) - y_val[k]) for k in range(Hs + 1, H_mem + 1)])
        rho = companion_radius(Om, p)
        tab.append(dict(p=int(p), extrap_error=float(rms(e)), radius=rho, stable=bool(rho <= 1 + tol)))
    st = [t for t in tab if t["stable"]]
    if st:
        choice = min(st, key=lambda t: t["extrap_error"]); stable = True
    else:
        choice = min(tab, key=lambda t: t["radius"]); stable = False
    return choice["p"], tab, tol, stable
