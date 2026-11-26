"""The dictionary V: the functions of the state in which the prediction is written.

Contents:
  * input coordinates (standardised state in low dimension, PCA components in high dimension);
  * generic blocks (monomials, quadratic products, PCA);
  * functions learned from data (random Fourier features + ridge regression), used to approximate
    the Koopman invariant functions (limit, basin indicator, Cesàro mean) and the unimodular
    eigenfunctions (harmonic mean);
  * the long-time diagnosis that decides which functions to learn (Prop. 8 and its weak extension).

Everything is compatible with Taylor series (module ``taylor``), so that the certificate can compute
exact Lie derivatives when the equations are known.
"""
import numpy as np
from .taylor import TS, comp, stack, lin_last, ones_col, cos_sin, tanh, columns
from .linalg import rms


# ====================================================================== input coordinates
class Coordinates:
    """Affine coordinates z = (x - m) M of the state, learned on the training initial states.

    Mode 'state': standardised state (low dimension). Mode 'pca': first r principal components,
    standardised (high dimension, e.g. a discretised PDE)."""

    def __init__(self, X0, mode="state", r=10):
        m = X0.mean(0)
        if mode == "state":
            sd = X0.std(0); sd[sd < 1e-12] = 1.0
            M = np.diag(1 / sd); self.explained_variance = 1.0
        else:
            _, s, Vt = np.linalg.svd(X0 - m, full_matrices=False)
            r = int(min(r, (s > 1e-10 * s[0]).sum()))
            sd = s[:r] / np.sqrt(len(X0))
            M = Vt[:r].T / sd
            self.explained_variance = float((s[:r] ** 2).sum() / (s ** 2).sum())
        self.mode, self.M, self.off, self.mean = mode, M, m @ M, m
        self.dim = M.shape[1]
        self._Minv = np.linalg.pinv(M)

    def __call__(self, X):
        return lin_last(X, self.M) - self.off

    def inverse(self, Z):
        """State x whose coordinates are Z (directions not kept by the PCA are set to the mean)."""
        return self.mean + Z @ self._Minv


# ====================================================================== generic blocks
def exponents(deg, d):
    """Multi-indices of total degree ≤ deg, in graded order."""
    res = []
    for tot in range(deg + 1):
        def rec(prefix, rem, left):
            if left == 1:
                res.append(prefix + (rem,)); return
            for a in range(rem, -1, -1):
                rec(prefix + (a,), rem - a, left - 1)
        rec((), tot, d)
    return res


def monomials(Z, deg, deg_min=0):
    """Monomials of the coordinates Z with deg_min ≤ |e| ≤ deg (Taylor-series compatible)."""
    d = (Z.c if isinstance(Z, TS) else Z).shape[-1]
    zs = [comp(Z, i) for i in range(d)]
    feats = {(0,) * d: ones_col(Z)[..., 0] if not isinstance(Z, TS) else TS(ones_col(Z).c[..., 0])}
    out = []
    for e in exponents(deg, d):
        if sum(e) > 0:
            i = next(j for j in range(d) if e[j] > 0)
            prev = tuple(e[j] - (j == i) for j in range(d))
            feats[e] = feats[prev] * zs[i]
        if sum(e) >= deg_min:
            out.append(feats[e])
    return stack(out)


def quadratic_products(Z, m):
    """Products z_i z_j (i ≤ j < m) of the first m coordinates."""
    zs = [comp(Z, i) for i in range(m)]
    return stack([zs[i] * zs[j] for i in range(m) for j in range(i, m)])


def default_base(coords, d):
    """Base dictionary (before enrichment). The constant and ψ are added by the model.
    Low dimension: monomials of degree ≤ 3 (d ≤ 2) or ≤ 2 (d = 3, 4). High dimension: 5 PCA components."""
    if coords.mode == "state":
        deg = 3 if d <= 2 else 2
        return f"monomials deg<={deg}", (lambda X: monomials(coords(X), deg, 1)), deg
    r = min(5, coords.dim)
    return f"PCA {r}", (lambda X: columns(coords(X), 0, r)), None


def generic_blocks(coords, d, deg_base):
    """Candidate blocks "without knowledge of the system" for the greedy selection."""
    if coords.mode == "state":
        sup = 2 if d <= 2 else 1
        if d > 4: sup = 0
        return [(f"monomials deg {g}", (lambda g: lambda X: monomials(coords(X), g, g))(g))
                for g in range(deg_base + 1, deg_base + sup + 1)]
    r = coords.dim
    out = []
    if r > 5:
        out.append((f"PCA {r}", lambda X: columns(coords(X), 5, r)))
    m = min(6, r)
    out.append((f"quadratic PCA {m}", lambda X: quadratic_products(coords(X), m)))
    return out


# ====================================================================== learned functions
class LearnedFunction:
    """φ(x) = Σ_j w_j √(2/D) cos(⟨ω_j, z(x)⟩ + φ_j), z = input coordinates.

    If ``w`` is two-dimensional (D, m), φ returns m outputs. ``sharpen = (centre, half_gap, kappa)``
    replaces φ by centre + h·tanh(κ(φ − centre)/h) ("sharpened" basin indicator, Prop. 8(iv))."""

    def __init__(self, coords, Om, ph, w, sharpen=None):
        self.coords, self.Om, self.ph = coords, Om, ph
        self.w = w * np.sqrt(2 / Om.shape[1]); self.sharpen = sharpen

    def sharpened(self, centre, half_gap, kappa=3.0):
        f = LearnedFunction.__new__(LearnedFunction); f.__dict__.update(self.__dict__)
        f.sharpen = (centre, half_gap, kappa)
        return f

    def __call__(self, X):
        Z = self.coords(X)
        if isinstance(Z, TS):
            c, _ = cos_sin(lin_last(Z, self.Om) + self.ph)
            out = lin_last(c, self.w)
        else:
            sh = Z.shape[:-1]; Zf = Z.reshape(-1, Z.shape[-1])
            out = np.empty((len(Zf),) + self.w.shape[1:])
            for i in range(0, len(Zf), 4000):
                out[i:i + 4000] = np.cos(Zf[i:i + 4000] @ self.Om + self.ph) @ self.w
            out = out.reshape(sh + self.w.shape[1:])
        if self.sharpen is not None:
            c0, h, kap = self.sharpen
            out = tanh((out - c0) * (kap / h)) * h + c0
        return out


def fit_rff(Z, y, rng, coords, D=1000, scales=(0.5, 1.0, 2.0), lambdas=(1e-6, 1e-4, 1e-2)):
    """Ridge regression of y (N,) or (N, m) on D random Fourier features of z.
    Hyperparameters (length scale, regularisation) are chosen on an internal 80/20 split.
    Z must already be standardised (output of ``Coordinates``). Returns (function, relative error)."""
    ntr = int(0.8 * len(Z)); best = None; y2 = y if y.ndim == 2 else y[:, None]
    for ell in scales:
        Om = rng.normal(0, 1 / ell, (Z.shape[1], D)); ph = rng.uniform(0, 2 * np.pi, D)
        F = np.cos(Z @ Om + ph) * np.sqrt(2 / D)
        A = F[:ntr].T @ F[:ntr]; r = F[:ntr].T @ y2[:ntr]
        for lam in lambdas:
            w = np.linalg.solve(A + lam * ntr * np.eye(D), r)
            e = rms(F[ntr:] @ w - y2[ntr:])
            if best is None or e < best[0]:
                best = (e, Om, ph, lam)
    e, Om, ph, lam = best
    F = np.cos(Z @ Om + ph) * np.sqrt(2 / D)
    w = np.linalg.solve(F.T @ F + lam * len(F) * np.eye(D), F.T @ y2)
    if y.ndim == 1: w = w[:, 0]
    return LearnedFunction(coords, Om, ph, w), float(e / (np.std(y2) + 1e-300))


# ====================================================================== long-time diagnosis
def two_means(b):
    """2-class k-means in 1D. Returns (centres, labels, spread/gap ratio)."""
    c = np.quantile(b, [0.25, 0.75]).astype(float)
    for _ in range(50):
        lab = np.abs(b - c[0]) > np.abs(b - c[1])
        if lab.all() or (~lab).all():
            return c, lab, np.inf
        c = np.array([b[~lab].mean(), b[lab].mean()])
    within = np.sqrt(np.mean((b - c[lab.astype(int)]) ** 2)); gap = abs(c[1] - c[0])
    return c, lab, within / max(gap, 1e-300)


def diagnose(Y, itr, dt, late_fraction=0.2, strong_threshold=0.03, start=None):
    """Long-time behaviour of ψ, from the observations Y (L+1, N).

    Returns (report, targets, b, window): ``targets`` lists the functions of x0 to learn
    (name, values (N,) or (N, 2), sharpening parameters or None).
    The late window starts at ``start`` (when ψ is observed beyond the target horizon, the model puts
    the end of the horizon there); otherwise it covers the last ``late_fraction`` of the data.
      * strong convergence (late agitation < threshold): b = ψ(T); otherwise b = late Cesàro mean;
      * if b is bimodal: basin indicator (2 classes);
      * if the late spectrum has a clear peak: harmonic mean (unimodular eigenfunction)."""
    s0 = np.std(Y[0]) + 1e-300; R = {}
    i0 = int((1 - late_fraction) * len(Y)) if start is None else start
    win = Y[i0:]
    agit = rms(win - win.mean(0)) / s0
    strong = agit < strong_threshold
    R["late_agitation"] = float(agit)
    R["convergence"] = "strong" if strong else "weak or none"
    b = Y[-1].copy() if strong else win.mean(0)
    targets = []
    cent, lab, rap = two_means(b[itr])
    frac = min(lab.mean(), 1 - lab.mean()) if np.isfinite(rap) else 0
    R["bimodality"] = dict(spread_gap_ratio=float(rap), min_fraction=float(frac), centres=cent.tolist())
    basins = rap < 0.25 and frac > 0.03 and abs(cent[1] - cent[0]) > 0.2 * s0
    if basins:
        b_cl = np.where(np.abs(b - cent[0]) > np.abs(b - cent[1]), cent[1], cent[0])
        targets.append(("basin", b_cl, (cent.mean(), abs(cent[1] - cent[0]) / 2)))
        R["invariant_function"] = f"basin indicator (2 classes, centres {np.round(cent, 3).tolist()})"
    else:
        targets.append(("limit" if strong else "cesaro", b, None))
        R["invariant_function"] = "limit of psi" if strong else "Cesàro mean of psi"
    if not strong and start is None and len(Y) // 2 > len(win):
        # without a window beyond the horizon, the period search uses the second half of the data
        # (the last fraction alone often covers fewer than two periods)
        i0 = len(Y) // 2
    win_c, win = win, Y[i0:]
    if not strong and len(win) >= 16:
        Wc = win - win.mean(0); P = np.mean(np.abs(np.fft.rfft(Wc, axis=0)) ** 2, 1)
        j = int(np.argmax(P[1:])) + 1; part = P[max(1, j - 1):j + 2].sum() / max(P[1:].sum(), 1e-300)
        R["spectral_peak_share"] = float(part)
        if part > 0.3 and j >= 2:            # at least two periods in the window
            L = len(win); om0 = 2 * np.pi * j / (L * dt); tk = (i0 + np.arange(L)) * dt
            grid = om0 + np.linspace(-1, 1, 81) * 2 * np.pi / (L * dt)
            energy = [np.mean(np.abs(np.exp(-1j * o * tk) @ Wc) ** 2) for o in grid]
            om = grid[int(np.argmax(energy))]
            h = (np.exp(-1j * om * tk) @ win) / L
            targets.append(("harmonic", np.c_[h.real, h.imag], None))
            R["periodic_component"] = dict(omega=float(om), period=float(2 * np.pi / om), share=float(part))
    return R, targets, b, win_c
