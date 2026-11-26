"""Truncated Taylor series in t (forward-mode automatic differentiation).

Used only by the a posteriori certificate (module ``certificate``), when the equations are known:
for a vector field F written with these operations, the Lie derivatives are exact,
L^k ψ(x0) = k! · ψ(c_k), where c_k are the Taylor coefficients of the solution x(t) starting at x0.

Every function in this module accepts either numpy arrays or ``TS`` objects, so a dictionary or a
vector field written with ``comp``, ``stack``, ``lin_last``, ``concat``, ``cos_sin``, ``tanh`` works
both on data and on Taylor series.
"""
import math
import numpy as np


class TS:
    """Truncated Taylor series: ``c[k]`` is the coefficient of t^k (arrays of the same shape)."""
    __array_priority__ = 1000

    def __init__(self, c):
        self.c = c

    def __add__(self, o):
        if isinstance(o, TS):
            return TS(self.c + o.c)
        c = self.c.copy(); c[0] = c[0] + o
        return TS(c)
    __radd__ = __add__

    def __neg__(self):
        return TS(-self.c)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, TS):
            a, b = self.c, o.c; K1 = a.shape[0]
            out = np.zeros(np.broadcast_shapes(a.shape, b.shape))
            for j in range(K1):
                out[j:] += a[j] * b[:K1 - j]
            return TS(out)
        return TS(self.c * o)
    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, TS):
            raise TypeError("division by a series is not supported")
        return TS(self.c / o)


def comp(X, i):
    """i-th component (last axis)."""
    return TS(X.c[..., i]) if isinstance(X, TS) else X[..., i]


def stack(L):
    """Stack components along a new last axis."""
    if isinstance(L[0], TS):
        return TS(np.stack([l.c for l in L], -1))
    return np.stack(L, -1)


def concat(L):
    """Concatenate along the last axis."""
    if isinstance(L[0], TS):
        return TS(np.concatenate([l.c for l in L], -1))
    return np.concatenate(L, -1)


def lin_last(X, M):
    """Linear map on the last axis: X @ M."""
    return TS(X.c @ M) if isinstance(X, TS) else X @ M


def columns(X, a, b=None):
    return TS(X.c[..., a:b]) if isinstance(X, TS) else X[..., a:b]


def ones_col(X):
    """Column of ones, shaped like X without its last axis."""
    if isinstance(X, TS):
        o = np.zeros(X.c.shape[:-1] + (1,)); o[0] = 1
        return TS(o)
    return np.ones(X.shape[:-1] + (1,))


def spec(X, mult):
    """Diagonal Fourier operator on the last axis (periodic grid)."""
    c = X.c if isinstance(X, TS) else X
    n = c.shape[-1]
    out = np.fft.irfft(mult * np.fft.rfft(c, axis=-1), n=n, axis=-1)
    return TS(out) if isinstance(X, TS) else out


def cos_sin(U):
    """(cos U, sin U), also for a series (recurrence s' = c u', c' = -s u')."""
    if not isinstance(U, TS):
        return np.cos(U), np.sin(U)
    u = U.c; K1 = u.shape[0]
    s, c = np.zeros_like(u), np.zeros_like(u)
    s[0], c[0] = np.sin(u[0]), np.cos(u[0])
    for k in range(1, K1):
        s[k] = sum(j * u[j] * c[k - j] for j in range(1, k + 1)) / k
        c[k] = -sum(j * u[j] * s[k - j] for j in range(1, k + 1)) / k
    return TS(c), TS(s)


def tanh(U):
    """tanh U, also for a series (t' = (1 - t²) u')."""
    if not isinstance(U, TS):
        return np.tanh(U)
    u = U.c; K1 = u.shape[0]; t = np.zeros_like(u); w = np.zeros_like(u)
    t[0] = np.tanh(u[0]); w[0] = 1 - t[0] ** 2
    for k in range(1, K1):
        t[k] = sum(j * u[j] * w[k - j] for j in range(1, k + 1)) / k
        w[k] = -sum(t[i] * t[k - i] for i in range(k + 1))
    return TS(t)


def taylor(F, x0, K):
    """Taylor coefficients (K+1, N, d) of the solution of x' = F(x), x(0) = x0."""
    c = np.zeros((K + 1,) + x0.shape); c[0] = x0
    for k in range(K):
        c[k + 1] = F(TS(c[:k + 1])).c[k] / (k + 1)
    return c


def lie_derivatives(F, psi, x0, K):
    """L^k ψ(x0), k = 0..K, for a linear (or affine) ψ: L^k ψ = k! ψ(c_k). Returns (K+1, N)."""
    c = taylor(F, x0, K)
    out = [psi(c[0])]
    for k in range(1, K + 1):
        out.append(math.factorial(k) * (psi(c[k]) - psi(np.zeros_like(c[k]))))
    return np.array(out)


def lie1(F, dico, X):
    """(Φ(X), LΦ(X)) for a dictionary written with the operations of this module."""
    c = np.zeros((2,) + X.shape); c[0] = X; c[1] = F(X)
    D = dico(TS(c)).c
    return D[0], D[1]
