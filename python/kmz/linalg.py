"""Small linear-algebra helpers (numpy only)."""
import numpy as np


def rms(a, axis=None):
    """Root mean square."""
    return np.sqrt(np.mean(np.asarray(a) ** 2, axis=axis))


def whiten(G, tol=1e-9):
    """Matrix Wh (n, r) such that Whᵀ G Wh = I: empirical orthonormalisation of a dictionary,
    keeping only the numerical rank (redundant functions are dropped)."""
    w, U = np.linalg.eigh((G + G.T) / 2)
    keep = w > tol * w.max()
    return U[:, keep] / np.sqrt(w[keep])


def expm(A):
    """Matrix exponential (scaling and squaring, Taylor of order 18)."""
    nA = np.linalg.norm(A, 1)
    s = max(0, int(np.ceil(np.log2(nA + 1e-300))) + 1) if nA > 0.5 else 0
    B = A / 2 ** s; E = np.eye(len(A)); T = np.eye(len(A))
    for k in range(1, 19):
        T = T @ B / k; E = E + T
    for _ in range(s):
        E = E @ E
    return E
