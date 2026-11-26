"""A posteriori certificate of Theorem F, when the equations are known.

We certify the autocorrelation a(t) = ⟨ψ_c, T(t)ψ_c⟩ (ψ centred, normalised) of the continuous-time
gEDMD model A = PLP on the selected dictionary: |a(t) − ⟨ψ_c, e^{tA}ψ_c⟩| ≤ certificate(t).

Two forms:
  * general form (Thm F): moment gaps up to order K−1, plus a remainder in t^K/K! bounded by
    ‖L^Kψ‖ + ‖A^Kψ‖; this is a true bound;
  * remainder estimated on the data: the sup of the remainder is taken over the states observed along
    the trajectories (grid of s). This is an estimate, not a strict bound.

Limits (see BILAN.md, § 6): only about a(t), not about the prediction with memory; useful only at
short times (t ≲ 1 for ODEs, t ≲ 4 for dissipative PDEs); learned functions have large Lie derivatives
and degrade the certificate. ψ must be affine in the state.
"""
import math
import numpy as np
from .linalg import rms, whiten, expm
from .taylor import lie_derivatives, lie1


def _check_affine(psi, d, rng):
    a, b = rng.normal(size=(3, d)), rng.normal(size=(3, d)); z = np.zeros((1, d))
    gap = psi(a + b) - psi(a) - psi(b) + psi(z)
    if np.max(np.abs(gap)) > 1e-8 * (1 + np.max(np.abs(psi(a)))):
        raise ValueError("certificate: the observable ψ must be affine in the state")


def certificate_F(model, field, times=(0.25, 0.5, 1.0, 2.0), m=3000, Kmax=12, data_remainder=True,
                  s_max=2.0, ds=0.1):
    """Thm F certificate for a fitted ``KMZ`` model.

    field : vector field F(X), written with the operations of ``kmz.taylor`` (exact Lie derivatives).
    times : times at which to certify (multiples of dt; the truth a(t) is only computed if t lies in the
        state window).
    Returns a dict: rows (t, a, a_V, err, cert_F, cert_data, trivial_bound), useful_horizon (largest t
    where the certificate beats the trivial bound), and a few diagnostics."""
    o = model; psi = o.psi; dt = o.dt; S = o.data.states
    _check_affine(psi, o.d, np.random.default_rng(0))
    idx = o.itr[:m]; X = o.X0[idx]; mm = len(idx)
    mu = psi(X).mean(); nr = rms(psi(X) - mu)
    psin = lambda c: psi(c) * (1 / nr)
    psic = lambda Z: (psi(Z) - mu) * (1 / nr)
    Lk = lie_derivatives(field, psin, X, Kmax); Lk[0] = psic(X)
    mk = np.array([np.mean(Lk[0] * Lk[k]) for k in range(Kmax + 1)])
    nLk = np.array([rms(l) for l in Lk])
    D0, D1 = lie1(field, o.dictionary, X)
    Wh = whiten(D0.T @ D0 / mm); E0, E1 = D0 @ Wh, D1 @ Wh
    A = E0.T @ E1 / mm; cps = E0.T @ Lk[0] / mm
    Ap = [np.eye(len(A))]
    for _ in range(Kmax): Ap.append(Ap[-1] @ A)
    mh = np.array([cps @ Ap[k] @ cps for k in range(Kmax + 1)])
    nAk = np.array([np.linalg.norm(Ap[k] @ cps) for k in range(Kmax + 1)])
    H_st = S.shape[1] - 1
    step = max(1, int(round(ds / dt))); ns = min(int(round(s_max / dt)), H_st)
    rem_t, rem_V, s_list = [], [], []
    if data_remainder:
        for j in range(0, ns + 1, step):
            Lj = lie_derivatives(field, psin, S[idx, j], Kmax)
            rem_t.append(np.abs(Lk[0] @ Lj.T / mm)); eA = expm(j * dt * A)
            rem_V.append([abs(cps @ Ap[k] @ eA @ cps) for k in range(Kmax + 1)])
            s_list.append(j * dt)
    rem_t, rem_V = np.array(rem_t), np.array(rem_V)
    out = []
    for t in times:
        j = int(round(t / dt)); t = j * dt
        a = float(np.mean(Lk[0] * psic(S[idx, j]))) if j <= H_st else None
        aV = float(cps @ expm(t * A) @ cps)
        gap = lambda K: sum(abs(mk[k] - mh[k]) * t ** k / math.factorial(k) for k in range(K))
        cF = min(gap(K) + (nLk[K] + nAk[K]) * t ** K / math.factorial(K) for K in range(1, Kmax + 1))
        cD = None
        if data_remainder and len(s_list) and t <= s_list[-1] + 1e-9:
            sel = [i for i, s in enumerate(s_list) if s <= t + 1e-9]
            cD = min(gap(K) + (rem_t[sel, K].max() + rem_V[sel, K].max()) * t ** K / math.factorial(K)
                     for K in range(1, Kmax + 1))
        triv = float(rms(psic(S[idx, j])) + np.linalg.norm(expm(t * A) @ cps)) if j <= H_st else None
        out.append(dict(t=float(t), a=a, a_V=aV, err=None if a is None else abs(a - aV), cert_F=float(cF),
                        cert_data=None if cD is None else float(cD), trivial_bound=triv))
    best = lambda r: min(v for v in [r["cert_F"], r["cert_data"]] if v is not None)
    useful = [r["t"] for r in out if r["trivial_bound"] is not None and best(r) < r["trivial_bound"]]
    return dict(rows=out, useful_horizon=max(useful) if useful else 0.0,
                moment_gaps=[float(abs(mk[k] - mh[k])) for k in range(Kmax + 1)],
                asym_A=float(np.linalg.norm(A + A.T) / np.linalg.norm(A)), n=int(Wh.shape[1]))
