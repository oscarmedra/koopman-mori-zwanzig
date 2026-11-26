"""The benchmark of the 10 systems of the thesis (6 ODEs, 4 PDEs), for tests and examples.

The equations are only used here to **generate data**: the KMZ model only sees trajectories.
The vector fields are written with the operations of ``kmz.taylor``, which also allows the
a posteriori certificate (exact Lie derivatives).

    from kmz.systems import simulate, BENCHMARK
    data = simulate("damped_duffing", N=4000, duration=20.0, seed=0)
"""
import numpy as np
from .taylor import comp, stack, lin_last, spec
from .data import Data


# ====================================================================== integrators
def rk4(F, X, n_steps, dt, nsub, keep=True):
    h = dt / nsub; out = [X]
    for _ in range(n_steps):
        for _ in range(nsub):
            k1 = F(X); k2 = F(X + h / 2 * k1); k3 = F(X + h / 2 * k2); k4 = F(X + h * k3)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if keep: out.append(X)
    return np.stack(out, 1) if keep else X


class ETDRK4:
    """Kassam–Trefethen, u_t = L u + N(u), L diagonal in Fourier (rfft), batch of solutions (N, n)."""

    def __init__(self, Lhat, Nfun, h, M=32):
        self.N, self.h = Nfun, h
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - .5) / M); LR = h * Lhat[:, None] + r[None, :]
        self.E, self.E2 = np.exp(h * Lhat), np.exp(h * Lhat / 2)
        self.Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, 1))
        self.f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, 1))
        self.f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, 1))
        self.f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, 1))

    def step(self, v, n):
        Nf = lambda v: np.fft.rfft(self.N(np.fft.irfft(v, n=n, axis=-1)), axis=-1)
        Nv = Nf(v); a = self.E2 * v + self.Q * Nv; Na = Nf(a)
        b = self.E2 * v + self.Q * Na; Nb = Nf(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nv); Nc = Nf(c)
        return self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3


def etd(sch, U, n_steps, nsub, keep=True):
    n = U.shape[-1]; v = np.fft.rfft(U, axis=-1); out = [U]
    for _ in range(n_steps):
        for _ in range(nsub): v = sch.step(v, n)
        if keep: out.append(np.fft.irfft(v, n=n, axis=-1))
    return np.stack(out, 1) if keep else np.fft.irfft(v, n=n, axis=-1)


def fourier_basis(n, kmax, Lx):
    """Matrix (n, 2kmax+1): u -> (a0, a1, b1, ..., a_kmax, b_kmax)."""
    x = np.arange(n) * Lx / n; cols = [np.ones(n) / n]
    for k in range(1, kmax + 1):
        cols += [2 * np.cos(2 * np.pi * k * x / Lx) / n, 2 * np.sin(2 * np.pi * k * x / Lx) / n]
    return np.array(cols).T


def _modes(r, N, n, Lx, kmax, amp, mean_sd=0.0):
    x = np.arange(n) * Lx / n; U = np.zeros((N, n)) + mean_sd * r.normal(size=(N, 1))
    for kk in range(1, kmax + 1):
        U += amp / kk * (r.normal(size=(N, 1)) * np.cos(2 * np.pi * kk * x / Lx)
                         + r.normal(size=(N, 1)) * np.sin(2 * np.pi * kk * x / Lx))
    return U


# ====================================================================== the benchmark
class System:
    """A benchmark system: vector field F (Taylor-series compatible), law of the initial states,
    sampling step dt, observable, and integrator ``flow(X, n_steps, keep)``."""

    def __init__(self, name, title, kind, d, F, sample, dt, flow, observable, obs_name, behaviour):
        self.name, self.title, self.kind, self.d, self.F = name, title, kind, d, F
        self.sample, self.dt, self.flow, self.observable = sample, dt, flow, observable
        self.obs_name, self.behaviour = obs_name, behaviour

    def __repr__(self):
        return f"System({self.title}; d={self.d}, dt={self.dt:g}, observable {self.obs_name}; {self.behaviour})"


def _ode(name, title, d, F, sample, dt, nsub, obs, behaviour):
    return System(name, title, "ODE", d, F, sample, dt, lambda X, n, keep=True: rk4(F, X, n, dt, nsub, keep),
                  obs, f"x{obs}", behaviour)


def _build():
    S = {}
    # ---------------- ODEs
    dl = 0.5
    Fd = lambda X: stack([comp(X, 1), -dl * comp(X, 1) + comp(X, 0) - comp(X, 0) * comp(X, 0) * comp(X, 0)])
    S["damped_duffing"] = _ode("damped_duffing", "Damped Duffing (ẍ = −0.5ẋ + x − x³)", 2, Fd,
                               lambda r, N: r.uniform(-2, 2, (N, 2)), 0.1, 10, 0, "2 basins (x → ±1)")
    Fdc = lambda X: stack([comp(X, 1), comp(X, 0) - comp(X, 0) * comp(X, 0) * comp(X, 0)])
    def s_dc(r, N):
        g = np.linspace(-3, 3, 20001); p = np.exp(g ** 2 / 2 - g ** 4 / 4); cdf = np.cumsum(p); cdf /= cdf[-1]
        return np.stack([np.interp(r.uniform(size=N), cdf, g), r.normal(size=N)], 1)
    S["conservative_duffing"] = _ode("conservative_duffing", "Conservative Duffing", 2, Fdc, s_dc, 0.1, 10, 0,
                                     "conservative (phase mixing)")
    Fv = lambda X: stack([comp(X, 1), (1 - comp(X, 0) * comp(X, 0)) * comp(X, 1) - comp(X, 0)])
    S["van_der_pol"] = _ode("van_der_pol", "Van der Pol (μ=1)", 2, Fv, lambda r, N: r.uniform(-3, 3, (N, 2)),
                            0.1, 20, 0, "limit cycle")
    Fl = lambda X: stack([comp(X, 0) - comp(X, 0) * comp(X, 1), comp(X, 0) * comp(X, 1) - comp(X, 1)])
    S["lotka_volterra"] = _ode("lotka_volterra", "Lotka–Volterra", 2, Fl, lambda r, N: r.gamma(4, 1 / 4, (N, 2)),
                               0.1, 20, 0, "conservative (closed orbits)")
    Fp = lambda X: stack([-comp(X, 1) * comp(X, 2), comp(X, 0) * comp(X, 2), -comp(X, 1)])
    def s_p(r, N):
        th = r.vonmises(0, 1, N); return np.stack([np.cos(th), np.sin(th), r.normal(size=N)], 1)
    S["pendulum"] = _ode("pendulum", "Pendulum (cos θ, sin θ, ω)", 3, Fp, s_p, 0.1, 10, 1, "conservative")
    sg, rh, bt = 10.0, 28.0, 8 / 3
    FL = lambda X: stack([sg * (comp(X, 1) - comp(X, 0)), comp(X, 0) * (rh - comp(X, 2)) - comp(X, 1),
                          comp(X, 0) * comp(X, 1) - bt * comp(X, 2)])
    def s_L(r, N):
        X = np.c_[r.normal(0, 5, (N, 2)), 25 + r.normal(0, 5, N)]
        return rk4(FL, X, 1, 30.0, 6000, keep=False)
    S["lorenz"] = _ode("lorenz", "Lorenz-63", 3, FL, s_L, 0.02, 4, 0, "chaos")
    # ---------------- PDEs (Fourier Galerkin, periodic, 64 points)
    n = 64
    def _pde(name, title, Lx, Lhat_f, Nl, dealias, sample, dt, h, psi_vec, obs_name, behaviour):
        k = 2 * np.pi * np.fft.rfftfreq(n, d=Lx / n); Lhat = Lhat_f(k); mask = np.ones_like(k)
        if dealias: mask[k > (2 / 3) * k.max()] = 0
        F = lambda U: spec(U, Lhat) + spec(Nl(U, k), mask)
        sch = ETDRK4(Lhat, lambda U: spec(Nl(U, k), mask), h); nsub = int(round(dt / h))
        obs = lambda U: lin_last(U, psi_vec)
        return System(name, title, "PDE", n, F, sample, dt, lambda U, m, keep=True: etd(sch, U, m, nsub, keep),
                      obs, obs_name, behaviour)
    dx = lambda U, k: spec(U, 1j * k); Lx = 2 * np.pi; mean_ = np.ones(n) / n
    S["burgers"] = _pde("burgers", "Viscous Burgers (ν=0.05)", Lx, lambda k: -0.05 * k ** 2,
                        lambda U, k: -0.5 * dx(U * U, k), True, lambda r, N: _modes(r, N, n, Lx, 3, 0.6, 0.5),
                        0.1, 0.01, np.eye(n)[0], "u(0)", "convergence to the mean")
    LK = 22.0
    S["ks"] = _pde("ks", "Kuramoto–Sivashinsky (L=22)", LK, lambda k: k ** 2 - k ** 4,
                   lambda U, k: -0.5 * dx(U * U, k), True, None, 0.25, 0.05, fourier_basis(n, 1, LK)[:, 1],
                   "a1 (cos 1 mode)", "chaos")
    S["ks"].sample = lambda r, N: S["ks"].flow(_modes(r, N, n, LK, 4, 0.5), 800, keep=False)
    S["allen_cahn"] = _pde("allen_cahn", "Allen–Cahn (ε²=0.36)", Lx, lambda k: -0.36 * k ** 2,
                           lambda U, k: U - U * U * U, False, lambda r, N: _modes(r, N, n, Lx, 3, 0.5, 0.3),
                           0.2, 0.02, mean_, "mean of u", "2 basins (u → ±1)")
    xg = np.arange(n) * Lx / n; a = 0.3
    def s_N(r, N):
        A = r.uniform(0.2, 1.2, (N, 1)); w = r.uniform(0.3, 1.5, (N, 1)); c = r.uniform(0, Lx, (N, 1))
        dd = np.angle(np.exp(1j * (xg[None, :] - c)))
        return A * np.exp(-dd ** 2 / (2 * w ** 2)) + 0.05 * _modes(r, N, n, Lx, 3, 1.0)
    S["nagumo"] = _pde("nagumo", "Bistable Nagumo (D=0.1, a=0.3)", Lx, lambda k: -0.1 * k ** 2,
                       lambda U, k: -U * U * U + (1 + a) * U * U - a * U, False, s_N, 0.2, 0.02, mean_,
                       "mean of u", "2 basins (u → 0 or 1)")
    return S


BENCHMARK = _build()


def simulate(name, N, duration, seed=0, state_duration=None, subsample=1):
    """Generate ``Data`` for a benchmark system: N trajectories over [0, duration].

    state_duration: if given, states are only kept over [0, state_duration] (ψ over [0, duration]).
    subsample: keep only one sample out of s (effective dt = s·dt)."""
    s = BENCHMARK[name]; rng = np.random.default_rng(seed)
    X0 = s.sample(rng, N); dt = s.dt * subsample
    n_steps = int(round(duration / dt))
    if state_duration is None:
        X = [X0]; Z = X0
        for _ in range(n_steps):
            Z = s.flow(Z, subsample, keep=False); X.append(Z)
        return Data(np.stack(X, 1), dt, observable=s.observable)
    n_st = int(round(state_duration / dt)); X = [X0]; Z = X0; psi = Data(X0[:, None], dt, s.observable).psi
    Y = [psi(X0)]
    for k in range(n_steps):
        Z = s.flow(Z, subsample, keep=False); Y.append(psi(Z))
        if k + 1 <= n_st: X.append(Z)
    return Data(np.stack(X, 1), dt, observable=s.observable, obs=np.array(Y).T)
