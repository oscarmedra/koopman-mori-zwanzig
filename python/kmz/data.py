"""The data format: trajectories sampled at a constant time step."""
import numpy as np


class Data:
    """N trajectories of a dynamical system, sampled every ``dt``.

    Parameters
    ----------
    states : array (N, T+1, d)
        States x(0), x(dt), ..., x(T·dt) of each trajectory. For a single trajectory, pass
        (1, T+1, d) or (T+1, d). For a scalar system, pass (N, T+1, 1).
    dt : float
        Time step between two samples.
    observable : int or function
        The quantity ψ to predict. An integer i means the i-th state coordinate. A function must
        accept an array (..., d) and return (...). Default: coordinate 0.
    obs : array (N, L+1), optional
        Values of ψ observed for longer than the states (L ≥ T): the "ψ measured for a long time,
        full states only over a short window" scenario. If absent, ψ is computed from the states.
    names : list of str, optional
        Names of the coordinates (used in plots).

    The equations of the system are never needed.
    """

    def __init__(self, states, dt, observable=0, obs=None, names=None):
        X = np.asarray(states, dtype=float)
        if X.ndim == 2:   # a single trajectory (T+1, d)
            X = X[None]
        if X.ndim != 3:
            raise ValueError("states must have shape (N, T+1, d)")
        if not np.all(np.isfinite(X)):
            raise ValueError("states contains non-finite values (NaN or infinity)")
        self.states = X; self.dt = float(dt); self.names = names
        if isinstance(observable, (int, np.integer)):
            i = int(observable)
            from .taylor import comp
            self.psi = lambda Z, i=i: comp(Z, i)
            self.observable_name = names[i] if names else f"x{i}"
        elif callable(observable):
            self.psi = observable
            name = getattr(observable, "__name__", "psi")
            self.observable_name = "ψ" if name == "<lambda>" else name
        else:
            raise ValueError("observable: an integer (coordinate index) or a function")
        Y = np.asarray(self.psi(X), dtype=float).T          # (T+1, N)
        if obs is not None:
            obs = np.asarray(obs, dtype=float)
            if obs.ndim == 1: obs = obs[None]
            if obs.shape[0] != X.shape[0] or obs.shape[1] < X.shape[1]:
                raise ValueError("obs must have shape (N, L+1) with L+1 ≥ T+1")
            gap = np.max(np.abs(obs[:, :X.shape[1]].T - Y)) / (np.std(Y) + 1e-300)
            if gap > 1e-6:
                raise ValueError("obs does not match observable(states) on the common window")
            Y = obs.T
        self.Y = Y                                           # (L+1, N), time first

    @property
    def N(self): return self.states.shape[0]
    @property
    def d(self): return self.states.shape[2]
    @property
    def n_state_steps(self): return self.states.shape[1] - 1
    @property
    def n_obs_steps(self): return self.Y.shape[0] - 1
    @property
    def x0(self): return self.states[:, 0]

    def subset(self, idx):
        """Data restricted to the trajectories with indices ``idx``."""
        D = Data.__new__(Data); D.__dict__.update(self.__dict__)
        D.states = self.states[idx]; D.Y = self.Y[:, idx]
        return D

    def __repr__(self):
        s = (f"Data({self.N} trajectories, dimension {self.d}, dt={self.dt:g}, "
             f"states over {self.n_state_steps} steps")
        if self.n_obs_steps > self.n_state_steps:
            s += f", observable over {self.n_obs_steps} steps"
        return s + ")"
