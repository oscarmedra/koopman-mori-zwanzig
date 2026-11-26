"""The KMZ model: ``model = KMZ().fit(data)`` then ``model.predict(x0)``.

What the model predicts: the best approximation, in the dictionary V, of ψ(x(t)) as a function of
the initial state x0 (least squares over the distribution of the initial states in the data).
It is not a solution of the equations, and the error never goes below the dictionary "floor"
(the part of ψ(x(t)) that no function of V can represent).

Three stages (references: recherche/demonstrations.tex):
  1. Dictionary (Prop. 8 and weak extension): long-time diagnosis, learned invariant functions,
     greedy selection by the validation floor (Thm H(iii)), then final selection by the error of the
     full model when the memory window is shorter than the horizon.
  2. Memory (Cor. 4, Thm H): memory kernels Ω_l estimated from correlations; hybrid model; choice of p.
  3. A posteriori error: validation error at each time (global), and a local error bar (nearest
     neighbours among the validation trajectories). Optional Thm F certificate when the equations are
     known (module ``certificate``).
"""
import warnings
import numpy as np
from .linalg import rms, whiten
from .taylor import concat, stack, ones_col
from .data import Data
from . import dictionary as dic
from .memory import memory_kernels, hybrid, choose_p


class KMZWarning(UserWarning):
    """Warning specific to kmz (unstable memory, extrapolation, out-of-domain state, ...)."""


# ====================================================================== result of a prediction
class Prediction:
    """Result of ``KMZ.predict``.

    Attributes
    ----------
    t : (K+1,) times.
    value : (K+1,) or (K+1, M) prediction of ψ(x(t)).
    error : same shape, error bar (estimated root-mean-square error); NaN beyond the data.
    global_error : (K+1,) validation error averaged over all validation trajectories.
    out_of_domain : bool or (M,) bool, initial state far from all training data.
    """

    def __init__(self, t, value, error, global_error, out_of_domain, valid_horizon):
        self.t, self.value, self.error = t, value, error
        self.global_error, self.out_of_domain = global_error, out_of_domain
        self.valid_horizon = valid_horizon

    def interval(self, n_sigma=2.0):
        """(low, high) = value ∓ n_sigma · error."""
        return self.value - n_sigma * self.error, self.value + n_sigma * self.error

    def plot(self, ax=None, truth=None, n_sigma=2.0, color="C0", label="KMZ prediction"):
        """Plot the prediction and its ± n_sigma·error band (a single initial condition)."""
        import matplotlib.pyplot as plt
        if ax is None: _, ax = plt.subplots(figsize=(7, 3.5))
        v = self.value if self.value.ndim == 1 else self.value[:, 0]
        e = self.error if self.error.ndim == 1 else self.error[:, 0]
        ax.plot(self.t, v, color=color, label=label)
        ax.fill_between(self.t, v - n_sigma * e, v + n_sigma * e, color=color, alpha=0.25,
                        label=f"± {n_sigma:g} × estimated error")
        if truth is not None:
            ax.plot(self.t[:len(truth)], truth, "k--", lw=1, label="true trajectory")
        ax.set_xlabel("t"); ax.legend(loc="best", fontsize=8)
        return ax

    def __repr__(self):
        s = "Prediction(" + (f"{self.value.shape[1]} initial conditions, " if self.value.ndim == 2 else "")
        return s + f"t from 0 to {self.t[-1]:g}, error estimated up to t = {self.valid_horizon:g})"


# ====================================================================== the model
class KMZ:
    """Koopman–Mori–Zwanzig predictor learned from trajectories.

    Parameters
    ----------
    dictionary : 'auto' (default) or 'base'
        'auto' enriches the base dictionary with learned functions (invariant functions, basin
        indicator, harmonic) and generic blocks, kept only if they help on validation data.
        'base' keeps the base dictionary (monomials or PCA, plus the constant and ψ).
    base : function X -> (..., m), optional
        User-supplied base dictionary (the constant and ψ are added). Always kept.
    candidates : list of (name, function X -> (..., m)), optional
        Functions proposed by the user (energy, conserved quantity, ...). They enter the greedy
        selection like the others and are kept only if they lower the error.
        See ``evaluate_dictionary`` to test a function before adding it.
    coords : 'auto', 'state' or 'pca'
        Input coordinates of the learned functions. 'auto': standardised state if d ≤ 10, else PCA.
    n_pca : number of PCA components in 'pca' mode.
    validation_fraction : share of trajectories held out for validation (default 0.3).
    gain_threshold : minimal relative gain to add a block to the dictionary (default 2 %).
    neighbors : number of neighbours for the local error bar (default 50).
    seed : random seed.
    verbose : print the fitting log.
    """

    def __init__(self, dictionary="auto", base=None, candidates=None, coords="auto", n_pca=10,
                 validation_fraction=0.3, gain_threshold=0.02, neighbors=50, seed=0, verbose=False):
        if dictionary not in ("auto", "base"):
            raise ValueError("dictionary: 'auto' or 'base'")
        self.dictionary_mode, self.user_base, self.coords_mode = dictionary, base, coords
        self.user_candidates = list(candidates or [])
        self.n_pca, self.validation_fraction, self.threshold = n_pca, validation_fraction, gain_threshold
        self.neighbors, self.seed, self.verbose = neighbors, seed, verbose
        self.log = []; self.warnings = []

    # ------------------------------------------------------------------ helpers
    def _log(self, *a):
        s = " ".join(str(x) for x in a); self.log.append(s)
        if self.verbose: print(s, flush=True)

    def _warn(self, msg):
        self.warnings.append(msg); self._log("  WARNING: " + msg)
        warnings.warn(msg, KMZWarning, stacklevel=3)

    def _dico(self, blocks=None):
        blocks = self.blocks if blocks is None else blocks
        return lambda X: concat([f(X) for _, f in blocks])

    @property
    def dictionary(self):
        """Function X -> Φ(X) of the selected dictionary."""
        return self._dico()

    @property
    def block_names(self):
        return [b[0] for b in self.blocks]

    def _ortho_basis(self, Phi0):
        return whiten(Phi0[self.itr].T @ Phi0[self.itr] / len(self.itr))

    def _validation_floor(self, blocks, ks):
        """Validation error of the learned projection ψ(x_k) ≈ Φ(x0)β_k (β_k fitted on training).
        With the full memory, this is the error of the model (Thm H(iii))."""
        Phi0 = self._dico(blocks)(self.X0); Wh = self._ortho_basis(Phi0); E = Phi0 @ Wh
        B = E[self.itr].T @ self.Y[ks][:, self.itr].T / len(self.itr)
        err = rms(E[self.iva] @ B - self.Y[ks][:, self.iva].T, axis=0)
        return err, Wh.shape[1]

    # ------------------------------------------------------------------ fitting
    def fit(self, data, horizon=None, memory_window=None):
        """Fit the model on ``Data``.

        horizon : time up to which we want to predict (default: the whole observed duration of ψ).
        memory_window : duration over which the states are used to estimate the correlations
            (default: the whole duration of the states). Beyond it, the memory is extrapolated.
        """
        if not isinstance(data, Data):
            raise TypeError("fit expects a Data object (see kmz.Data)")
        D = data; self.data = D; self.dt = D.dt; self.psi = D.psi; self.d = D.d
        self.log, self.warnings = [], []
        rng = np.random.default_rng(self.seed); self.rng = rng
        N = D.N
        if N < 50:
            raise ValueError(f"too few trajectories ({N}): a few hundred at least are needed")
        perm = rng.permutation(N); ntr = int((1 - self.validation_fraction) * N)
        self.itr, self.iva = np.sort(perm[:ntr]), np.sort(perm[ntr:])
        self.X0 = D.x0; self.Y = D.Y
        H_obs, H_st = D.n_obs_steps, D.n_state_steps
        H_target = H_obs if horizon is None else int(round(horizon / self.dt))
        if H_target > H_obs:
            self._warn(f"horizon {horizon:g} beyond the observed duration {H_obs*self.dt:g}: "
                       "targeting the observed duration")
            H_target = H_obs
        H_mem = H_st if memory_window is None else int(round(memory_window / self.dt))
        H_mem = min(H_mem, H_st)
        if H_mem < 4:
            raise ValueError("the memory window must cover at least 4 time steps")
        self.H_target, self.H_mem, self.H_obs = H_target, H_mem, H_obs
        self._log(f"{D}")
        self._log(f"  training {len(self.itr)}, validation {len(self.iva)}; target horizon t = {H_target*self.dt:g}, "
                  f"memory window t = {H_mem*self.dt:g}")
        # ---- coordinates and base dictionary
        mode = self.coords_mode if self.coords_mode != "auto" else ("state" if self.d <= 10 else "pca")
        self.coords = dic.Coordinates(self.X0[self.itr], mode, self.n_pca)
        if mode == "pca":
            self._log(f"  PCA of the initial states: {self.coords.dim} components, explained variance {self.coords.explained_variance:.3f}")
        if self.user_base is not None:
            name_b, f_b, deg = "user base", self.user_base, (3 if self.d <= 2 else 2)
        else:
            name_b, f_b, deg = dic.default_base(self.coords, self.d)
        psi = self.psi
        self.blocks = [(f"V ({name_b})", lambda X: concat([ones_col(X), f_b(X), stack([psi(X)])]))]
        self.candidates = []
        if self.dictionary_mode == "auto":
            self.candidates = dic.generic_blocks(self.coords, self.d, deg or 0)
        self.candidates += self.user_candidates
        # ---- stage 1: diagnosis, learning, selection
        self._diagnose()
        self._selection()
        self._final_selection()
        self._asymptotic_diagnosis()
        # ---- stage 3: a posteriori error
        self._prepare_error()
        self._log(self.summary())
        return self

    def _diagnose(self):
        # late window: beyond the target horizon if ψ is observed there (long enough), else the last 20 %
        start = self.H_target + 1 if self.H_obs - self.H_target >= max(16, self.H_obs // 5) else None
        R, targets, b, win = dic.diagnose(self.Y, self.itr, self.dt, start=start)
        self.diag, self.b, self.win = R, b, win
        txt = f"  diagnosis: convergence {R['convergence']} (late agitation {R['late_agitation']:.3f}), {R['invariant_function']}"
        if "periodic_component" in R: txt += f", period {R['periodic_component']['period']:.3f}"
        self._log(txt)
        self.learned = []
        if self.dictionary_mode != "auto": return
        Ztr = self.coords(self.X0[self.itr])
        for name, target, sh in targets:
            f, e = dic.fit_rff(Ztr, target[self.itr], self.rng, self.coords)
            self._log(f"  learned [{name}]: internal relative validation error {e:.3f}")
            if target.ndim == 1:
                self.candidates.append((name, (lambda f: lambda X: stack([f(X)]))(f)))
            else:
                self.candidates.append((name, f))
            if sh is not None:
                f2 = f.sharpened(sh[0], sh[1], 3.0)
                self.candidates.append((name + "_sharp", lambda X, f2=f2: stack([f2(X)])))
            self.learned.append((name, f, e))

    def _selection(self):
        ks = np.arange(1, self.H_target + 1)
        err, n = self._validation_floor(self.blocks, ks); J = rms(err)
        self._log(f"  selection (criterion: mean validation floor over t ≤ {self.H_target*self.dt:g}):")
        self._log(f"    {self.blocks[0][0]}: J = {J:.4f} (n={n})")
        hist = [dict(blocks=self.block_names, J=float(J), n=int(n))]
        rest = list(self.candidates); excluded = set()
        while rest:
            trials = []
            for c in rest:
                if c[0].replace("_sharp", "") in excluded: continue
                e, nn = self._validation_floor(self.blocks + [c], ks); trials.append((rms(e), nn, c))
            if not trials: break
            trials.sort(key=lambda t: t[0])
            for Jc, nn, c in trials: self._log(f"      + {c[0]}: J = {Jc:.4f} (n={nn})")
            Jb, nb, cb = trials[0]
            if Jb < J * (1 - self.threshold):
                self.blocks.append(cb); rest.remove(cb); J = Jb
                excluded.add(cb[0].replace("_sharp", ""))
                hist.append(dict(blocks=self.block_names, J=float(J), n=int(nb)))
                self._log(f"    -> added {cb[0]} (J = {J:.4f})")
            else:
                self._log(f"    -> stop: best gain {1 - Jb/J:.1%} < {self.threshold:.0%}"); break
        self.selection_history = hist

    def _final_selection(self):
        """The floor is the model error only if the full memory covers the horizon. Otherwise, the
        dictionaries along the greedy path are compared by the validation error of the full model."""
        everything = list(self.blocks); paths = [everything[:len(h["blocks"])] for h in self.selection_history]
        self.final_selection = None
        if self.H_mem >= self.H_target or len(paths) == 1:
            self._memory(); return
        self._log("  final selection (validation error of the model with memory, full horizon):")
        res = []
        for pth in paths:
            self.blocks = pth; self._memory(verbose=False)
            e = float(rms(self.validation_error(self.H_target * self.dt)[1:]))
            res.append(dict(blocks=[b[0] for b in pth], model_val_error=e, p=int(self.p), n=int(self.n)))
            self._log(f"    {[b[0] for b in pth]}: validation error {e:.4f} (p={self.p}, n={self.n})")
        i = int(np.argmin([r["model_val_error"] for r in res]))
        self.blocks = paths[i]; self._memory(); self.final_selection = res
        self._log(f"    -> kept {res[i]['blocks']}")

    def _memory(self, verbose=True):
        """Stage 2: correlations of the selected dictionary over the window, kernels Ω_l, choice of p."""
        d = self._dico(); itr = self.itr; Ntr = len(itr); S = self.data.states
        Phi0 = d(self.X0); self.Wh = self._ortho_basis(Phi0); E0 = Phi0 @ self.Wh; self.n = self.Wh.shape[1]
        E0tr = E0[itr]; C = [E0tr.T @ E0tr / Ntr]
        for k in range(1, self.H_mem + 1):
            Ek = d(S[itr, k]) @ self.Wh; C.append(E0tr.T @ Ek / Ntr)
            if k == 1: S1 = Ek.T @ Ek / Ntr
        self.C = C; self.Om = memory_kernels(C); self.cpsi = E0tr.T @ self.Y[0][itr] / Ntr
        self.eps = float(np.sqrt(max(np.linalg.eigvalsh(S1 - C[1].T @ C[1]).max(), 0)))
        self.norm_Om = [float(np.linalg.norm(O, 2)) for O in self.Om]
        self.p, self.p_table, self.radius_tol, self.memory_stable = choose_p(
            C, self.Om, E0[self.iva], self.cpsi, self.Y[:self.H_mem + 1][:, self.iva], self.H_target)
        if self.H_mem >= self.H_target:
            self.memory_stable = True    # no extrapolation over the target horizon
        if not verbose: return
        self._log(f"  memory: n={self.n}, residual ε={self.eps:.3f}, ||Ω_l|| l=0..5: {np.round(self.norm_Om[:6], 4).tolist()}")
        for t in self.p_table:
            self._log(f"    p={t['p']:3d}: extrapolation error (validation) {t['extrap_error']:.4f}, radius {t['radius']:.4f}"
                      + ("" if t["stable"] else "  [unstable]"))
        self._log(f"    -> p = {self.p}")
        if not self.memory_stable:
            self._warn("no truncation of the memory is stable: beyond the memory window "
                       f"(t > {self.H_mem*self.dt:g}) the error may grow slowly. Check the error bar.")

    def _asymptotic_diagnosis(self):
        """Long-time floor predicted for the selected V: Prop. 8(ii) (strong convergence) or weak
        extension √(c² − ||Pb||²). Not applicable to a limit cycle."""
        Phi0 = self._dico()(self.X0); Wh = self._ortho_basis(Phi0); Ev = Phi0[self.iva] @ Wh
        Gi = np.linalg.pinv(Ev.T @ Ev / len(self.iva), rcond=1e-10)
        Pb = Ev @ (Gi @ (Ev.T @ self.b[self.iva] / len(self.iva)))
        dd = dict(dist_b_V=float(rms(self.b[self.iva] - Pb)))
        strong = self.diag["convergence"] == "strong"
        if not strong:
            dd["weak_prediction"] = float(np.sqrt(max(np.mean(self.win[:, self.iva] ** 2) - np.mean(Pb ** 2), 0)))
        cyc = self.diag.get("periodic_component", {}).get("share", 0) > 0.9
        dd["predicted_asymptotic_floor"] = None if cyc else (dd["dist_b_V"] if strong else dd["weak_prediction"])
        self.diag.update(dd)

    # ------------------------------------------------------------------ prediction
    def coefficients(self, K, p=None, seed=None):
        """Coefficients c_k (in the orthonormal basis) such that ψ̂(x0, kΔ) = E(x0)·c_k."""
        p = self.p if p is None else p
        return [M @ self.cpsi for M in hybrid(self.C, self.Om, p, K, seed)]

    def _predict_steps(self, X0, K, p=None, seed=None):
        E = self._dico()(X0) @ self.Wh
        return np.array([E @ c for c in self.coefficients(K, p, seed)])

    def validation_error(self, horizon=None):
        """RMS error of the model at each time k·dt on the validation trajectories: (K+1,)."""
        K = self.H_obs if horizon is None else min(int(round(horizon / self.dt)), self.H_obs)
        pr = self._predict_steps(self.X0[self.iva], K)
        return rms(pr - self.Y[:K + 1][:, self.iva], axis=1)

    def _prepare_error(self):
        K = self.H_obs
        # direct projection: ψ(x_k) ≈ E(x0)·B_k, B_k fitted on training (possible up to the end of the
        # observations of ψ; it is the estimator of the floor, without memory)
        E0 = self._dico()(self.X0) @ self.Wh
        self.B_proj = E0[self.itr].T @ self.Y[:K + 1][:, self.itr].T / len(self.itr)    # (n, K+1)
        self.proj_residuals = (E0[self.iva] @ self.B_proj).T - self.Y[:K + 1][:, self.iva]
        self.proj_error = rms(self.proj_residuals, axis=1)
        pr = self._predict_steps(self.X0[self.iva], K)
        self.val_residuals = pr - self.Y[:K + 1][:, self.iva]           # (K+1, Nva)
        self.val_error = rms(self.val_residuals, axis=1)
        self.Z_tr = self.coords(self.X0[self.itr]); self.Z_va = self.coords(self.X0[self.iva])
        self.k_nn = int(max(5, min(self.neighbors, len(self.iva) // 10)))
        # "out of domain" threshold: 2 × the 99.5th percentile of the validation -> nearest training distance
        dva = self._min_dist(self.Z_va, self.Z_tr)
        self.domain_threshold = float(np.quantile(dva, 0.995)) * 2.0
        # calibration of the local bar (leave-one-out on the validation set)
        sig = self._local_error(self.Z_va, exclude_self=True)
        z = np.abs(self.val_residuals) / np.maximum(sig, 1e-300)
        self.coverage = {f"{s:g} sigma": float(np.mean(z[1:] <= s)) for s in (1, 2, 3)}

    @staticmethod
    def _min_dist(A, B):
        out = np.empty(len(A))
        for i in range(0, len(A), 2000):
            d2 = (A[i:i+2000] ** 2).sum(1)[:, None] - 2 * A[i:i+2000] @ B.T + (B ** 2).sum(1)[None]
            out[i:i+2000] = np.sqrt(np.maximum(d2.min(1), 0))
        return out

    def _local_error(self, Z, exclude_self=False, residuals=None):
        """σ_loc(x0, t): RMS of the validation residuals of the k nearest neighbours of x0: (K+1, M)."""
        R = self.val_residuals if residuals is None else residuals
        B = self.Z_va; k = self.k_nn; out = np.empty((R.shape[0], len(Z)))
        for i in range(0, len(Z), 1000):
            A = Z[i:i+1000]
            d2 = (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None]
            if exclude_self:
                idx = np.argsort(d2, 1)[:, 1:k + 1]
            else:
                idx = np.argpartition(d2, k, 1)[:, :k]
            out[:, i:i+1000] = np.sqrt(np.mean(R[:, idx] ** 2, axis=2))
        return out

    def evaluate_dictionary(self, extra=None, field=None):
        """Evaluate the selected dictionary and get advice to adjust it (see ``kmz.evaluation``).

        extra : list of (name, function) to test in addition (energy, conserved quantity, ...).
        field : vector field, if the equations are known (adds the Thm F certificate)."""
        from .evaluation import evaluate_dictionary
        return evaluate_dictionary(self, extra=extra, field=field)

    def predict(self, x0, horizon=None, error="local", method="memory"):
        """Predict ψ(x(t)) for t = 0, dt, ..., horizon, from the initial state x0.

        x0 : (d,) or (M, d). horizon : final time (default: the target horizon of the fit; one may go
            beyond, but the error is then no longer estimated and is NaN).
        error : 'local' (default, neighbours of x0 in the validation data) or 'global'
            (average over the whole validation set).
        method : 'memory' (default, Mori–Zwanzig model: correlations over the state window, memory
            extrapolated beyond, prediction possible beyond the data) or 'projection' (direct regression
            of ψ(x(t)) on Φ(x0), time by time: reaches the dictionary floor, but only up to the end of
            the observations of ψ).
        Returns a ``Prediction`` object (value, error, t, ...)."""
        x0 = np.asarray(x0, dtype=float); single = x0.ndim == 1
        X0 = x0[None] if single else x0
        if X0.shape[1] != self.d:
            raise ValueError(f"x0 must have {self.d} coordinates")
        K = self.H_target if horizon is None else int(round(horizon / self.dt))
        if method == "projection":
            if K > self.H_obs:
                raise ValueError(f"method='projection': impossible beyond t = {self.H_obs*self.dt:g} (end of the data)")
            val = ((self._dico()(X0) @ self.Wh) @ self.B_proj[:, :K + 1]).T
            res, eg = self.proj_residuals, self.proj_error
        elif method == "memory":
            val = self._predict_steps(X0, K); res, eg = self.val_residuals, self.val_error
        else:
            raise ValueError("method: 'memory' or 'projection'")
        Kv = min(K, self.H_obs)
        Z = self.coords(X0)
        if error == "local":
            sig = self._local_error(Z, residuals=res)[:Kv + 1]
        else:
            sig = np.repeat(eg[:Kv + 1, None], len(X0), 1)
        err = np.full_like(val, np.nan); err[:Kv + 1] = sig
        out = self._min_dist(Z, self.Z_tr) > self.domain_threshold
        if out.any():
            self._warn(f"{int(out.sum())} initial state(s) far from the training data: "
                       "the prediction and its error bar are not reliable")
        if K > self.H_obs:
            self._warn(f"beyond t = {self.H_obs*self.dt:g} (end of the data), the error is not estimated (NaN)")
        t = np.arange(K + 1) * self.dt
        g = np.full(K + 1, np.nan); g[:Kv + 1] = eg[:Kv + 1]
        if single:
            return Prediction(t, val[:, 0], err[:, 0], g, bool(out[0]), Kv * self.dt)
        return Prediction(t, val, err, g, out, Kv * self.dt)

    # ------------------------------------------------------------------ certificate (known equations)
    def certificate(self, field, times=(0.25, 0.5, 1.0, 2.0), **kw):
        """A posteriori certificate of Thm F on a(t) = ⟨ψ, T(t)ψ⟩ (see ``kmz.certificate``).
        Needs the vector field F (written with the operations of ``kmz.taylor``)."""
        from .certificate import certificate_F
        return certificate_F(self, field, times, **kw)

    # ------------------------------------------------------------------ summary
    def report(self):
        """Dictionary of all the choices made by the model (for JSON archiving)."""
        return dict(diagnosis=self.diag, blocks=self.block_names, selection_history=self.selection_history,
                    final_selection=self.final_selection, n=int(self.n), eps=self.eps, p=int(self.p),
                    memory_stable=bool(self.memory_stable), p_choice=self.p_table, norm_Omega=self.norm_Om[:30],
                    validation_error=self.val_error.tolist(), local_bar_coverage=self.coverage,
                    warnings=self.warnings)

    def summary(self):
        """Readable summary of what the model has learned."""
        D = self.diag; e = self.val_error; s0 = float(np.std(self.Y[0][self.iva]))
        L = ["KMZ model:",
             f"  long-time behaviour: convergence {D['convergence']}; {D['invariant_function']}"
             + (f"; period {D['periodic_component']['period']:.3g}" if "periodic_component" in D else ""),
             f"  selected dictionary: {', '.join(self.block_names)} ({self.n} functions)",
             (f"  memory: correlations known over the whole horizon (t ≤ {self.H_mem*self.dt:g}), no extrapolation"
              if self.H_mem >= self.H_target else
              f"  memory: correlations known up to t = {self.H_mem*self.dt:g}, then p = {self.p} memory terms"
              + ("" if self.memory_stable else " (WARNING: no stable truncation)")),
             f"  validation error (std of ψ: {s0:.3g}): t=0: {e[0]:.3g}; "
             f"t={self.H_target*self.dt/2:g}: {e[self.H_target//2]:.3g}; t={self.H_target*self.dt:g}: {e[self.H_target]:.3g}"]
        if D.get("predicted_asymptotic_floor") is not None:
            L.append(f"  predicted long-time floor (Prop. 8): {D['predicted_asymptotic_floor']:.3g}")
        L.append(f"  local error bar: {self.coverage['2 sigma']:.0%} of the validation errors fall within ± 2 bars")
        return "\n".join(L)

    def __repr__(self):
        return self.summary() if hasattr(self, "val_error") else "KMZ(not fitted)"
