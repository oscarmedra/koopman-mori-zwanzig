"""Basin-of-attraction detector learned from trajectories.

Idea (Prop. 8 and weak extension): the late time average of the state,
    b(x0) = mean of x(t) over the end of the trajectories,
is (approximately) a Koopman invariant function: it only depends on the attractor x0 goes to.
If the values of b form clearly separated groups, each group is a basin and the basin indicators 1_B
are eigenfunctions with eigenvalue 1. These indicators are then learned as functions of x0 alone
(regression on random Fourier features), which gives the "basin function": it classifies a new
initial state without simulating its trajectory.

    res = find_basins(data)
    res.n_basins, res.accuracy
    res.classify(x0)                # basin index
    res.map(); res.plot("map.png")
"""
import numpy as np
from .linalg import rms
from .data import Data
from . import dictionary as dic


def _kmeans(Z, k, rng, trials=5, iters=60):
    """k-means (k-means++ initialisation), best of several trials."""
    best = None
    for _ in range(trials):
        C = [Z[rng.integers(len(Z))]]
        for _ in range(1, k):
            d2 = np.min(((Z[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
            C.append(Z[rng.choice(len(Z), p=d2 / d2.sum())] if d2.sum() > 0 else Z[rng.integers(len(Z))])
        C = np.array(C)
        for _ in range(iters):
            lab = np.argmin(((Z[:, None] - C[None]) ** 2).sum(-1), 1)
            Cn = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(Cn, C): break
            C = Cn
        inertia = ((Z - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]


class BasinResult:
    """Result of ``find_basins``.

    Main attributes
    ---------------
    n_basins : number of detected basins (1 = no distinct basins).
    attractors : (n_basins, d) late time average of the state in each basin.
    proportions : share of trajectories in each basin.
    accuracy : share of (non-ambiguous) validation trajectories whose basin is correctly predicted
        from x0 alone.
    confusion_matrix : (n_basins, n_basins), rows = true basin, columns = predicted basin.
    n_ambiguous : unconverged trajectories (late mean between two attractors), excluded.
    labels : basin of each trajectory (-1 = ambiguous).
    diagnostics : dict (convergence, separation, choice of the number of groups).
    """

    def summary(self):
        D = self.diagnostics
        L = [f"Basins: {self.n_basins} detected over {self.N} trajectories "
             f"(late agitation {D['late_agitation']:.3f}, separation {D['separation_ratio']:.3f})"]
        if self.n_basins == 1:
            L.append("  " + D["conclusion"])
        else:
            for j in range(self.n_basins):
                L.append(f"  basin {j}: {self.proportions[j]:.1%} of the trajectories, attractor ≈ "
                         f"{np.array2string(self.attractors[j][:4], precision=3)}{' ...' if self.attractors.shape[1] > 4 else ''}")
            L.append(f"  classification accuracy (validation, {self.n_validation} trajectories): {self.accuracy:.2%}")
            L.append(f"  ambiguous (unconverged) trajectories excluded: {self.n_ambiguous}")
        return "\n".join(L)

    __repr__ = summary

    # ------------------------------------------------------------------ classifying new states
    def scores(self, X):
        """Scores (M, n_basins) of the learned basin function (≈ basin indicators)."""
        X = np.atleast_2d(np.asarray(X, float))
        if self.n_basins == 1: return np.ones((len(X), 1))
        return self.function(X)

    def classify(self, X):
        """Predicted basin index for each initial state (int for a single state)."""
        single = np.asarray(X).ndim == 1
        lab = np.argmax(self.scores(X), 1)
        return int(lab[0]) if single else lab

    def margin(self, X):
        """Gap between the two best scores: small near basin boundaries."""
        s = np.sort(self.scores(X), 1)
        return s[:, -1] - (s[:, -2] if s.shape[1] > 1 else 0)

    # ------------------------------------------------------------------ map
    def map(self, n=200, plane=(0, 1), bounds=None, point=None):
        """Basin map on an n×n grid.

        In dimension 2: in the state plane (default bounds: those of the data).
        In dimension > 2: slice in the plane of the coordinates ``plane`` (standardised coordinates, or
        PCA components for large systems), the other coordinates being fixed at those of ``point``
        (default: the mean). Returns a dict (axis1, axis2, classes, margin, labels)."""
        co = self.coords; i, j = plane
        if co.mode == "state" and self.d == 2:
            lo, hi = self.x0.min(0), self.x0.max(0)
            if bounds is not None: lo, hi = np.array(bounds[0]), np.array(bounds[1])
            a1, a2 = np.linspace(lo[i], hi[i], n), np.linspace(lo[j], hi[j], n)
            G1, G2 = np.meshgrid(a1, a2, indexing="xy")
            X = np.zeros((n * n, 2)); X[:, i] = G1.ravel(); X[:, j] = G2.ravel()
            nm = self.names or [f"x{q}" for q in range(self.d)]
            names = (nm[i], nm[j])
        else:
            Z0 = co(self.x0)
            lo, hi = np.quantile(Z0, 0.01, 0), np.quantile(Z0, 0.99, 0)
            if bounds is not None: lo, hi = np.full(co.dim, bounds[0]), np.full(co.dim, bounds[1])
            zp = np.zeros(co.dim) if point is None else co(np.asarray(point, float)[None])[0]
            a1, a2 = np.linspace(lo[i], hi[i], n), np.linspace(lo[j], hi[j], n)
            G1, G2 = np.meshgrid(a1, a2, indexing="xy")
            Z = np.repeat(zp[None], n * n, 0); Z[:, i] = G1.ravel(); Z[:, j] = G2.ravel()
            X = co.inverse(Z)
            names = ((f"x{i} (standardised)", f"x{j} (standardised)") if co.mode == "state" else (f"PCA {i+1}", f"PCA {j+1}"))
        sc = self.scores(X)
        return dict(axis1=a1, axis2=a2, classes=np.argmax(sc, 1).reshape(n, n),
                    margin=self.margin(X).reshape(n, n), labels=names)

    def plot(self, filename=None, n=200, plane=(0, 1), ax=None, show_errors=True):
        """Draw the basin map (and the misclassified validation states)."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        c = self.map(n=n, plane=plane)
        if ax is None: _, ax = plt.subplots(figsize=(5.5, 5))
        cm = ListedColormap(["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc", "#fc9272", "#d9d9d9"][:max(self.n_basins, 1)])
        ax.pcolormesh(c["axis1"], c["axis2"], c["classes"], cmap=cm, shading="auto", vmin=-0.5,
                      vmax=max(self.n_basins, 1) - 0.5)
        if self.n_basins > 1:
            ax.contour(c["axis1"], c["axis2"], c["classes"], levels=np.arange(self.n_basins) + 0.5, colors="k", linewidths=0.8)
        if show_errors and self.n_basins > 1 and self.coords.mode == "state" and self.d == 2:
            X = self.x0[self.iva]; ok = self.true_va >= 0
            wrong = ok & (self.pred_va != self.true_va)
            ax.plot(X[wrong, plane[0]], X[wrong, plane[1]], "kx", ms=4, label=f"misclassified ({wrong.sum()})")
            amb = ~ok
            if amb.any(): ax.plot(X[amb, plane[0]], X[amb, plane[1]], ".", color="0.3", ms=2, label=f"unconverged ({amb.sum()})")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel(c["labels"][0]); ax.set_ylabel(c["labels"][1])
        title = f"{self.n_basins} basin(s)"
        if self.n_basins > 1: title += f", accuracy {self.accuracy:.1%}"
        ax.set_title(title)
        if filename: ax.figure.savefig(filename, dpi=130, bbox_inches="tight")
        return ax


def find_basins(data, late_fraction=0.25, k_max=6, validation_fraction=0.3, separation_threshold=0.3,
                min_fraction=0.01, n_pca=10, seed=0, verbose=False):
    """Learn the basin function from trajectories.

    Parameters
    ----------
    data : ``Data`` (only the states are used; the observable is ignored).
    late_fraction : final share of the trajectories over which the state is averaged (default 25 %).
        Trajectories must be long enough to have reached their attractor.
    k_max : maximal number of basins searched.
    separation_threshold : k groups are accepted if (95th percentile of distances to the centre) /
        (smallest gap between centres) is below this threshold (default 0.3). Otherwise: one basin.
    min_fraction : minimal size of a basin (share of the trajectories).
    Returns a ``BasinResult``."""
    if not isinstance(data, Data):
        raise TypeError("find_basins expects a Data object")
    rng = np.random.default_rng(seed)
    X = data.states; N, T1, d = X.shape; x0 = X[:, 0]
    if T1 < 8:
        raise ValueError("trajectories too short to judge the long-time behaviour")
    perm = rng.permutation(N); ntr = int((1 - validation_fraction) * N)
    itr, iva = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    i0 = int((1 - late_fraction) * T1)
    late = X[:, i0:]; b = late.mean(1)                       # late Cesàro mean (N, d)
    scale = float(np.sqrt(np.mean(x0.var(0)))) + 1e-300
    agit = float(rms(late - b[:, None]) / scale)
    Zb = (b - b[itr].mean(0)) / scale
    if d > 10:                                               # reduce (PCA) the descriptors
        _, s, Vt = np.linalg.svd(Zb[itr], full_matrices=False); Zb = Zb @ Vt[:10].T
    # ---- choice of the number of groups
    trials = []
    for k in range(2, k_max + 1):
        C, lab = _kmeans(Zb[itr], k, rng)
        fr = np.bincount(lab, minlength=k) / len(lab)
        dist = np.sqrt(((Zb[itr] - C[lab]) ** 2).sum(1))
        gaps = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[np.triu_indices(k, 1)]
        rap = float(np.quantile(dist, 0.95) / max(gaps.min(), 1e-300))
        trials.append(dict(k=k, ratio=rap, min_fraction=float(fr.min()), C=C))
        if verbose: print(f"  k={k}: spread/gap ratio {rap:.3f}, smallest group {fr.min():.1%}")
    valid = [e for e in trials if e["ratio"] < separation_threshold and e["min_fraction"] >= min_fraction]
    res = BasinResult()
    res.N, res.d, res.x0, res.itr, res.iva = N, d, x0, itr, iva
    res.names = data.names
    res.coords = dic.Coordinates(x0[itr], "state" if d <= 10 else "pca", n_pca)
    D = dict(late_agitation=agit, trials=[{k: v for k, v in e.items() if k != "C"} for e in trials])
    if not valid:
        res.n_basins = 1; res.attractors = b.mean(0)[None]; res.proportions = np.array([1.0])
        res.accuracy = 1.0; res.confusion_matrix = np.array([[len(iva)]])
        res.n_ambiguous = 0; res.n_validation = len(iva); res.function = None
        res.true_va = np.zeros(len(iva), int); res.pred_va = np.zeros(len(iva), int)
        res.labels = np.zeros(N, int)
        D["separation_ratio"] = min(e["ratio"] for e in trials) if trials else np.nan
        D["conclusion"] = ("no clearly separated groups in the long-time behaviour: a single basin "
                           "(a single attractor, a cycle, chaos, or a continuous invariant function "
                           "as in a conservative system)")
        res.diagnostics = D
        if verbose: print(res.summary())
        return res
    e = min(valid, key=lambda e: e["ratio"]); k = e["k"]; C = e["C"]
    D["separation_ratio"] = e["ratio"]
    dist_all = np.sqrt(((Zb[:, None] - C[None]) ** 2).sum(-1))
    lab = np.argmin(dist_all, 1)
    gap_min = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[np.triu_indices(k, 1)].min()
    ambiguous = dist_all.min(1) > gap_min / 3                # late mean between two attractors
    lab_v = np.where(ambiguous, -1, lab)
    # basin order: from the most to the least populated
    order = np.argsort(-np.bincount(lab[~ambiguous], minlength=k)); renum = np.empty(k, int); renum[order] = np.arange(k)
    lab_v = np.where(lab_v >= 0, renum[np.maximum(lab_v, 0)], -1)
    # ---- learning the basin function (indicators, from x0 alone)
    ok_tr = itr[lab_v[itr] >= 0]
    Y1 = np.eye(k)[lab_v[ok_tr]]
    f, err_rel = dic.fit_rff(res.coords(x0[ok_tr]), Y1, rng, res.coords)
    res.function = f; res.n_basins = k
    res.attractors = np.array([b[lab_v == j].mean(0) for j in range(k)])
    res.proportions = np.array([(lab_v == j).mean() for j in range(k)]) / max((lab_v >= 0).mean(), 1e-300)
    pred_va = np.argmax(f(x0[iva]), 1); true_va = lab_v[iva]; ok = true_va >= 0
    res.true_va, res.pred_va = true_va, pred_va
    res.accuracy = float(np.mean(pred_va[ok] == true_va[ok]))
    M = np.zeros((k, k), int)
    for a_, p_ in zip(true_va[ok], pred_va[ok]): M[a_, p_] += 1
    res.confusion_matrix = M; res.n_ambiguous = int(ambiguous.sum()); res.n_validation = int(ok.sum())
    res.labels = lab_v
    D["training_relative_error"] = err_rel
    D["conclusion"] = f"{k} clearly separated basins"
    res.diagnostics = D
    if verbose: print(res.summary())
    return res
