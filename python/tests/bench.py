"""Evaluation of the package on the benchmark of the 10 systems (shared by the tests and benchmark_report.py).

Scenario (the one of recherche/outil/BILAN.md): ψ observed for a long time, full states over a short
window (40 steps, 30 for KS). The equations are only used to simulate; the model only sees the data.
Sizes are reduced compared with the BILAN (N = 6,000 for ODEs, 2,000 for PDEs) so that the tests run in
a few minutes.
"""
import functools, os, time, warnings
import numpy as np
from kmz import KMZ, find_basins, KMZWarning
from kmz.linalg import rms
from kmz.systems import simulate

SCALE = float(os.environ.get("KMZ_SCALE", "1"))      # factor on N (KMZ_SCALE=0.3: quick tests)

# subsample, duration of ψ, state window, target horizon, N, N test, basin subsample, basin duration
CONFIG = {
    "damped_duffing":       (1, 20.0, 4.0, 20.0, 6000, 3000, 5, 30.0),
    "van_der_pol":          (1, 40.0, 4.0, 20.0, 6000, 3000, 5, 40.0),
    "conservative_duffing": (1, 60.0, 4.0, 30.0, 6000, 3000, 5, 40.0),
    "lotka_volterra":       (1, 60.0, 4.0, 30.0, 6000, 3000, 5, 40.0),
    "pendulum":             (1, 60.0, 4.0, 30.0, 6000, 3000, 5, 40.0),
    "lorenz":               (1, 12.0, 0.8, 6.0, 6000, 3000, 5, 8.0),
    "burgers":              (2, 40.0, 8.0, 40.0, 2000, 1000, 10, 40.0),
    "ks":                   (4, 250.0, 30.0, 150.0, 2000, 1000, 4, 100.0),
    "allen_cahn":           (2, 80.0, 16.0, 80.0, 2000, 1000, 10, 80.0),
    "nagumo":               (2, 60.0, 16.0, 60.0, 2000, 1000, 10, 60.0),
}
BISTABLE = {"damped_duffing", "allen_cahn", "nagumo"}
CHAOS = {"lorenz", "ks"}


@functools.lru_cache(maxsize=None)
def evaluate(name):
    ss, duration, d_st, horizon, N, Nte, ss_b, d_b = CONFIG[name]
    N, Nte = max(300, int(N * SCALE)), max(300, int(Nte * SCALE))
    t0 = time.time(); r = dict(name=name)
    D = simulate(name, N, duration, seed=1, state_duration=d_st, subsample=ss)
    Dte = simulate(name, Nte, horizon, seed=2, state_duration=d_st, subsample=ss)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", KMZWarning)
        m = KMZ(seed=0).fit(D, horizon=horizon)
        r["fit_warnings"] = [str(x.message) for x in w]
    K = m.H_target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", KMZWarning)
        p = m.predict(Dte.x0)
    Y = Dte.Y[:K + 1]
    err_test = rms(p.value - Y, axis=1)
    # floor on the test set: projection fitted on the test set itself, with the selected dictionary
    E = m.dictionary(Dte.x0) @ m.Wh; G = np.linalg.pinv(E.T @ E / Nte, rcond=1e-10)
    floor = np.array([rms(E @ (G @ (E.T @ Y[k] / Nte)) - Y[k]) for k in range(K + 1)])
    z = np.abs(p.value - Y)[1:] / p.error[1:]
    r.update(model=m, err_test=err_test, err_val=m.val_error[:K + 1], floor=floor,
             std=Y.std(1), test_coverage_2s=float(np.mean(z <= 2)), H_mem=m.H_mem, K=K, dt=m.dt,
             blocks=m.block_names, p=m.p, memory_stable=m.memory_stable, fit_time=time.time() - t0)
    # basins: complete trajectories, subsampled (only the end matters)
    Db = simulate(name, N, d_b, seed=3, subsample=ss_b)
    t1 = time.time(); b = find_basins(Db, seed=0)
    r.update(basins=b, basin_time=time.time() - t1, total_time=time.time() - t0)
    return r
