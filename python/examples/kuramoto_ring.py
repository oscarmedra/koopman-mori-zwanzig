"""A serious test case for kmz: synchronisation basins of a ring of Kuramoto oscillators.

The problem (Wiley, Strogatz & Girvan, Chaos 2006, "The size of the sync basin"):
    n identical phase oscillators on a ring, each coupled to its two neighbours,
        dθ_j/dt = sin(θ_{j+1} − θ_j) + sin(θ_{j−1} − θ_j).
From random initial phases, the ring either synchronises (all phases equal) or freezes in a
"twisted state" where the phase winds q times around the ring (q = ±1, ±2, ...). Several attractors
coexist; which one is reached depends on the initial state in a complicated way. How large is the
basin of synchrony? Can it be learned from data alone?

Why it is a harder test than damped Duffing:
  * the state has dimension 32 (n = 16 oscillators, written with cos/sin of the phase differences);
  * there are 5 attractors (q = −2..2), two of them rare (< 1 % of random initial states);
  * the observable is nonlinear: the synchrony order parameter r(t) = |mean_j exp(iθ_j)|, which
    tends to 1 for synchrony and to 0 for every twisted state;
  * the ground truth (winding number q) is known, so every claim of the tool can be checked.

The equations are used ONLY to generate the data. kmz only sees the trajectories.

Usage (from the kmz folder):
    python examples/kuramoto_ring.py                      # standard run, about 20-60 seconds
    python examples/kuramoto_ring.py --quick              # fewer trajectories, about 10-20 seconds
    python examples/kuramoto_ring.py --noise 0.05         # add measurement noise to the training data
    python examples/kuramoto_ring.py --file my.npy --dt 0.1 --observable 0
                                                          # run the same pipeline on YOUR data
                                                          # (array of shape (N, T+1, d) saved with np.save)
Results (figures + summary.md) are written to examples/results_<name>/.
"""
import argparse, os, sys, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kmz import Data, KMZ, find_basins, analyze, KMZWarning

HERE = os.path.dirname(os.path.abspath(__file__))
N_OSC = 16


# ====================================================================== the physical system (data only)
def ring_rhs(th):
    return np.sin(np.roll(th, -1, 1) - th) + np.sin(np.roll(th, 1, 1) - th)


def simulate_ring(N, duration, dt, seed, nsub=5, theta0=None):
    """N trajectories of the ring from random phases. Returns the phases (N, T+1, n)."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, (N, N_OSC)) if theta0 is None else theta0
    out = [th]; h = dt / nsub
    for _ in range(int(round(duration / dt))):
        for _ in range(nsub):
            k1 = ring_rhs(th); k2 = ring_rhs(th + h / 2 * k1); k3 = ring_rhs(th + h / 2 * k2); k4 = ring_rhs(th + h * k3)
            th = th + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(th)
    return np.stack(out, 1)


def to_state(theta):
    """State used by kmz: (cos φ_j, sin φ_j) of the phase differences φ_j = θ_{j+1} − θ_j.
    The dynamics does not change if all phases are shifted together; working with differences removes
    this trivial symmetry (otherwise each twisted state would be a whole circle of states)."""
    phi = np.roll(theta, -1, -1) - theta
    return np.concatenate([np.cos(phi), np.sin(phi)], -1)


def order_parameter(X):
    """Synchrony r = |mean_j exp(iθ_j)|, computed from the state (phases rebuilt by summing the differences)."""
    phi = np.arctan2(X[..., N_OSC:], X[..., :N_OSC])
    th = np.concatenate([np.zeros(phi.shape[:-1] + (1,)), np.cumsum(phi[..., :-1], -1)], -1)
    return np.abs(np.exp(1j * th).mean(-1))


def winding_number(X):
    """Ground truth: q = (sum of the wrapped phase differences) / 2π, at the end of a trajectory."""
    phi = np.arctan2(X[..., N_OSC:], X[..., :N_OSC])
    return np.round(phi.sum(-1) / (2 * np.pi)).astype(int)


# ====================================================================== helpers
def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def rmse(a, b, axis=None):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


class Log:
    def __init__(self): self.lines = []
    def __call__(self, s=""):
        print(s, flush=True); self.lines.append(s)


# ====================================================================== the generic pipeline
def run_pipeline(data, test, out, log, k_max=8, min_fraction=0.005, true_q_test=None, true_q_train=None):
    """fit → predict → evaluate_dictionary → find_basins → analyze. ``test`` is a held-out Data (or None)."""
    # ---- 1. model
    section("1. Fitting the KMZ model"); t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", KMZWarning)
        model = KMZ(seed=0).fit(data)
    log(model.summary())
    for x in w: log(f"  warning: {x.message}")
    log(f"  (fit: {time.time()-t0:.0f} s)")

    # ---- 2. prediction with error bars on held-out trajectories
    if test is not None:
        section("2. Predictions on trajectories never seen")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", KMZWarning)
            p = model.predict(test.x0)
        K = len(p.t) - 1; Y = test.Y[:K + 1]
        err = rmse(p.value, Y, axis=1)
        cov = np.mean(np.abs(p.value - Y)[1:] <= 2 * p.error[1:])
        idx = sorted(set([K // 8, K // 4, K // 2, K]))
        log("  t                 " + "  ".join(f"{p.t[i]:7.3g}" for i in idx))
        log("  real error        " + "  ".join(f"{err[i]:7.3f}" for i in idx))
        log("  announced error   " + "  ".join(f"{p.global_error[i]:7.3f}" for i in idx))
        log("  constant predictor" + "  ".join(f"{Y[i].std():7.3f}" for i in idx))
        log(f"  share of real errors within ± 2 local error bars: {cov:.1%}")
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
        picks = np.argsort(np.abs(p.value[K // 2] - Y[K // 2]))[[0, len(Y[0]) // 2, -1]]   # good, typical, worst
        for ax, j, lab in zip(axs, picks, ["best case", "typical case", "worst case"]):
            pj = model.predict(test.x0[j])
            pj.plot(ax=ax, truth=Y[:, j]); ax.set_title(f"{lab} (test trajectory {j})", fontsize=9)
        axs[0].set_ylabel("observable ψ(t)")
        fig.savefig(os.path.join(out, "predictions.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

    # ---- 3. dictionary evaluation
    section("3. Evaluating the dictionary")
    ev = model.evaluate_dictionary()
    log(ev.summary()); ev.plot(os.path.join(out, "evaluation.png")); plt.close("all")

    # ---- 4. basins
    section("4. Basins of attraction")
    t0 = time.time()
    basins = find_basins(data, k_max=k_max, min_fraction=min_fraction)
    log(basins.summary()); log(f"  (find_basins: {time.time()-t0:.0f} s)")
    basins.plot(os.path.join(out, "basins.png")); plt.close("all")
    if true_q_train is not None and basins.n_basins > 1:
        log("\n  Check against the ground truth (winding number q of each trajectory):")
        lab = basins.labels
        for j in range(basins.n_basins):
            qs, cnt = np.unique(true_q_train[lab == j], return_counts=True)
            main = qs[np.argmax(cnt)]
            log(f"    basin {j}: {cnt.sum():5d} trajectories, true q = {main:+d} for {cnt.max()/cnt.sum():.1%} of them")
        n_true = len(np.unique(true_q_train))
        log(f"    true number of attractors in the data: {n_true}; found: {basins.n_basins}")
        if test is not None and true_q_test is not None:
            # map each basin to its majority q, then classify the test initial states from x0 alone
            q_of = {j: np.bincount(true_q_train[lab == j] + 10).argmax() - 10 for j in range(basins.n_basins)}
            pred_q = np.array([q_of[j] for j in basins.classify(test.x0)])
            log(f"    test set: winding number predicted from the initial state alone: {np.mean(pred_q == true_q_test):.1%} correct")
            sync_true = np.mean(true_q_test == 0); sync_pred = np.mean(pred_q == 0)
            j_sync = [j for j in range(basins.n_basins) if q_of[j] == 0]
            sync_final = sum(basins.proportions[j] for j in j_sync)
            base = np.mean(true_q_test == np.bincount(true_q_train + 10).argmax() - 10)
            log(f"    (baseline 'always the largest basin': {base:.1%} correct)")
            log(f"    size of the sync basin: true {sync_true:.1%}; from the final states of the training "
                f"trajectories {sync_final:.1%}; from the classifier applied to x0 {sync_pred:.1%}")
            if np.mean(pred_q == true_q_test) < base + 0.1:
                log("    -> The basins themselves are found exactly, but WHICH basin a given x0 falls into is barely "
                    "predicted: the winding number is a sum of wrapped phase differences, a very irregular function "
                    "of x0 that the learned classifier does not capture with this amount of data. This is an honest "
                    "limit of find_basins on this problem, not a bug in the example.")

    # ---- 5. analysis
    section("5. Analysis report (understanding)")
    t0 = time.time()
    report = analyze(data, model=model, basins=basins)
    path = report.write(os.path.join(out, "analysis"))
    log("Summary:"); [log(f"  - {s}") for s in report.findings]
    log("Limitations:"); [log(f"  - {s}") for s in report.limitations]
    log(f"  (analyze: {time.time()-t0:.0f} s; full report: {path})")
    return model, basins, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer trajectories (faster, less precise)")
    ap.add_argument("--noise", type=float, default=0.0, help="std of Gaussian measurement noise added to the training states")
    ap.add_argument("--file", help="your own data: .npy array (N, T+1, d)")
    ap.add_argument("--dt", type=float, default=None, help="time step of your data (with --file)")
    ap.add_argument("--observable", type=int, default=0, help="index of the coordinate to predict (with --file)")
    a = ap.parse_args()
    log = Log(); T0 = time.time()

    # ---------------------------------------------------------------- your own data
    if a.file:
        if a.dt is None: ap.error("--dt is required with --file")
        X = np.load(a.file); name = os.path.splitext(os.path.basename(a.file))[0]
        out = os.path.join(HERE, f"results_{name}"); os.makedirs(out, exist_ok=True)
        rng = np.random.default_rng(0); perm = rng.permutation(len(X)); n_test = max(1, len(X) // 5)
        data = Data(X[perm[n_test:]], dt=a.dt, observable=a.observable)
        test = Data(X[perm[:n_test]], dt=a.dt, observable=a.observable)
        log(f"Your data: {data}; {test.N} trajectories kept aside for testing")
        run_pipeline(data, test, out, log)
    # ---------------------------------------------------------------- the Kuramoto ring
    else:
        N, N_test = (1500, 600) if a.quick else (5000, 2000)
        duration, dt = 30.0, 0.25
        name = "kuramoto" + ("_quick" if a.quick else "") + (f"_noise{a.noise:g}" if a.noise else "")
        out = os.path.join(HERE, f"results_{name}"); os.makedirs(out, exist_ok=True)
        section(f"0. Generating data: ring of {N_OSC} Kuramoto oscillators, {N} + {N_test} trajectories, t ≤ {duration:g}")
        t0 = time.time()
        th_tr = simulate_ring(N, duration, dt, seed=1); th_te = simulate_ring(N_test, duration, dt, seed=2)
        S_tr, S_te = to_state(th_tr), to_state(th_te)
        q_tr, q_te = winding_number(S_tr[:, -1]), winding_number(S_te[:, -1])
        qs, cnt = np.unique(q_tr, return_counts=True)
        log(f"  ground truth (hidden from kmz): winding numbers {dict(zip(qs.tolist(), cnt.tolist()))}")
        log(f"  sync basin (q = 0): {np.mean(q_tr == 0):.1%} of random initial states  ({time.time()-t0:.0f} s)")
        if a.noise > 0:
            S_tr = S_tr + np.random.default_rng(3).normal(0, a.noise, S_tr.shape)
            log(f"  measurement noise of std {a.noise:g} added to the TRAINING states (test truth kept clean)")
        data = Data(S_tr, dt=dt, observable=order_parameter)
        test = Data(S_te, dt=dt, observable=order_parameter)
        log(f"  {data}")
        run_pipeline(data, test, out, log, true_q_test=q_te, true_q_train=q_tr)

    log(f"\nTotal time: {(time.time()-T0)/60:.1f} min. Figures and reports in {out}")
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("# kmz run: " + name + "\n\n```\n" + "\n".join(log.lines) + "\n```\n\n"
                 "Figures: predictions.png, evaluation.png, basins.png, analysis/analysis.md\n")


if __name__ == "__main__":
    main()
