"""
Cell-fate decision in a gene regulatory network, analysed with kmz.

The biology (used ONLY to generate data; kmz never sees the equations)
    G genes (default 4: A, B, C, D). Each gene activates itself and is repressed by all the
    others (the classic "self-activation + mutual repression" motif of cell differentiation,
    e.g. GATA1/PU.1 in blood cells). For each gene i, with mRNA m_i and protein p_i:
        dm_i/dt = 1.5 * p_i^4 / (0.5^4 + p_i^4) * 1 / (1 + (sum_{j≠i} p_j / Kr)^4) + 0.1 - m_i
        dp_i/dt = m_i - p_i
    State of dimension 2G (8 by default). The landscape ("Waddington landscape") has G + 1
    stable states: an undifferentiated progenitor (all genes low) and one fate per gene (that
    gene high, the others low). Separatrices between them are curved, high-dimensional surfaces.

Why it is a good test
    * multistability with 5 (or 6) attractors: find_basins must find them all, and say from the
      initial state alone which fate a cell will take (truth: the gene that ends up highest);
    * observable = protein level of gene A, as measured by a fluorescent reporter: its long-time
      value depends on the basin, so the dictionary needs the basin function;
    * second observable, not a coordinate: the "commitment index" (max protein - mean protein);
    * a check of the Koopman modes against the truth: inside the progenitor basin, the dynamics
      near the fixed point is linear, and its eigenvalues are computed exactly from the Jacobian.

Put this file in kmz/examples/ and run from the kmz/ folder:
    python examples/gene_fate.py                     # about 30 seconds
    python examples/gene_fate.py --quick             # about 15 seconds
    python examples/gene_fate.py --genes 5           # 6 attractors, state of dimension 10
    python examples/gene_fate.py --noise 0.02        # measurement noise on the training data
Results in examples/results_gene_fate/ (figures, analysis/analysis.md).
"""
import argparse, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # kmz/ folder
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kmz import Data, KMZ, find_basins, analyze

A_SELF, K_ACT, BASAL, HILL = 1.5, 0.5, 0.1, 4


def make_rhs(G, Kr):
    def rhs(Z):
        m, p = Z[..., :G], Z[..., G:]
        act = A_SELF * p**HILL / (K_ACT**HILL + p**HILL)
        others = p.sum(-1, keepdims=True) - p
        rep = 1.0 / (1.0 + (others / Kr) ** HILL)
        return np.concatenate([act * rep + BASAL - m, m - p], axis=-1)
    return rhs


def simulate(rhs, Z0, duration, dt_rec, dt=0.05):
    """RK4 for all cells at once; returns (N, T+1, d)."""
    n_sub = int(round(dt_rec / dt)); T = int(round(duration / dt_rec))
    out = np.empty((Z0.shape[0], T + 1, Z0.shape[1])); Z = Z0.copy(); out[:, 0] = Z
    for k in range(1, T + 1):
        for _ in range(n_sub):
            k1 = rhs(Z); k2 = rhs(Z + dt / 2 * k1); k3 = rhs(Z + dt / 2 * k2); k4 = rhs(Z + dt * k3)
            Z = Z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[:, k] = Z
    return out


def true_fate(Z_end, G):
    """0 = progenitor (all low), i+1 = fate of gene i."""
    p = Z_end[:, G:]
    return np.where(p.max(1) > 0.8, p.argmax(1) + 1, 0)


def jacobian(rhs, z, h=1e-6):
    d = len(z); J = np.empty((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = h
        J[:, j] = (rhs(z + e) - rhs(z - e)) / (2 * h)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--genes", type=int, default=4, choices=[3, 4, 5])
    ap.add_argument("--noise", type=float, default=0.0)
    args = ap.parse_args()
    G = args.genes; Kr = {3: 0.7, 4: 0.8, 5: 0.9}[G]
    N, N_test = (1500, 500) if args.quick else (4000, 1000)
    duration, dt_rec = 30.0, 0.5
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_gene_fate"); os.makedirs(out, exist_ok=True)
    genes = "ABCDE"[:G]
    names = [f"m{g}" for g in genes] + [f"p{g}" for g in genes]
    rhs = make_rhs(G, Kr); rng = np.random.default_rng(0); t0 = time.time()

    print(f"{G} genes, state of dimension {2*G}; simulating {N} + {N_test} cells over t = {duration:g}...")
    S = simulate(rhs, rng.uniform(0, 1, (N, 2 * G)), duration, dt_rec)
    S_test = simulate(rhs, rng.uniform(0, 1, (N_test, 2 * G)), duration, dt_rec)
    fate, fate_test = true_fate(S[:, -1], G), true_fate(S_test[:, -1], G)
    labels = ["progenitor"] + [f"fate {g}" for g in genes]
    print("  true fates: " + ", ".join(f"{labels[k]} {np.mean(fate == k):.1%}" for k in range(G + 1))
          + f"  ({time.time() - t0:.0f} s)")
    S_fit = S + args.noise * np.random.default_rng(1).standard_normal(S.shape) if args.noise > 0 else S

    # 1. observable 1: reporter protein of gene A --------------------------------------------
    print("\n1. Observable: protein of gene A (fluorescent reporter)")
    data = Data(S_fit, dt=dt_rec, observable=G, names=names)
    test = Data(S_test, dt=dt_rec, observable=G, names=names)
    t1 = time.time(); model = KMZ().fit(data)
    print(model.summary()); print(f"  (fit: {time.time() - t1:.0f} s)")
    P = model.predict(test.x0); truth = test.Y
    real = np.sqrt(np.mean((P.value - truth) ** 2, 1)); ann = np.sqrt(np.mean(P.error ** 2, 1))
    for k in np.linspace(1, len(P.t) - 1, 6).astype(int):
        print(f"  t = {P.t[k]:5.1f}   real error {real[k]:.3f}   announced {ann[k]:.3f}   (std of pA {truth[k].std():.3f})")
    print(f"  share of real errors within ± 2 local error bars: {np.mean(np.abs(P.value - truth) <= 2 * P.error):.1%}")
    commit_A = P.value[-1] > 0.8
    print(f"  'cell becomes fate A' read off the final prediction: {np.mean(commit_A == (fate_test == 1)):.1%} correct "
          f"(baseline 'never A': {np.mean(fate_test != 1):.1%})")
    P.plot(truth=truth[:, 0]); plt.gcf().savefig(f"{out}/prediction_pA.png", dpi=120); plt.close("all")

    # 2. dictionary evaluation -------------------------------------------------------------
    print("\n2. Dictionary evaluation")
    ev = model.evaluate_dictionary(); print(ev.summary()); ev.plot(f"{out}/evaluation.png")

    # 3. basins = cell fates ------------------------------------------------------------------
    print("\n3. Basins of attraction (cell fates)")
    t1 = time.time(); B = find_basins(data); print(B.summary()); print(f"  (find_basins: {time.time() - t1:.0f} s)")
    print("  Check against the true fates:")
    q_of = {}
    for j in range(B.n_basins):
        sel = B.labels == j
        if sel.sum() == 0: continue
        k = np.bincount(fate[sel], minlength=G + 1); q_of[j] = int(k.argmax())
        print(f"    basin {j}: {sel.sum():5d} cells, {labels[q_of[j]]} for {k.max() / sel.sum():.1%} of them")
    print(f"    true number of fates present: {len(np.unique(fate))}; found: {B.n_basins}")
    pred = np.array([q_of.get(j, -1) for j in B.classify(test.x0)])
    base = np.mean(fate_test == np.bincount(fate).argmax())
    print(f"    test cells, fate predicted from the initial state alone: {np.mean(pred == fate_test):.1%} correct "
          f"(baseline 'always the largest fate': {base:.1%})")
    try:
        B.plot(); plt.gcf().savefig(f"{out}/basins.png", dpi=120); plt.close("all")
    except Exception as e:
        print(f"    (basin plot skipped: {e})")

    # 4. observable 2: commitment index (a function, not a coordinate) ----------------------------
    print("\n4. Observable: commitment index = max protein - mean protein")
    def commitment(X):
        p = X[..., G:]
        return p.max(-1) - p.mean(-1)
    data_c = Data(S_fit, dt=dt_rec, observable=commitment, names=names)
    test_c = Data(S_test, dt=dt_rec, observable=commitment, names=names)
    m_c = KMZ().fit(data_c); Pc = m_c.predict(test_c.x0)
    rc = np.sqrt(np.mean((Pc.value - test_c.Y) ** 2, 1)); ac = np.sqrt(np.mean(Pc.error ** 2, 1))
    print(f"  dictionary: {', '.join(m_c.block_names)}")
    for k in [len(Pc.t) // 4, len(Pc.t) // 2, len(Pc.t) - 1]:
        print(f"  t = {Pc.t[k]:5.1f}   real error {rc[k]:.3f}   announced {ac[k]:.3f}   (std {test_c.Y[k].std():.3f})")

    # 5. Koopman modes inside the progenitor basin vs the exact linearisation -------------------
    print("\n5. Koopman modes of the progenitor basin vs the exact eigenvalues (Jacobian)")
    idx = np.nonzero(fate == 0)[0]
    z_star = simulate(rhs, S[idx[:1], -1], 200.0, 1.0)[0, -1]
    lam_true = np.exp(np.linalg.eigvals(jacobian(rhs, z_star)) * dt_rec)
    print("  exact λ = exp(μ dt) at the progenitor fixed point: "
          + ", ".join(f"{l.real:.4f}{'±%.4fi' % abs(l.imag) if abs(l.imag) > 1e-8 else ''}"
                      for l in sorted(set(np.round(lam_true, 6)), key=lambda z: -abs(z))))
    # exact Koopman eigenvalues near the fixed point: λ_i and the products λ_i·λ_j (degree 2)
    lam_all = np.concatenate([lam_true, np.outer(lam_true, lam_true)[np.triu_indices(len(lam_true))]])
    rep_p = analyze(data.subset(idx), search_basins=False)
    for md in rep_p.modes[:6]:
        near = lam_all[np.argmin(np.abs(lam_all - md["lam"]))]
        print(f"  kmz mode λ = {md['lam'].real:+.4f}{'±%.4fi' % md['lam'].imag if md['lam'].imag > 1e-9 else ''}"
              f"  weight {md['weight']:.1%}  {'reliable  ' if md['reliable'] else 'unreliable'}  nearest exact λ {near.real:.4f}"
              f"  (gap {abs(near - md['lam']):.4f})")
    print("  'nearest exact λ' includes the products λi·λj, which are also exact Koopman eigenvalues. The cells start\n"
          "  far from the fixed point (nonlinear region), so only part of the spectrum is linear: a gap < 0.05\n"
          "  is a match, a larger gap means the mode mixes nonlinear effects.")

    # 6. full analysis report -------------------------------------------------------------------
    print("\n6. Analysis report (whole data set, observable pA)")
    rep = analyze(data, model=model, basins=B); print(rep.summary()); rep.write(f"{out}/analysis")
    print(f"\nTotal time {time.time() - t0:.0f} s. Figures and report in {out}/")


if __name__ == "__main__":
    main()
