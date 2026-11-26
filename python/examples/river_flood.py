"""
Flood wave in a river reach (1D Saint-Venant equations) analysed with kmz.

The physics (used ONLY to generate data; kmz never sees the equations)
    dh/dt + dq/dx = 0
    dq/dt + d(q^2/h + g h^2/2)/dx = g h (S0 - Sf),   Sf = n^2 q|q| / h^(10/3)
    h: water depth (m), q: discharge per unit width (m^2/s), wide rectangular channel.
    Reach 10 km, 50 cells, bed slope S0, constant base inflow upstream, free outflow
    downstream. Each trajectory starts from the base flow plus one or two random flood bumps
    (height, position, width), as after a storm on the catchment.
    The difficulty: the Manning roughness n changes from one flood to the next (vegetation,
    sediment) and is NOT part of the measured state. kmz cannot see this hidden parameter:
    it must say honestly how uncertain its predictions become because of it.

What the script does
    1. simulates N training floods and N_test independent test floods,
    2. kmz learns to predict the depth at a downstream gauging station (km 9) from the
       initial state of the reach, with an error bar, and is compared with the truth,
    3. flood alert: probability that the station exceeds a threshold, real vs announced,
    4. dictionary evaluation and analysis report (Markdown + figures).

Put this file in kmz/examples/ and run from the kmz/ folder:
    python examples/river_flood.py                 # about 30 seconds
    python examples/river_flood.py --quick         # about 10 seconds
    python examples/river_flood.py --noise 0.02    # 2 cm of measurement noise on the training data
    python examples/river_flood.py --threshold 2.3 # your own alert depth (m)
Results in examples/results_river/ (summary in the terminal, figures, analysis/analysis.md).
"""
import argparse, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # kmz/ folder
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kmz import Data, KMZ, analyze

G = 9.81
L_REACH, NX = 10_000.0, 50            # m, cells
DX = L_REACH / NX
X = (np.arange(NX) + 0.5) * DX
S0, MANNING, Q0 = 5e-4, 0.03, 2.0     # slope, nominal Manning n, base discharge (m^2/s)
H_N = (Q0 * MANNING / np.sqrt(S0)) ** 0.6   # normal depth of the base flow
STATION = int(9_000 / DX)             # gauging station at km 9


def hll_flux(hL, qL, hR, qR):
    uL, uR = qL / hL, qR / hR
    cL, cR = np.sqrt(G * hL), np.sqrt(G * hR)
    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)
    FL = np.stack([qL, qL * uL + 0.5 * G * hL**2])
    FR = np.stack([qR, qR * uR + 0.5 * G * hR**2])
    UL, UR = np.stack([hL, qL]), np.stack([hR, qR])
    Fhll = (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL)
    return np.where(sL >= 0, FL, np.where(sR <= 0, FR, Fhll))


def step(h, q, dt, n_man):
    """One finite-volume step for all trajectories at once (arrays of shape (N, NX))."""
    # ghost cells: upstream imposed discharge Q0, downstream free outflow
    hg = np.concatenate([h[:, :1], h, h[:, -1:]], axis=1)
    qg = np.concatenate([np.full((h.shape[0], 1), Q0), q, q[:, -1:]], axis=1)
    F = hll_flux(hg[:, :-1], qg[:, :-1], hg[:, 1:], qg[:, 1:])   # (2, N, NX+1)
    h_new = h - dt / DX * (F[0, :, 1:] - F[0, :, :-1])
    q_new = q - dt / DX * (F[1, :, 1:] - F[1, :, :-1]) + dt * G * h * S0
    h_new = np.maximum(h_new, 0.05)
    # friction treated semi-implicitly (stable for any dt)
    q_new = q_new / (1 + dt * G * n_man**2 * np.abs(q_new) / h_new ** (7 / 3))
    return h_new, q_new


def simulate(N, duration_min=180, record_min=5, seed=0):
    """N floods, recorded every record_min minutes. Returns states (N, T, 2*NX), t in minutes."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.0, 2.0, (N, 1))            # flood bump height (m)
    xc = rng.uniform(500, 6000, (N, 1))          # position (m)
    w = rng.uniform(300, 1500, (N, 1))           # width (m)
    bump = A * np.exp(-((X[None] - xc) / w) ** 2)
    second = rng.random((N, 1)) < 0.5            # half the storms have a second flood bump
    bump += second * rng.uniform(0, 1.5, (N, 1)) * np.exp(-((X[None] - rng.uniform(500, 6000, (N, 1)))
                                                            / rng.uniform(300, 1500, (N, 1))) ** 2)
    n_man = rng.uniform(0.022, 0.045, (N, 1))    # HIDDEN: roughness differs from one flood to the next
    h = H_N + bump
    u_n = Q0 / H_N
    q = h * u_n * (h / H_N) ** (2 / 3)            # locally uniform-flow discharge
    dt_sim, n_rec = 10.0, int(record_min * 60 / 10.0)
    T = int(duration_min / record_min) + 1
    out = np.empty((N, T, 2 * NX))
    for k in range(T):
        out[:, k] = np.concatenate([h, q], axis=1)
        if k < T - 1:
            for _ in range(n_rec):
                h, q = step(h, q, dt_sim, n_man)
    return out, np.arange(T) * record_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0, help="measurement noise (m, and m^2/s) on training data")
    ap.add_argument("--threshold", type=float, default=None, help="flood alert depth at the station (m)")
    args = ap.parse_args()
    N, N_test = (800, 300) if args.quick else (3000, 1000)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_river"); os.makedirs(out, exist_ok=True)
    t0 = time.time()

    print(f"Reach 10 km, base depth {H_N:.2f} m, station at km 9. Simulating {N} + {N_test} floods...")
    S, t = simulate(N, seed=1)
    S_test, _ = simulate(N_test, seed=2)
    if args.noise > 0:
        S = S + args.noise * np.random.default_rng(3).standard_normal(S.shape)
    dt = float(t[1] - t[0])                      # minutes
    names = [f"h_{i}" for i in range(NX)] + [f"q_{i}" for i in range(NX)]
    data = Data(S, dt=dt, observable=STATION, names=names)
    test = Data(S_test, dt=dt, observable=STATION, names=names)
    print(f"  done ({time.time() - t0:.0f} s); state dimension {data.d}, {data.n_state_steps} steps of {dt:g} min")

    # 1. learn and predict -------------------------------------------------------------
    t1 = time.time()
    model = KMZ(verbose=True).fit(data)
    print(f"\n{model.summary()}\n  (fit: {time.time() - t1:.0f} s)")
    P = model.predict(test.x0)
    truth = test.Y                                # (T, N_test): depth at the station
    real = np.sqrt(np.mean((P.value - truth) ** 2, axis=1))
    announced = np.sqrt(np.mean(P.error ** 2, axis=1))
    inside = np.mean(np.abs(P.value - truth) <= 2 * P.error)
    print("\n1. Depth at the station, test floods")
    for k in np.linspace(1, len(P.t) - 1, 6).astype(int):
        print(f"  t = {P.t[k]:5.0f} min   real error {real[k]:.3f} m   announced {announced[k]:.3f} m")
    print(f"  share of real errors within +/- 2 local error bars: {inside:.1%}")
    print(f"  (std of the depth itself at the end: {truth[-1].std():.3f} m)")

    peak_err = np.abs(P.value.max(0) - truth.max(0))
    print(f"  peak depth: mean error {peak_err.mean():.3f} m (peaks range {truth.max(0).min():.2f}"
          f"-{truth.max(0).max():.2f} m)")
    t_peak_err = np.abs(P.t[P.value.argmax(0)] - P.t[truth.argmax(0)])
    print(f"  time of peak: median error {np.median(t_peak_err):.0f} min")

    # 2. flood alert ---------------------------------------------------------------------
    thr = args.threshold if args.threshold else float(np.quantile(truth.max(0), 0.7))
    from math import erf
    phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / np.sqrt(2))))
    k_pk = P.value.argmax(0); cols = np.arange(P.value.shape[1])
    p_exceed = phi((P.value[k_pk, cols] - thr) / np.maximum(P.error[k_pk, cols], 1e-6))
    real_alert = truth.max(0) > thr
    pred_alert = p_exceed > 0.5
    # the predicted curve is a conditional mean: it flattens peaks when n is unknown, hence missed alerts
    print(f"\n2. Flood alert: station above {thr:.2f} m during the 3 hours")
    print(f"  true alerts {real_alert.sum()}, detected {np.sum(pred_alert & real_alert)}, "
          f"missed {np.sum(~pred_alert & real_alert)}, false alarms {np.sum(pred_alert & ~real_alert)}")
    bins = np.linspace(0, 1, 6)
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (p_exceed >= a) & (p_exceed < b + 1e-9)
        if sel.sum():
            print(f"  announced probability {a:.1f}-{b:.1f}: {sel.sum():4d} floods, real frequency {real_alert[sel].mean():.2f}")

    # figures -----------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, j in zip(ax, [np.argmin(peak_err), np.argsort(peak_err)[len(peak_err) // 2], np.argmax(peak_err)]):
        a.plot(P.t, truth[:, j], "k", label="truth")
        a.plot(P.t, P.value[:, j], "C0", label="kmz")
        a.fill_between(P.t, P.value[:, j] - 2 * P.error[:, j], P.value[:, j] + 2 * P.error[:, j],
                       color="C0", alpha=.25, label="± 2 error bars")
        a.axhline(thr, color="r", ls=":", label="alert")
        a.set_xlabel("t (min)"); a.set_ylabel("depth at km 9 (m)")
    ax[0].set_title("best"); ax[1].set_title("typical"); ax[2].set_title("worst"); ax[0].legend()
    fig.tight_layout(); fig.savefig(f"{out}/predictions.png", dpi=120); plt.close(fig)
    fig, a = plt.subplots(figsize=(6, 4))
    a.plot(P.t, real, label="real error"); a.plot(P.t, announced, "--", label="announced error")
    a.set_xlabel("t (min)"); a.set_ylabel("RMS (m)"); a.legend()
    fig.tight_layout(); fig.savefig(f"{out}/errors.png", dpi=120); plt.close(fig)

    # 3. dictionary and analysis ------------------------------------------------------------
    print("\n3. Dictionary evaluation")
    ev = model.evaluate_dictionary()
    print(ev.summary())
    print("  Note: here the 'conserved quantity' the advice points to is the hidden roughness n: it does not change\n"
          "  during a flood and decides how fast the wave travels and flattens. Adding an estimate of it to the\n"
          "  state (e.g. from the first minutes of the hydrograph) is what would lower the floor.")
    ev.plot(f"{out}/evaluation.png")
    print("\n4. Analysis report")
    rep = analyze(data, model=model, search_basins=False)
    print(rep.summary())
    rep.write(f"{out}/analysis")
    print(f"\nTotal time {time.time() - t0:.0f} s. Figures and report in {out}/")


if __name__ == "__main__":
    main()
