"""
Rupture de barrage (équations de Saint-Venant 1D) apprise et prédite avec kmz.

La physique (utilisée UNIQUEMENT pour générer les données ; kmz ne voit jamais les équations)
    dh/dt + dq/dx = 0
    dq/dt + d(q^2/h + g h^2/2)/dx = 0          (fond plat, sans frottement)
    h : hauteur d'eau (m), q = h u : débit par unité de largeur (m^2/s).
    Canal de 3 km, barrage en x = 1000 m. A t = 0 le barrage disparaît : retenue de hauteur h0 à
    l'amont, eau au repos de hauteur h1 à l'aval (h1 = 0 : fond sec). Chaque trajectoire a son
    couple (h0, h1) ; 25 % des cas sont à fond sec.
    Solveur de référence : volumes finis, flux HLL (vitesses d'onde de Toro, fronts secs traités),
    ordre 2 (MUSCL minmod + Heun), CFL 0.45.

Solution exacte (pour juger le solveur ET le modèle)
    fond sec (h1 = 0) : solution de Ritter, h = (2 c0 - x'/t)^2 / (9 g) dans l'onde de détente ;
    fond mouillé      : solution de Stoker (détente + plateau h_m + ressaut mobile), calculée en
                        résolvant le problème de Riemann exact par dichotomie.

Ce que fait le script
    0. vérifie le solveur contre Ritter et Stoker (erreur L1 sur le profil final),
    1. simule N ruptures d'entraînement et N_test ruptures de test indépendantes,
    2. kmz apprend à prédire la hauteur à une station aval (par défaut x = 1300 m, 300 m sous le
       barrage) à partir de l'état initial seul, avec une barre d'erreur ; comparaison avec la vérité,
    3. indicateurs utiles pour l'alerte : hauteur maximale et heure d'arrivée de l'onde à la station,
    4. cas de référence : Ritter (fond sec) et Stoker (fond mouillé) pour quelques h0, kmz contre
       solveur contre solution analytique,
    5. évaluation du dictionnaire (conseils) et rapport d'analyse (Markdown + figures).

Mettre ce fichier dans kmz/examples/ et lancer depuis le dossier kmz/ :
    python examples/dam_break.py                  # ~1 à 2 minutes
    python examples/dam_break.py --quick          # ~20-40 s
    python examples/dam_break.py --noise 0.02     # 2 cm de bruit de mesure sur les données d'entraînement
    python examples/dam_break.py --gauge 1600     # autre position de la station (m)
    python examples/dam_break.py --no-analysis    # sans le rapport d'analyse (plus rapide)
Résultats dans examples/results_dam_break/ (résumé dans le terminal et dans summary.md, figures,
analysis/analysis.md).
"""
import argparse, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # dossier kmz/
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kmz import Data, KMZ, KMZWarning, analyze

G = 9.81
L_CHAN, NX = 3000.0, 200          # longueur du canal (m), nombre de mailles (dx = 15 m)
DX = L_CHAN / NX
X = (np.arange(NX) + 0.5) * DX    # centres des mailles
X_DAM = 1000.0                    # position du barrage
DURATION, RECORD = 60.0, 2.0      # durée simulée (s) et pas d'enregistrement (s)
H0_RANGE = (2.0, 10.0)            # hauteur de la retenue (m)
RATIO_RANGE = (0.02, 0.5)         # h1 / h0 pour les cas à fond mouillé
P_DRY = 0.25                      # proportion de cas à fond sec
H_DRY = 1e-8                      # seuil sec
# Avec h0 <= 10 m et 60 s, les ondes (tête de détente c0 t <= 600 m, front sec 2 c0 t <= 1200 m)
# n'atteignent pas les bords : la solution de Riemann exacte reste valable dans tout le canal.


# ----------------------------------------------------------------------------- solveur de référence
def _vel(h, q):
    return np.where(h > H_DRY, q / np.maximum(h, H_DRY), 0.0)


def hll_flux(hL, qL, hR, qR):
    """Flux HLL avec les estimations de vitesses de Toro (fronts secs compris)."""
    uL, uR = _vel(hL, qL), _vel(hR, qR)
    cL, cR = np.sqrt(G * hL), np.sqrt(G * hR)
    hs = (0.5 * (cL + cR) + 0.25 * (uL - uR)) ** 2 / G          # estimation de h* (détente double)
    qL_ = np.where(hs > hL, np.sqrt(0.5 * (hs + hL) * hs / np.maximum(hL, H_DRY) ** 2), 1.0)
    qR_ = np.where(hs > hR, np.sqrt(0.5 * (hs + hR) * hs / np.maximum(hR, H_DRY) ** 2), 1.0)
    sL = np.where(hL > H_DRY, uL - cL * qL_, uR - 2 * cR)
    sR = np.where(hR > H_DRY, uR + cR * qR_, uL + 2 * cL)
    FL = np.stack([qL, qL * uL + 0.5 * G * hL ** 2])
    FR = np.stack([qR, qR * uR + 0.5 * G * hR ** 2])
    UL, UR = np.stack([hL, qL]), np.stack([hR, qR])
    den = np.where(np.abs(sR - sL) > 1e-12, sR - sL, 1e-12)
    Fh = (sR * FL - sL * FR + sL * sR * (UR - UL)) / den
    F = np.where(sL >= 0, FL, np.where(sR <= 0, FR, Fh))
    return F, np.maximum(np.abs(sL), np.abs(sR))


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _rhs(h, q):
    """-dF/dx pour toutes les trajectoires (tableaux (N, NX)), bords transmissifs, MUSCL minmod."""
    hg = np.concatenate([h[:, :1], h[:, :1], h, h[:, -1:], h[:, -1:]], axis=1)
    ug = np.concatenate([_vel(h, q)[:, :1]] * 2 + [_vel(h, q)] + [_vel(h, q)[:, -1:]] * 2, axis=1)
    # reconstruction sur (h, u) : préserve la positivité de h
    dh = _minmod(hg[:, 1:-1] - hg[:, :-2], hg[:, 2:] - hg[:, 1:-1])
    du = _minmod(ug[:, 1:-1] - ug[:, :-2], ug[:, 2:] - ug[:, 1:-1])
    hc, uc = hg[:, 1:-1], ug[:, 1:-1]                    # NX + 2 mailles (avec une fantôme de chaque côté)
    hR_c, hL_c = hc + 0.5 * dh, hc - 0.5 * dh            # valeurs aux faces droite / gauche de chaque maille
    uR_c, uL_c = uc + 0.5 * du, uc - 0.5 * du
    hL, uL = hR_c[:, :-1], uR_c[:, :-1]                  # état à gauche de la face i+1/2
    hR, uR = hL_c[:, 1:], uL_c[:, 1:]
    F, smax = hll_flux(hL, hL * uL, hR, hR * uR)         # NX + 1 faces
    return -(F[:, :, 1:] - F[:, :, :-1]) / DX, smax.max()


def simulate_ic(h0, h1, duration=DURATION, record=RECORD, cfl=0.45):
    """Ruptures de barrage pour les tableaux h0 (N,), h1 (N,). Renvoie états (N, T, 2 NX) et temps."""
    h0 = np.atleast_1d(np.asarray(h0, float)); h1 = np.atleast_1d(np.asarray(h1, float))
    h = np.where(X[None] < X_DAM, h0[:, None], h1[:, None])
    q = np.zeros_like(h)
    T = int(round(duration / record)) + 1
    out = np.empty((len(h0), T, 2 * NX))
    out[:, 0] = np.concatenate([h, q], axis=1)
    t = 0.0
    for k in range(1, T):
        t_rec = k * record
        while t < t_rec - 1e-12:
            smax = max(np.sqrt(G * h.max()) * 3, 1e-6)
            dt = min(cfl * DX / smax, t_rec - t)
            (Rh, Rq), _ = _rhs(h, q)
            h1_, q1_ = np.maximum(h + dt * Rh, 0), q + dt * Rq      # Heun (RK2 TVD)
            q1_ = np.where(h1_ > H_DRY, q1_, 0)
            (Rh2, Rq2), _ = _rhs(h1_, q1_)
            h = np.maximum(0.5 * (h + h1_ + dt * Rh2), 0)
            q = 0.5 * (q + q1_ + dt * Rq2)
            q = np.where(h > H_DRY, q, 0)
            t += dt
        out[:, k] = np.concatenate([h, q], axis=1)
    return out, np.arange(T) * record


def draw_cases(N, rng):
    h0 = rng.uniform(*H0_RANGE, N)
    h1 = h0 * rng.uniform(*RATIO_RANGE, N)
    h1[rng.random(N) < P_DRY] = 0.0
    return h0, h1


# ----------------------------------------------------------------------------- solution exacte
def riemann_dam(h0, h1):
    """Etat intermédiaire (h_m, u_m) et vitesse du ressaut S du problème de rupture de barrage."""
    c0 = np.sqrt(G * h0)
    if h1 <= 0:
        return 0.0, 2 * c0, 2 * c0                        # Ritter : front sec à 2 c0
    f = lambda hm: 2 * (c0 - np.sqrt(G * hm)) - (hm - h1) * np.sqrt(0.5 * G * (hm + h1) / (hm * h1))
    a, b = h1, h0                                         # f(h1) > 0, f(h0) < 0
    for _ in range(100):
        m = 0.5 * (a + b)
        a, b = (m, b) if f(m) > 0 else (a, m)
    hm = 0.5 * (a + b); um = 2 * (c0 - np.sqrt(G * hm))
    return hm, um, hm * um / (hm - h1)


def exact_depth(x, t, h0, h1):
    """Hauteur exacte h(x, t) : Ritter si h1 = 0, Stoker sinon (x en m, t en s, scalaires ou tableaux)."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    xi = np.divide(x - X_DAM, t, out=np.zeros(np.broadcast(x, t).shape), where=t > 0)
    c0 = np.sqrt(G * h0); hm, um, S = riemann_dam(h0, h1)
    cm = np.sqrt(G * hm)
    fan = (2 * c0 - xi) ** 2 / (9 * G)
    h = np.where(xi <= -c0, h0, np.where(xi <= um - cm, fan, np.where(xi <= S, hm, h1)))
    return np.where(t > 0, h, np.where(x < X_DAM, h0, h1))


def arrival_time(t, y, base, height, frac=0.1):
    """Premier instant où y dépasse base + frac * height (NaN si jamais)."""
    above = y > base + frac * height
    out = np.full(y.shape[1], np.nan)
    any_ = above.any(0)
    out[any_] = t[above.argmax(0)[any_]]
    return out


# ----------------------------------------------------------------------------- programme
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0, help="bruit de mesure (m et m^2/s) sur l'entraînement")
    ap.add_argument("--gauge", type=float, default=1300.0, help="position de la station aval (m)")
    ap.add_argument("--no-analysis", action="store_true", help="sauter le rapport analyze()")
    args = ap.parse_args()
    N, N_test = (600, 200) if args.quick else (2000, 500)
    gauge = int(np.clip(args.gauge // DX, 0, NX - 1)); xg = X[gauge]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_dam_break")
    os.makedirs(out, exist_ok=True)
    lines = []
    def say(s=""):
        print(s, flush=True); lines.append(s)
    t0 = time.time()

    # 0. le solveur contre la solution exacte -----------------------------------------------
    say(f"Canal {L_CHAN/1000:g} km, {NX} mailles (dx = {DX:g} m), barrage à x = {X_DAM:g} m, "
        f"station à x = {xg:g} m.")
    say("\n0. Vérification du solveur (profil à t = 60 s, erreur L1 relative)")
    checks = [(6.0, 0.0), (6.0, 0.6), (6.0, 2.0)]
    Sref, tref = simulate_ic([c[0] for c in checks], [c[1] for c in checks])
    for j, (a, b) in enumerate(checks):
        he = exact_depth(X, tref[-1], a, b)
        e = np.sum(np.abs(Sref[j, -1, :NX] - he)) / np.sum(np.abs(he - b) + 1e-12)
        say(f"  h0 = {a:g} m, h1 = {b:g} m ({'Ritter' if b == 0 else 'Stoker'}) : {e:.2%}")

    # 1. données ----------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    say(f"\nSimulation de {N} ruptures d'entraînement + {N_test} de test...")
    h0_tr, h1_tr = draw_cases(N, rng)
    h0_te, h1_te = draw_cases(N_test, np.random.default_rng(2))
    S, t = simulate_ic(h0_tr, h1_tr)
    S_test, _ = simulate_ic(h0_te, h1_te)
    if args.noise > 0:
        S = S + args.noise * np.random.default_rng(3).standard_normal(S.shape)
        S[:, :, :NX] = np.maximum(S[:, :, :NX], 0)
    dt = float(t[1] - t[0])
    names = [f"h_{i}" for i in range(NX)] + [f"q_{i}" for i in range(NX)]
    data = Data(S, dt=dt, observable=gauge, names=names)
    test = Data(S_test, dt=dt, observable=gauge, names=names)
    say(f"  fait ({time.time() - t0:.0f} s) ; dimension de l'état {data.d}, {data.n_state_steps} pas de {dt:g} s")

    # 2. apprentissage et prédiction --------------------------------------------------------
    t1 = time.time()
    model = KMZ(verbose=False).fit(data)
    say(f"\n{model.summary()}\n  (apprentissage : {time.time() - t1:.0f} s)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", KMZWarning)
        P = model.predict(test.x0)
        Pp = model.predict(test.x0, method="projection")
    truth = test.Y                                  # (T, N_test)
    real = np.sqrt(np.mean((P.value - truth) ** 2, axis=1))
    real_p = np.sqrt(np.mean((Pp.value - truth) ** 2, axis=1))
    announced = np.sqrt(np.mean(P.error ** 2, axis=1))
    inside = np.mean(np.abs(P.value - truth) <= 2 * P.error)
    say(f"\n1. Hauteur à la station (x = {xg:g} m), {N_test} ruptures de test")
    for k in np.linspace(1, len(P.t) - 1, 6).astype(int):
        say(f"  t = {P.t[k]:4.0f} s   erreur réelle {real[k]:.3f} m (projection {real_p[k]:.3f})   "
            f"annoncée {announced[k]:.3f} m   écart-type de h : {truth[k].std():.3f} m")
    say(f"  part des erreurs réelles dans +/- 2 barres d'erreur locales : {inside:.1%}")
    rel = np.sqrt(np.mean((P.value - truth) ** 2)) / np.sqrt(np.mean((truth - truth.mean(1, keepdims=True)) ** 2))
    say(f"  erreur RMS globale / écart-type de h : {rel:.2f} (0 = parfait, 1 = prédire la moyenne)")

    # 3. indicateurs d'alerte -----------------------------------------------------------------
    hmax_err = np.abs(P.value.max(0) - truth.max(0))
    ta_true = arrival_time(P.t, truth, h1_te, h0_te - h1_te)
    ta_pred = arrival_time(P.t, P.value, h1_te, h0_te - h1_te)
    ok = np.isfinite(ta_true) & np.isfinite(ta_pred)
    say("\n2. Indicateurs d'alerte à la station")
    say(f"  hauteur maximale : erreur moyenne {hmax_err.mean():.3f} m (maxima entre {truth.max(0).min():.2f} "
        f"et {truth.max(0).max():.2f} m)")
    say(f"  arrivée de l'onde (h > h1 + 10 % de h0 - h1) : erreur médiane {np.nanmedian(np.abs(ta_pred - ta_true)[ok]):.1f} s, "
        f"90e centile {np.nanpercentile(np.abs(ta_pred - ta_true)[ok], 90):.1f} s "
        f"({ok.sum()}/{N_test} cas où l'onde arrive dans les deux)")

    # 4. cas de référence analytiques ---------------------------------------------------------
    say("\n3. Cas de référence : kmz contre solveur contre solution exacte (hauteur à la station)")
    ref = [(3.0, 0.0), (6.0, 0.0), (9.0, 0.0), (6.0, 0.6), (6.0, 2.4), (9.0, 1.8)]
    Sr, _ = simulate_ic([r[0] for r in ref], [r[1] for r in ref])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", KMZWarning)
        Pr = model.predict(Sr[:, 0])
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for j, ((a, b), ax) in enumerate(zip(ref, axs.ravel())):
        he = exact_depth(xg, Pr.t, a, b)
        e_k = np.sqrt(np.mean((Pr.value[:, j] - he) ** 2)); e_s = np.sqrt(np.mean((Sr[j, :, gauge] - he) ** 2))
        cov = np.mean(np.abs(Pr.value[:, j] - Sr[j, :, gauge]) <= 2 * Pr.error[:, j])
        kind = "Ritter" if b == 0 else "Stoker"
        say(f"  {kind:6s} h0 = {a:g} m, h1 = {b:g} m : RMS kmz/exacte {e_k:.3f} m, solveur/exacte {e_s:.3f} m, "
            f"dans +/- 2 barres {cov:.0%}")
        ax.plot(Pr.t, he, "k", lw=2, label=f"exacte ({kind})")
        ax.plot(Pr.t, Sr[j, :, gauge], "C1--", label="solveur VF")
        ax.plot(Pr.t, Pr.value[:, j], "C0", label="kmz")
        ax.fill_between(Pr.t, Pr.value[:, j] - 2 * Pr.error[:, j], Pr.value[:, j] + 2 * Pr.error[:, j],
                        color="C0", alpha=.25, label="± 2 barres")
        ax.set_title(f"{kind} : h0 = {a:g} m, h1 = {b:g} m"); ax.set_ylabel(f"h à x = {xg:g} m (m)")
    for ax in axs[1]: ax.set_xlabel("t (s)")
    axs[0, 0].legend(); fig.tight_layout(); fig.savefig(f"{out}/reference_cases.png", dpi=120); plt.close(fig)

    # profils du solveur contre exacte
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (a, b) in enumerate(checks):
        ax.plot(X, exact_depth(X, tref[-1], a, b), "k", lw=1.5)
        ax.plot(X, Sref[j, -1, :NX], "--", label=f"VF h0 = {a:g}, h1 = {b:g}")
    ax.axvline(xg, color="gray", ls=":", label="station"); ax.axvline(X_DAM, color="r", ls=":", label="barrage")
    ax.set_xlabel("x (m)"); ax.set_ylabel("h (m)"); ax.set_title("t = 60 s : solveur (tirets) contre exacte (noir)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/solver_check.png", dpi=120); plt.close(fig)

    # prédictions sur le test : meilleur, typique, pire
    err_j = np.sqrt(np.mean((P.value - truth) ** 2, axis=0))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, j, name in zip(axs, [np.argmin(err_j), np.argsort(err_j)[len(err_j) // 2], np.argmax(err_j)],
                           ["meilleur", "typique", "pire"]):
        ax.plot(P.t, truth[:, j], "k", label="vérité (solveur)")
        ax.plot(P.t, P.value[:, j], "C0", label="kmz")
        ax.fill_between(P.t, P.value[:, j] - 2 * P.error[:, j], P.value[:, j] + 2 * P.error[:, j],
                        color="C0", alpha=.25, label="± 2 barres")
        ax.set_title(f"{name} : h0 = {h0_te[j]:.1f} m, h1 = {h1_te[j]:.2f} m")
        ax.set_xlabel("t (s)"); ax.set_ylabel("h station (m)")
    axs[0].legend(); fig.tight_layout(); fig.savefig(f"{out}/predictions.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(P.t, real, label="erreur réelle (mémoire)"); ax.plot(P.t, real_p, ":", label="erreur réelle (projection)")
    ax.plot(P.t, announced, "--", label="erreur annoncée"); ax.plot(P.t, truth.std(1), "k:", lw=1, label="écart-type de h")
    ax.set_xlabel("t (s)"); ax.set_ylabel("RMS (m)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/errors.png", dpi=120); plt.close(fig)

    # 5. dictionnaire et analyse --------------------------------------------------------------
    say("\n4. Évaluation du dictionnaire")
    ev = model.evaluate_dictionary()
    say(ev.summary())
    ev.plot(f"{out}/evaluation.png")
    if not args.no_analysis:
        say("\n5. Rapport d'analyse")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", KMZWarning)
            rep = analyze(data, model=model, search_basins=False)
        say(rep.summary())
        rep.write(f"{out}/analysis")
    say(f"\nDurée totale {time.time() - t0:.0f} s. Figures et rapport dans {out}/")
    with open(f"{out}/summary.md", "w") as f:
        f.write("```\n" + "\n".join(lines) + "\n```\n")


if __name__ == "__main__":
    main()
