"""Analyse a system from its trajectories: understand rather than predict.

    from kmz import analyze
    report = analyze(data)              # fits a KMZ model and looks for basins if needed
    print(report)                       # summary
    report.write("analysis/")           # Markdown report + figures

The report gathers what the data say about the system, in five parts:

1. **The data**: how many, how long, which region of state space they cover, and their effective
   dimension (a PDE discretised on 64 points may only explore 7 directions).
2. **The long-time behaviour**: convergence, basins (with a map), oscillation, mixing. These are the
   Koopman **invariant functions** (eigenvalue 1).
3. **The Koopman modes**: functions φ of the state that, along trajectories, are simply multiplied by a
   number λ at each step: φ(x(t+dt)) = λ φ(x(t)). Each mode has a decay rate and a frequency
   (λ = exp((−σ + 2iπf) dt)). They are computed in the model's dictionary (EDMD), each one is checked on
   the validation trajectories, and its weight in ψ is given. A mode that fails the checks is an
   artefact of the dictionary: it must not be interpreted.
4. **The memory**: how long the part of the system the dictionary does not see keeps influencing ψ
   (decay of the kernels Ω_l), and what would be lost by ignoring it (Markovian model).
5. **The predictability**: until when ψ is predictable from the initial state, and the dictionary floor.

The tool describes; it does not interpret. Linking a mode, a basin or a memory time to a physical
mechanism remains the researcher's work, and the report says so.
"""
import os, warnings
import numpy as np
from .linalg import rms
from .data import Data


# ====================================================================== Koopman modes (EDMD)
def koopman_modes(model, n_max=12):
    """Eigenvalues, eigenfunctions and weights in ψ of the one-step EDMD in the model's dictionary.

    Returns a list of dicts sorted by decreasing weight: lam, rate (σ, per unit time), frequency (f),
    period, half_life, weight (share of ψ carried by the mode), residual (relative gap between φ(x(t))
    and λ^k φ(x0) on the validation trajectories, at t ≈ 1), lag_gap (relative change of log(λ)/dt when
    the EDMD is redone with a lag t ≈ 1), reliable (residual < 0.2 and lag_gap < 0.1), v (eigenvector in
    the orthonormal basis). Conjugate pairs are merged (one mode, positive frequency, summed weight);
    the constant function (trivial invariant) is left out."""
    m = model; dt = m.dt; S = m.data.states
    A = m.C[1]                                                   # E(x1) ≈ E(x0) A (orthonormal basis)
    lam, V = np.linalg.eig(A)
    c = m.cpsi.astype(complex)
    a = np.linalg.lstsq(V, c, rcond=None)[0]                    # ψ = Σ a_j φ_j
    weight = np.abs(a) * np.linalg.norm(V, axis=0)
    weight = weight / max(weight.sum(), 1e-300)
    E0 = m.dictionary(m.X0[m.iva]) @ m.Wh
    k_max = min(m.H_mem, max(1, int(round(1.0 / dt))))         # test over t ≈ 1 (or the state window)
    # second test: a true mode keeps the same μ = log(λ)/dt when the EDMD is redone with lag k_max
    # (C(k) instead of C(1)). Artefact modes of the projection move (PKᵏP ≠ (PKP)ᵏ).
    mu_k = np.log(np.linalg.eigvals(m.C[k_max]) + 0j) / (k_max * dt) if k_max > 1 else None
    Ek = {}
    modes, seen = [], set()
    for j in np.argsort(-weight):
        if len(modes) >= n_max: break
        if j in seen: continue
        if abs(lam[j].imag) > 1e-9:                             # conjugate pair: handled once
            jc = int(np.argmin(np.abs(lam - np.conj(lam[j])) + (np.arange(len(lam)) == j)))
            seen.add(jc); w = weight[j] + weight[jc]
        else:
            w = weight[j]
        seen.add(j)
        v = V[:, j]; phi0 = E0 @ v
        if rms(np.abs(phi0 - phi0.mean())) < 1e-6 * max(rms(np.abs(phi0)), 1e-300):
            continue                                            # the constant function: trivial invariant
        kk = k_max
        while kk > 1 and abs(lam[j]) ** kk < 0.1: kk -= 1
        if kk not in Ek: Ek[kk] = m.dictionary(S[m.iva, kk]) @ m.Wh
        phik = Ek[kk] @ v
        res = float(rms(np.abs(phik - lam[j] ** kk * phi0)) / max(abs(lam[j]) ** kk * rms(np.abs(phi0)), 1e-300))
        mu = np.log(lam[j] + 0j) / dt if abs(lam[j]) > 1e-12 else complex(-np.inf, 0)
        sig = float(-mu.real); f = float(abs(mu.imag) / (2 * np.pi))
        if mu_k is not None and np.isfinite(mu):
            dmin = min(np.min(np.abs(mu_k - mu)), np.min(np.abs(mu_k - np.conj(mu))))
            gap = float(dmin / max(abs(mu), 0.1))
        else:
            gap = 0.0
        modes.append(dict(
            lam=complex(lam[j].real, abs(lam[j].imag)), modulus=float(abs(lam[j])), rate=sig, frequency=f,
            period=(1 / f if f > 1e-9 else None),
            half_life=(np.log(2) / sig if sig > 1e-9 else None), weight=float(w),
            residual=res, t_test=kk * dt, lag_gap=gap, reliable=bool(res < 0.2 and gap < 0.1), v=v,
            kind=("invariant" if abs(mu) < 0.01 else
                  ("growing" if abs(lam[j]) > 1.0 + 1e-3 else
                   ("oscillating" if f > 1e-9 and abs(lam[j].imag) > 1e-6 else "decaying")))))
    return modes


def _describe_mode(md):
    if md["kind"] == "invariant":
        return "invariant (λ ≈ 1): a quantity conserved along trajectories (invariant function)"
    s = []
    if md["period"] is not None and md["kind"] == "oscillating":
        s.append(f"oscillates with period {md['period']:.3g}")
    neutral = max(0.01, 0.05 * 2 * np.pi * md["frequency"])
    if abs(md["rate"]) <= neutral:
        s.append("without visible damping (neutral mode, typical of a limit cycle or a conserved quantity)")
    elif md["rate"] > 0:
        s.append(f"decays (half-life {md['half_life']:.3g})")
    else:
        s.append("grows (apparent instability: suspicious if the system is bounded)")
    return ", ".join(s)


# ====================================================================== the report
class AnalysisReport:
    """Result of ``analyze``: sections, numbers, figures (``write``), summary (``summary``)."""

    def summary(self):
        L = ["Analysis of the system from the data", "=" * 37, ""]
        for title, lines in self.sections:
            L.append(title); L.append("-" * len(title)); L += lines; L.append("")
        L.append("Summary"); L.append("-------"); L += [f"- {s}" for s in self.findings]; L.append("")
        L.append("Limitations of this analysis"); L.append("----------------------------"); L += [f"- {s}" for s in self.limitations]
        return "\n".join(L)

    __repr__ = summary

    # ------------------------------------------------------------------ figures
    def plot_spectrum(self, filename=None, ax=None):
        """Decay rate (horizontal) and frequency (vertical) of the modes; size = weight in ψ."""
        import matplotlib.pyplot as plt
        if ax is None: _, ax = plt.subplots(figsize=(5.5, 4))
        for md in self.modes:
            col = "C0" if md["reliable"] else "0.7"
            ax.scatter(md["rate"], md["frequency"], s=20 + 600 * md["weight"], color=col, alpha=0.8,
                       edgecolor="k" if md["reliable"] else "none")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("decay rate σ (1/time)"); ax.set_ylabel("frequency f (1/time)")
        ax.set_title("Koopman modes (size = weight in ψ; grey = unreliable)", fontsize=9)
        if filename: ax.figure.savefig(filename, dpi=130, bbox_inches="tight")
        return ax

    def plot_modes(self, filename=None, n=3):
        """Shape of the main reliable modes: map (dimension 2) or profile over the coordinates."""
        import matplotlib.pyplot as plt
        m = self.model; md = [x for x in self.modes if x["reliable"]][:n] or self.modes[:n]
        fig, axs = plt.subplots(1, len(md), figsize=(4.2 * len(md), 3.6), squeeze=False); axs = axs[0]
        for ax, x in zip(axs, md):
            if m.d == 2:
                lo, hi = m.X0.min(0), m.X0.max(0)
                g1, g2 = np.linspace(lo[0], hi[0], 120), np.linspace(lo[1], hi[1], 120)
                G = np.stack(np.meshgrid(g1, g2), -1).reshape(-1, 2)
                phi = (m.dictionary(G) @ m.Wh) @ x["v"]
                k = np.argmax(np.abs(phi)); phi = phi * np.exp(-1j * np.angle(phi[k]))   # reference phase
                im = ax.pcolormesh(g1, g2, phi.real.reshape(120, 120), cmap="RdBu_r", shading="auto")
                plt.colorbar(im, ax=ax, fraction=0.046)
                nm = m.data.names or ["x0", "x1"]; ax.set_xlabel(nm[0]); ax.set_ylabel(nm[1])
            else:
                prof = x["profile"]
                ax.plot(prof.real, label="real part")
                if np.abs(prof.imag).max() > 1e-9: ax.plot(prof.imag, label="imaginary part")
                ax.set_xlabel("state coordinate"); ax.legend(fontsize=7)
            ax.set_title(f"λ = {x['lam'].real:.3f}{x['lam'].imag:+.3f}i, weight {x['weight']:.0%}\n{x['kind']}"
                         + ("" if x["reliable"] else " (unreliable)"), fontsize=8)
        if filename: fig.savefig(filename, dpi=130, bbox_inches="tight")
        return fig

    def plot_memory(self, filename=None, ax=None):
        """Norm of the memory kernels ||Ω_l|| as a function of the lag l·dt."""
        import matplotlib.pyplot as plt
        if ax is None: _, ax = plt.subplots(figsize=(5.5, 3.5))
        no = np.array(self.model.norm_Om[1:]); t = np.arange(1, len(no) + 1) * self.model.dt
        ax.semilogy(t, no, "o-", ms=3)
        if self.memory.get("duration") is not None:
            ax.axvline(self.memory["duration"], color="C1", ls="--", lw=1, label="duration of influence")
            ax.legend(fontsize=8)
        ax.set_xlabel("lag"); ax.set_ylabel("||Ω_l||"); ax.set_title("Memory kernels", fontsize=9)
        if filename: ax.figure.savefig(filename, dpi=130, bbox_inches="tight")
        return ax

    def write(self, folder):
        """Write ``analysis.md`` and the figures into ``folder``. Returns the path of the report."""
        import matplotlib.pyplot as plt
        os.makedirs(folder, exist_ok=True)
        figs = {"The long-time behaviour": [], "The Koopman modes": [], "The memory": [], "The predictability": []}
        self.plot_spectrum(os.path.join(folder, "spectrum.png")); figs["The Koopman modes"].append(("Koopman modes", "spectrum.png"))
        self.plot_modes(os.path.join(folder, "modes.png")); figs["The Koopman modes"].append(("Shape of the main modes", "modes.png"))
        self.plot_memory(os.path.join(folder, "memory.png")); figs["The memory"].append(("Memory", "memory.png"))
        self.evaluation.plot(os.path.join(folder, "error.png")); figs["The predictability"].append(("Predictability", "error.png"))
        if self.basins is not None and self.basins.n_basins > 1:
            self.basins.plot(os.path.join(folder, "basins.png")); figs["The long-time behaviour"].append(("Basins", "basins.png"))
        plt.close("all")
        L = ["# Analysis of the system from the data", ""]
        for title, lines in self.sections:
            L.append(f"## {title}"); L.append("")
            L += ["    " + l if l.startswith("   ") else (l.strip() if l.startswith("  ") else l) for l in lines]
            for cap, f in figs.get(title, []):
                L.append(""); L.append(f"![{cap}]({f})")
            L.append("")
        L += ["## Summary", ""] + [f"- {s}" for s in self.findings] + ["", "## Limitations of this analysis", ""] + \
             [f"- {s}" for s in self.limitations]
        path = os.path.join(folder, "analysis.md")
        open(path, "w").write("\n".join(L) + "\n")
        return path


def analyze(data, model=None, basins=None, search_basins=True, verbose=False):
    """Analyse a system from its trajectories (see the module docstring).

    data : ``Data``. model : a ``KMZ`` already fitted on these data (otherwise one is fitted).
    basins : a ``BasinResult`` already computed (otherwise ``find_basins`` is called if the states cover
    enough time to judge the long-time behaviour)."""
    from .model import KMZ, KMZWarning
    from .basins import find_basins
    if not isinstance(data, Data):
        raise TypeError("analyze expects a Data object")
    D = data
    if model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", KMZWarning)
            model = KMZ(verbose=verbose).fit(D)
    m = model; r = AnalysisReport(); r.model = m; r.sections = []; find = []; lim = []
    dt = m.dt
    # ------------------------------------------------------------------ 1. data
    X0 = D.x0; s = np.linalg.svd(X0 - X0.mean(0), compute_uv=False); cum = np.cumsum(s ** 2) / (s ** 2).sum()
    dim99 = int(np.searchsorted(cum, 0.99) + 1)
    lines = [f"  {D.N} trajectories, state of dimension {D.d}, step dt = {dt:g}",
             f"  states known over t ≤ {D.n_state_steps*dt:g}, observable '{D.observable_name}' over t ≤ {D.n_obs_steps*dt:g}",
             f"  effective dimension of the initial states (99 % of the variance): {dim99}"]
    if D.d <= 6:
        lo, hi = X0.min(0), X0.max(0)
        lines.append("  region covered by the initial states: "
                     + " × ".join(f"[{a:.3g}, {b:.3g}]" for a, b in zip(lo, hi)))
    r.sections.append(("The data", lines))
    if dim99 < D.d:
        lim.append(f"The initial states only explore {dim99} directions out of {D.d}: everything said here holds for "
                   "this family of initial states, not for the whole state space.")
    # ------------------------------------------------------------------ 2. long times, basins
    dg = m.diag; lines = [f"  ψ: convergence {dg['convergence']} (agitation at the end of the data {dg['late_agitation']:.3f})",
                          f"  detected invariant function: {dg['invariant_function']}"]
    if "periodic_component" in dg:
        lines.append(f"  late periodic component: period {dg['periodic_component']['period']:.3g} "
                     f"({dg['periodic_component']['share']:.0%} of the power)")
    if basins is None and search_basins:
        if D.n_state_steps >= 8 and D.n_state_steps >= 0.5 * D.n_obs_steps:
            basins = find_basins(D)
        else:
            lines.append("  (basins not searched: the states are only known at the start of the trajectories)")
    r.basins = basins
    if basins is not None:
        lines.append("  " + basins.summary().replace("\n", "\n  "))
        if basins.n_basins > 1:
            big = int((basins.proportions >= 0.05).sum()); small = basins.n_basins - big
            txt = (f"The trajectories group into {basins.n_basins} distinct final behaviours (basins). The learned "
                   f"basin function separates them with {basins.accuracy:.1%} accuracy from the initial state alone "
                   "(map: basins.png); the errors concentrate near the boundaries.")
            if small:
                txt += (f" {small} of these groups each gather fewer than 5 % of the trajectories: they may be genuine "
                        "rare attractors, or slow (metastable) states not yet converged at the end of the data. "
                        "Longer trajectories would settle it.")
            find.append(txt)
    if dg["convergence"] == "strong" and (basins is None or basins.n_basins == 1):
        find.append("ψ converges to a limit that depends continuously on the initial state (or to a single value).")
    elif dg["convergence"] != "strong" and basins is not None and basins.n_basins > 1:
        find.append("ψ is not fully settled at the end of the data (slow convergence), but the trajectories already "
                    "group by basin.")
    elif dg["convergence"] != "strong":
        if dg.get("periodic_component", {}).get("share", 0) > 0.9:
            find.append(f"Everything converges to an oscillation of period ≈ {dg['periodic_component']['period']:.3g} "
                        "(probably a limit cycle): the phase on this cycle carries the information.")
        else:
            Yv = np.asarray(m.Y)[: m.H_obs + 1]; T = Yv.shape[0]; k1 = max(1, T // 10)
            drift = abs(Yv[-k1:].mean() - Yv[-2 * k1:-k1].mean()) / max(Yv[-2 * k1:].std(), 1e-300)
            spread = Yv[-1].std() / max(Yv.std(axis=1).max(), 1e-300)
            if drift > 0.2 or spread > 0.8:
                find.append("ψ is still evolving strongly at the end of the data (its mean or its spread over the "
                            "trajectories still changes, e.g. a wave that has not yet passed): the data are shorter "
                            "than the transient, so the long-time behaviour (convergence, cycle, chaos) cannot be "
                            "determined from them. Longer recordings would settle it.")
            else:
                find.append("ψ does not settle. Possible causes: a quantity that differs between trajectories and is "
                            "not in the state (hidden parameter), mixing oscillations (conservative system), or chaos. "
                            "The 'Predictability' section helps to decide.")
    r.sections.append(("The long-time behaviour", lines))
    # ------------------------------------------------------------------ 3. Koopman modes
    modes = koopman_modes(m); r.modes = modes
    E0 = m.dictionary(X0[m.itr]) @ m.Wh; Xc = X0[m.itr] - X0[m.itr].mean(0)
    for md in modes:
        phi = E0 @ md["v"]; phi = phi - phi.mean()
        md["profile"] = (Xc.T @ phi) / len(phi) / max(np.sqrt(np.mean(np.abs(phi) ** 2)), 1e-300)
    lines = ["  Modes with the largest weight in ψ (model dictionary, one-step EDMD). A mode is 'reliable' if it",
             "  passes two checks: it satisfies φ(x(t)) ≈ λ^k φ(x0) along the validation trajectories (residual",
             "  < 0.2), and it stays the same when the computation is redone with a lag t ≈ 1 (gap < 0.1).",
             "   weight   λ (± for a pair)       kind          half-life   period   residual (at t)   lag gap"]
    for md in modes[:8]:
        lam_txt = f"{md['lam'].real:+.4f}" + (f" ± {md['lam'].imag:.4f}i" if md['lam'].imag > 1e-9 else " " * 11)
        hl = f"{md['half_life']:.3g}" if md['half_life'] else "—"
        per = f"{md['period']:.3g}" if md['period'] and md['kind'] == 'oscillating' else "—"
        lines.append(f"   {md['weight']:5.1%}   {lam_txt}   {md['kind']:12s}  {hl:>8s}   {per:>7s}   "
                     f"{md['residual']:.3f} ({md['t_test']:.2g})      {md['lag_gap']:.3f}{'' if md['reliable'] else '   unreliable'}")
    r.sections.append(("The Koopman modes", lines))
    reliable = [md for md in modes if md["reliable"] and md["weight"] > 0.02]
    if reliable:
        desc = "; ".join(f"a mode that {_describe_mode(md)} ({md['weight']:.0%} of ψ)" if md["kind"] != "invariant"
                         else f"an {_describe_mode(md)} ({md['weight']:.0%} of ψ)" for md in reliable[:3])
        find.append(f"Reliable Koopman modes carrying ψ: {desc}.")
    else:
        find.append("No reliable Koopman mode carries a significant part of ψ: the dictionary does not contain an "
                    "eigenfunction of the dynamics (typical of chaos or of a too poor dictionary).")
    n_grow = sum(1 for md in modes[:8] if md["kind"] == "growing")
    if n_grow:
        lim.append(f"{n_grow} of the first modes {'has' if n_grow == 1 else 'have'} |λ| > 1 (growing): in a bounded or dissipative system this is an "
                   "artefact of the projection, not a real instability.")
    n_bad = sum(1 for md in modes[:8] if not md["reliable"])
    if n_bad:
        lim.append(f"{n_bad} of the first {min(8, len(modes))} modes fail one of the two reliability checks: they are "
                   "probably artefacts of the projection onto the dictionary (EDMD). Do not interpret them as "
                   "frequencies or damping rates of the system. Local modes around an attractor (e.g. the damped "
                   "oscillation at the bottom of a well) may be easier to see by analysing one basin at a time: "
                   "analyze(data.subset(basin_indices)).")
    # ------------------------------------------------------------------ 4. memory
    no = np.array(m.norm_Om); mem = {}
    lines = [f"  one-step projection residual ε = {m.eps:.3f} (share of the dynamics leaving the dictionary)"]
    if len(no) > 2:
        ref = no[1:].max(); above = np.nonzero(no[1:] > 0.1 * ref)[0]
        duration = (above[-1] + 1) * dt if len(above) else None
        l = np.arange(1, len(no)); ok = no[1:] > 1e-12
        slope = np.polyfit(l[ok], np.log(no[1:][ok]), 1)[0] if ok.sum() > 3 else 0.0
        mem = dict(duration=duration, characteristic_time=(-dt / slope if slope < -1e-6 else None))
        lines.append(f"  ||Ω_l|| for l = 1..5: {np.array2string(no[1:6], precision=3)}")
        ratios = no[2:min(len(no), 12)] / np.maximum(no[1:min(len(no), 12) - 1], 1e-300)
        mem["exploding"] = bool(len(ratios) >= 3 and np.median(ratios) > 2.0)
        if mem["exploding"]:
            lines.append(f"  WARNING: the kernels grow geometrically (×{np.median(ratios):.3g} per step): the projected "
                         "dynamics is numerically unstable in this dictionary. The kernels and the durations derived "
                         "from them are not interpretable. Predictions inside the state window are not affected "
                         "(they use the measured correlations directly); extrapolation beyond it is not reliable.")
        if duration is not None and not mem["exploding"]:
            lines.append(f"  duration of influence of the memory (||Ω_l|| > 10 % of its maximum): t ≈ {duration:.3g}"
                         + (" (the whole window: the memory does not die out within the data)" if above[-1] + 2 >= len(no) else ""))
        if mem["characteristic_time"] is not None and not mem["exploding"]:
            lines.append(f"  mean decay of the memory: characteristic time ≈ {mem['characteristic_time']:.3g}")
    ev = m.evaluate_dictionary(); r.evaluation = ev
    K = min(m.H_target, m.H_mem)
    if ev.predictability_horizon is not None:            # beyond it, any model only predicts the mean
        K = max(2, min(K, int(round(ev.predictability_horizon / dt))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mk = m._predict_steps(m.X0[m.iva], K, p=0, seed=0)
    e_mk = rms(mk - m.Y[:K + 1][:, m.iva], axis=1); e_m = m.val_error[:K + 1]
    gain = float(rms(e_mk[1:]) / max(rms(e_m[1:]), 1e-300))
    mem["markov_over_model"] = gain
    markov_diverges = gain > 1e3 or not np.isfinite(gain)
    if markov_diverges:
        lines.append(f"  without memory (Markovian model), the predictions over t ≤ {K*dt:g} diverge (error × > 1000): "
                     "the memory is indispensable, the exact factor is not meaningful")
    else:
        lines.append(f"  without memory (Markovian model), the mean error over t ≤ {K*dt:g} would be multiplied by {gain:.2f}")
    lines.append(f"  truncation kept for extrapolation: p = {m.p}" + ("" if m.memory_stable else " (no stable truncation)"))
    r.memory = mem
    r.sections.append(("The memory", lines))
    if markov_diverges or mem.get("exploding"):
        find.append("The memory is indispensable: without it the model diverges. But the memory kernels themselves "
                    "are numerically unstable in this dictionary, so no memory duration can be read from them; "
                    "predictions are only trustworthy inside the state window (t ≤ "
                    f"{m.H_mem*dt:g}), where the measured correlations are used directly.")
        lim.append("Memory kernels unstable (geometric growth): typical when an important variable is missing from "
                   "the dictionary (hidden parameter, too few PCA components). Do not extrapolate beyond the data.")
    elif gain > 1.5:
        find.append("The memory matters: the part of the system that the dictionary does not see keeps acting on ψ"
                    + (f" for t ≈ {mem['duration']:.3g}" if mem.get("duration") else "")
                    + f"; ignoring it would multiply the error by {gain:.2f} (over t ≤ {K*dt:g}).")
    else:
        find.append(f"The memory matters little over t ≤ {K*dt:g} (without it, the error would be multiplied by {gain:.2f}): "
                    "the dictionary captures most of what is predictable in ψ"
                    + (", even though the memory kernels do not die out within the window." if mem.get("duration") and
                       mem["duration"] >= (len(no) - 2) * dt else "."))
    if not m.memory_stable:
        lim.append("No truncation of the memory is stable: beyond the state window, the model's predictions degrade "
                   "(see the BILAN, Nagumo case).")
    # ------------------------------------------------------------------ 5. predictability
    lines = [f"  dictionary: {', '.join(ev.blocks)} ({ev.n} functions)",
             f"  error at the end of the horizon (t = {ev.t[-1]:g}): {ev.model_error[-1]:.3g}; dictionary floor "
             f"{ev.floor[-1]:.3g}; std of ψ {ev.std[-1]:.3g}"]
    if ev.predicted_floor is not None:
        lines.append(f"  predicted long-time floor (Prop. 8): {ev.predicted_floor:.3g}")
    if ev.predictability_horizon is not None:
        lines.append(f"  predictability horizon (error > 90 % of the std): t ≈ {ev.predictability_horizon:.3g}")
        find.append(f"Beyond t ≈ {ev.predictability_horizon:.3g}, the initial state says almost nothing about ψ any more: "
                    "only the mean is predictable with this dictionary. Possible causes, the first being the most common "
                    "with field data: a parameter that differs between trajectories and is not measured in the state "
                    "(hidden parameter, e.g. a friction or a forcing); a dependence on the initial state that the "
                    "dictionary does not capture (e.g. which basin); mixing or chaos.")
    else:
        find.append(f"ψ remains predictable over the whole studied duration (final error {ev.model_error[-1]:.2g} versus a "
                    f"std of {ev.std[-1]:.2g}).")
    r.sections.append(("The predictability", lines))
    # ------------------------------------------------------------------ limitations
    lim += ["Everything is measured in the model's dictionary: a phenomenon that its functions cannot represent is "
            "invisible here (this is the floor).",
            "The results hold for the region of state space covered by the data, and for the observed duration.",
            "Data assumed free of measurement noise: the behaviour with noise has not been tested.",
            "The tool describes what the data do; linking a mode, a basin or a memory time to a physical mechanism "
            "remains your work of interpretation."]
    r.findings, r.limitations = find, lim
    return r
