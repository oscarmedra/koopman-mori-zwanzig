"""Evaluate the dictionary of a fitted model, and get advice to adjust it.

    ev = model.evaluate_dictionary()
    print(ev)                 # diagnosis and recommendations
    ev.plot("eval.png")       # model error = dictionary part + memory part
    ev = model.evaluate_dictionary(extra=[("energy", lambda X: ...)])   # test your own functions

What is measured, and why:
  * **The floor** (error of the best combination of the dictionary functions, time by time, on the
    validation set). No model written in this dictionary can do better. It is the "dictionary part"
    of the error.
  * **The memory part**: the model's prediction is also a combination of the dictionary functions, so
    (Pythagoras) error² = floor² + gap², where the gap comes from the memory being estimated then
    extrapolated (truncation to p terms, short correlation window).
  * **The predicted long-time floor** (Prop. 8): distance from the invariant function b (limit or Cesàro
    mean of ψ) to the dictionary. It tells how low the floor will go if one waits.
  * **The contribution of each block**: the block is removed and we look at how much the floor rises.
  * The **residual ε** is shown for information only: it can increase while the error decreases (adding
    a basin function raises ε and lowers the error), so it must not be used to choose.
"""
import numpy as np
from .linalg import rms


class DictionaryEvaluation:
    """Result of ``KMZ.evaluate_dictionary``: numbers, and recommendations (list of strings)."""

    def summary(self):
        L = ["Dictionary evaluation", "====================="]
        L.append(f"Dictionary: {', '.join(self.blocks)} ({self.n} functions)")
        L.append(f"Long-time behaviour of ψ: {self.behaviour}")
        L.append("")
        L.append("Validation error (std of ψ in brackets):")
        L.append("      t      model   floor (dictionary)   memory part")
        for j in self.times:
            L.append(f"  {self.t[j]:7.3g}   {self.model_error[j]:7.3g}   {self.floor[j]:9.3g}            {self.memory_part[j]:7.3g}"
                     f"     ({self.std[j]:.3g})")
        if self.predicted_floor is not None:
            L.append(f"Predicted long-time floor (Prop. 8): {self.predicted_floor:.3g}")
        if self.predictability_horizon is not None:
            L.append(f"Predictability horizon (error < 90 % of the std): t ≈ {self.predictability_horizon:.3g}")
        L.append(f"Residual ε = {self.eps:.3g} (for information only, do not use it to choose)")
        if self.block_contribution:
            L.append(""); L.append("Contribution of each block (rise of the mean floor if removed):")
            for name, g in self.block_contribution.items():
                L.append(f"  {name:28s} {g:+.1%}")
        if self.trials:
            L.append(""); L.append("Candidate functions (drop of the mean floor if added):")
            for name, g in self.trials.items():
                L.append(f"  {name:28s} {g:+.1%}")
        if self.certificate is not None:
            L.append(""); L.append(f"Certificate (Thm F): useful up to t = {self.certificate['useful_horizon']:g}")
        L.append(""); L.append("Advice")
        L.append("------")
        for i, r in enumerate(self.recommendations, 1):
            L.append(f"{i}. {r}")
        return "\n".join(L)

    __repr__ = summary

    def plot(self, filename=None, ax=None):
        """Model error, dictionary floor and memory part over time."""
        import matplotlib.pyplot as plt
        if ax is None: _, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(self.t, self.model_error, "C0", label="model error (validation)")
        ax.plot(self.t, self.floor, "C1", label="dictionary floor")
        ax.plot(self.t, self.memory_part, "C2--", label="part due to the memory")
        ax.plot(self.t, self.std, color="0.6", lw=0.8, label="std of ψ (constant prediction)")
        if self.predicted_floor is not None:
            ax.axhline(self.predicted_floor, color="C1", ls=":", lw=1, label="predicted floor (Prop. 8)")
        ax.axvline(self.t_memory, color="k", lw=0.6, ls=":")
        ax.text(self.t_memory, ax.get_ylim()[1] * 0.95, " end of the\n state window", fontsize=7, va="top")
        ax.set_xlabel("t"); ax.set_ylabel("RMS error"); ax.legend(fontsize=7, loc="best")
        if filename: ax.figure.savefig(filename, dpi=130, bbox_inches="tight")
        return ax


def _mean_floor(m, blocks, ks):
    e, n = m._validation_floor(blocks, ks)
    return float(rms(e)), n


def evaluate_dictionary(m, extra=None, field=None):
    """See ``KMZ.evaluate_dictionary``."""
    K = m.H_target; ks = np.arange(0, K + 1); dt = m.dt
    ev = DictionaryEvaluation()
    ev.blocks, ev.n, ev.eps, ev.t_memory = m.block_names, m.n, m.eps, m.H_mem * dt
    ev.t = ks * dt
    ev.model_error = m.val_error[:K + 1]
    ev.floor = m.proj_error[:K + 1]
    ev.memory_part = np.sqrt(np.maximum(ev.model_error ** 2 - ev.floor ** 2, 0))
    ev.std = m.Y[:K + 1][:, m.iva].std(1)
    ev.times = sorted(set([0, min(m.H_mem, K) // 2, min(m.H_mem, K), (K + min(m.H_mem, K)) // 2, K]))
    D = m.diag; ev.diagnosis = D
    ev.predicted_floor = D.get("predicted_asymptotic_floor")
    ev.behaviour = f"convergence {D['convergence']}; {D['invariant_function']}" + (
        f"; periodic component of period {D['periodic_component']['period']:.3g}" if "periodic_component" in D else "")
    sd = ev.std + 1e-300
    lost = np.nonzero(ev.model_error[1:] > 0.9 * sd[1:])[0]
    ev.predictability_horizon = float(ev.t[lost[0] + 1]) if len(lost) else None
    # ---- contribution of each block (ablation) and candidates
    J0, _ = _mean_floor(m, m.blocks, ks[1:])
    ev.block_contribution = {}
    for i in range(1, len(m.blocks)):
        Ji, _ = _mean_floor(m, m.blocks[:i] + m.blocks[i + 1:], ks[1:])
        ev.block_contribution[m.blocks[i][0]] = Ji / J0 - 1
    ev.trials = {}
    kept = {b.replace("_sharp", "") for b in m.block_names}
    for name, f in [c for c in m.candidates if c[0].replace("_sharp", "") not in kept] + list(extra or []):
        Jc, _ = _mean_floor(m, m.blocks + [(name, f)], ks[1:])
        ev.trials[name] = 1 - Jc / J0
    ev.certificate = m.certificate(field) if field is not None else None
    ev.recommendations = _recommendations(m, ev, extra)
    return ev


def _recommendations(m, ev, extra):
    R = []; D = m.diag; K = m.H_target; H = min(m.H_mem, K); dt = m.dt
    sd = ev.std + 1e-300
    fl_end = ev.floor[K]
    var_b = float(np.var(m.b[m.iva])); var_psi = float(np.mean(sd[-max(1, K // 5):] ** 2))
    inv_share = var_b / max(var_psi, 1e-300)      # share of the late variance carried by the invariant b
    # ---- 1. what dominates the error: the dictionary or the memory?
    if K > H:
        after = slice(H + 1, K + 1)
        pm, pd = float(rms(ev.memory_part[after])), float(rms(ev.floor[after]))
        if pm > 0.5 * pd:
            who = "comes mostly from the memory" if pm > pd else "comes to a large extent from the memory"
            txt = (f"Beyond the state window (t > {H*dt:g}), the error {who} "
                   f"(memory part {pm:.3g} versus floor {pd:.3g}): the dictionary is not the problem. "
                   f"Options: extend the state window (fit(..., memory_window=...)) if you have the data; "
                   f"or, up to t = {m.H_obs*dt:g}, predict with predict(x0, method='projection'), which reaches "
                   f"the floor without memory.")
            if not m.memory_stable:
                txt += " Here no truncation of the memory is stable: the error grows beyond the window."
            R.append(txt)
        else:
            R.append(f"The memory is well extrapolated beyond t = {H*dt:g} (memory part {pm:.3g}, floor {pd:.3g}): "
                     "to gain, the dictionary must be improved.")
    # ---- 2. invariant functions
    if "basin" in D["invariant_function"] and not any("basin" in b for b in ev.blocks):
        g = ev.trials.get("basin_sharp", ev.trials.get("basin", 0))
        if g < m.threshold:
            R.append(f"ψ converges to values that depend on the initial state (basins), but the learned basin function "
                     f"only brings {g:.1%}: the basin is not well predicted from the dictionary coordinates. "
                     "find_basins(data, n_pca=...) with more coordinates may separate them better.")
        else:
            R.append(f"ψ converges to two values depending on the initial state (basins), but the basin function was not "
                     f"kept: it would lower the floor by {g:.0%}, but the memory then becomes too long for the state "
                     "window. With a longer window, or method='projection', it would help. find_basins(data) gives "
                     "the basin map.")
    elif any("basin" in b for b in ev.blocks) and ev.predicted_floor is not None:
        R.append(f"The basin function is kept. The remaining long-time floor ({ev.predicted_floor:.3g}) comes "
                 "mostly from initial states misclassified near the boundary between basins: more trajectories "
                 "starting near that boundary would lower it (see find_basins(data).plot()).")
    elif D["convergence"] == "strong" and ev.predicted_floor is not None and ev.predicted_floor > 0.1 * np.sqrt(var_psi + var_b):
        R.append(f"ψ converges to a limit b(x0) that the dictionary approximates poorly (distance {ev.predicted_floor:.3g}): "
                 "the floor will not go below it. Add a function that approximates this limit "
                 "(e.g. with candidates=[...]), or a richer base dictionary.")
    if D["convergence"] != "strong":
        cyc = D.get("periodic_component", {}).get("share", 0) > 0.9
        if "periodic_component" in D and not any("harmonic" in b for b in ev.blocks):
            g = ev.trials.get("harmonic", 0)
            per = D["periodic_component"]["period"]
            if g < m.threshold:
                R.append(f"The late signal has a clear period ({per:.3g}), but the corresponding eigenfunction "
                         f"(harmonic) only brings {g:.1%}: when the period depends on the initial state, the phases "
                         "mix and this function loses its predictive power.")
            else:
                R.append(f"The late signal has a clear period ({per:.3g}); the harmonic function would lower the floor "
                         f"by {g:.0%} but was discarded because the model with memory got worse with it.")
        if cyc:
            R.append("Limit cycle: everything converges to an oscillation, without a limit. What matters is the phase "
                     "on the cycle; the learned harmonic function approximates it.")
        elif inv_share < 0.1 and fl_end > 0.8 * sd[K]:
            R.append(f"Mixing behaviour (probably chaos): the late mean of ψ hardly depends on the initial state "
                     f"(variance share {inv_share:.0%}). Beyond t ≈ "
                     f"{ev.predictability_horizon if ev.predictability_horizon is not None else K*dt:.3g}, only the mean "
                     "is predictable, and no dictionary will change that. Shorten the horizon, or enrich the dictionary "
                     "to gain at short times (higher-degree monomials, more PCA components).")
        elif inv_share >= 0.1:
            R.append(f"The late mean of ψ depends on the initial state ({inv_share:.0%} of the variance): there are "
                     "quantities that stay fixed along each trajectory: a hidden parameter that differs between trajectories "
                     "(friction, forcing, material property...) or a conserved quantity (energy). The long-time floor "
                     "will only drop if the dictionary contains them. If you can measure or estimate one, test it: "
                     "model.evaluate_dictionary(extra=[('energy', E)]), then KMZ(candidates=[...]).")
    # ---- 3. has the dictionary reached its asymptotic floor?
    if ev.predicted_floor is not None and abs(fl_end - ev.predicted_floor) < 0.1 * max(fl_end, 1e-300):
        R.append(f"At the end of the horizon, the floor ({fl_end:.3g}) has reached the value predicted by Prop. 8 "
                 f"({ev.predicted_floor:.3g}): only a better invariant function can lower it.")
    # ---- 4. blocks and candidates
    weak = [n for n, g in ev.block_contribution.items() if g < 0.01]
    if weak:
        R.append(f"Block(s) that barely help the floor: {', '.join(weak)} (less than 1 %). They may have been kept for "
                 "the memory; otherwise, a smaller dictionary is more robust.")
    for name, f in (extra or []):
        g = ev.trials[name]
        if g >= m.threshold:
            R.append(f"The function '{name}' lowers the floor by {g:.1%}: add it with KMZ(candidates=[('{name}', f)]) "
                     "(it will be kept if the full model improves).")
        else:
            R.append(f"The function '{name}' barely helps ({g:+.1%}): it is already (almost) in the dictionary, "
                     "or it says nothing about the future of ψ.")
    good = [(n, g) for n, g in ev.trials.items() if g >= m.threshold and n not in dict(extra or [])]
    if good:
        R.append("Candidates not kept although they lower the floor: "
                 + ", ".join(f"{n} ({g:.0%})" for n, g in good)
                 + ". They were discarded because the full model (with memory) got worse with them.")
    # ---- 5. reliability of the numbers
    if len(m.iva) < 500:
        R.append(f"Only {len(m.iva)} validation trajectories: the announced errors are imprecise "
                 "(a few tens of percent).")
    if m.eps > 0.5 and fl_end < 0.5 * sd[K]:
        R.append(f"The residual ε is large ({m.eps:.2f}) while the error is small: this is normal with learned "
                 "invariant functions; do not remove a block on this criterion alone.")
    if not R:
        R.append("Nothing to report: the dictionary and the memory are consistent with the data.")
    R.append("Reminder: the model has not been tested on noisy data; with measurement noise, compare the announced "
             "error with a constant prediction before trusting it.")
    return R
