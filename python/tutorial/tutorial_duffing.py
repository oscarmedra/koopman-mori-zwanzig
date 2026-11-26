"""Tutorial: damped Duffing (ẍ = −0.5ẋ + x − x³), from data only.

Run from the kmz folder:   python tutorial/tutorial_duffing.py
Takes about 2 minutes. Figures are written to tutorial/figures/.
"""
import os, sys, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kmz import Data, KMZ, find_basins, analyze

FIG = os.path.join(os.path.dirname(__file__), "figures"); os.makedirs(FIG, exist_ok=True)
f = lambda X: np.stack([X[:, 1], -0.5 * X[:, 1] + X[:, 0] - X[:, 0] ** 3], 1)


# ---------------------------------------------------------------- 1. The data
# In real life, these trajectories come from measurements. Here we generate them with the equations,
# but the model will know nothing about them: it only receives the array of states.
def simulate(N, duration, seed, dt=0.1, nsub=10, X=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (N, 2)) if X is None else X; out = [X]; h = dt / nsub
    for _ in range(int(round(duration / dt))):
        for _ in range(nsub):
            k1 = f(X); k2 = f(X + h / 2 * k1); k3 = f(X + h / 2 * k2); k4 = f(X + h * k3)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(X)
    return np.stack(out, 1)          # (N, T+1, 2): trajectories, times, coordinates (x, v)

traj = simulate(6000, 20.0, seed=0)
data = Data(traj, dt=0.1, observable=0, names=["x", "v"])   # we want to predict x(t)
print(data)

# ---------------------------------------------------------------- 2. The model
model = KMZ().fit(data)
print(model.summary())

# ---------------------------------------------------------------- 3. Predicting with an error bar
test = simulate(2000, 20.0, seed=1)                     # trajectories never seen
x0s = np.array([[0.3, 0.2], [-1.5, 1.8], [0.05, -0.1]])
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, x0 in zip(axs, x0s):
    pred = model.predict(x0)
    truth = simulate(1, 20.0, seed=0, X=x0[None].copy())[0, :, 0]   # true trajectory from x0, for comparison
    pred.plot(ax=ax, truth=truth)
    ax.set_title(f"x0 = ({x0[0]:g}, {x0[1]:g})"); ax.set_ylim(-2.3, 2.3)
axs[0].set_ylabel("x(t)")
fig.savefig(f"{FIG}/predictions.png", dpi=130, bbox_inches="tight")

# error on the test trajectories, compared with the announced error
p = model.predict(test[:, 0])
err_test = np.sqrt(np.mean((p.value - test[:, :, 0].T) ** 2, 1))
cov = np.mean(np.abs(p.value - test[:, :, 0].T)[1:] <= 2 * p.error[1:])
print(f"mean test error at t=5, 10, 20     : {err_test[50]:.3f}, {err_test[100]:.3f}, {err_test[200]:.3f}")
print(f"announced error (validation)       : {p.global_error[50]:.3f}, {p.global_error[100]:.3f}, {p.global_error[200]:.3f}")
print(f"share of test errors within ± 2 local error bars: {cov:.1%}")

# comparison: base dictionary only
base = KMZ(dictionary="base").fit(data)
pb = base.predict(test[:, 0])
err_base = np.sqrt(np.mean((pb.value - test[:, :, 0].T) ** 2, 1))
print(f"base dictionary only, test error at t=20: {err_base[200]:.3f}")
fig, ax = plt.subplots(figsize=(6.5, 3.4)); t = p.t
ax.plot(t, err_base, label="base dictionary (monomials ≤ 3)")
ax.plot(t, err_test, label="selected dictionary (+ basin function)")
ax.plot(t, p.global_error, "k:", label="announced error (validation)")
ax.plot(t, test[:, :, 0].std(0), color="0.6", lw=0.8, label="std of x(t) (constant prediction)")
ax.set_xlabel("t"); ax.set_ylabel("RMS error on x(t)"); ax.legend(fontsize=8)
fig.savefig(f"{FIG}/errors.png", dpi=130, bbox_inches="tight")

# ---------------------------------------------------------------- 4. Short memory window
# Harder scenario: the states are only known for t ≤ 4, x(t) up to 20.
short = Data(traj[:, :41], dt=0.1, observable=0, obs=traj[:, :, 0])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ms = KMZ().fit(short)
ps = ms.predict(test[:, 0])
err_s = np.sqrt(np.mean((ps.value - test[:, :, 0].T) ** 2, 1))
print(f"state window t ≤ 4: p = {ms.p}, test error at t=20: {err_s[200]:.3f} (announced {ps.global_error[200]:.3f})")

# ---------------------------------------------------------------- 5. Evaluate the dictionary, get advice
energy = lambda X: (0.5 * X[..., 1] ** 2 - 0.5 * X[..., 0] ** 2 + 0.25 * X[..., 0] ** 4)[..., None]
ev = ms.evaluate_dictionary(extra=[("energy", energy)])
print(ev)
ev.plot(f"{FIG}/evaluation.png")
pp = ms.predict(test[:, 0], method="projection")
err_p = np.sqrt(np.mean((pp.value - test[:, :, 0].T) ** 2, 1))
print(f"same short window, method='projection': test error at t=20: {err_p[200]:.3f}")

# ---------------------------------------------------------------- 6. The basins
basins = find_basins(data)
print(basins.summary())
true_lab = np.sign(test[:, -1, 0])                          # true basin: sign of x(20)
pred_b = basins.classify(test[:, 0])
sign = np.sign(basins.attractors[:, 0])[pred_b]
print(f"classification accuracy on the test set: {np.mean(sign == true_lab):.2%}")
basins.plot(f"{FIG}/basins.png")

# ---------------------------------------------------------------- 7. Analyse: understand rather than predict
report = analyze(data, model=model, basins=basins)
print(report)
report.write(os.path.join(os.path.dirname(__file__), "duffing_analysis"))
print("figures:", sorted(os.listdir(FIG)))
