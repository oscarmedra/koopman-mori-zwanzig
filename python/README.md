# kmz: predict and analyse a dynamical system from its trajectories

`kmz` learns, from measured or simulated trajectories, to predict how a quantity ψ (a coordinate,
a spatial mean, …) evolves as a function of the initial state, **with an error bar**. It is built on
the Koopman operator and the Mori–Zwanzig formalism (see the companion paper and the formal proofs in `../lean/`).
**The equations of the system are never needed.** It also contains a basin-of-attraction detector, a
dictionary evaluator that gives advice, and an analysis tool to understand the system.

Version 0.2.2 (English API).
Dependency: numpy (matplotlib for figures, pytest for tests).

## In a few lines

```python
from kmz import Data, KMZ, find_basins, analyze

data   = Data(trajectories, dt=0.1, observable=0)   # trajectories: array (N, T+1, d)
model  = KMZ().fit(data)
pred   = model.predict(x0)            # pred.t, pred.value, pred.error (error bar)
ev     = model.evaluate_dictionary()  # judge the dictionary, get advice
basins = find_basins(data)            # basins.accuracy, basins.plot("map.png")
report = analyze(data)                # understand: invariants, Koopman modes, memory, predictability
```

The full tutorial, on damped Duffing, is in [`tutorial/TUTORIAL.md`](tutorial/TUTORIAL.md).

## Installation

Unzip, go into the `kmz/` folder (the one that contains `pyproject.toml`), then

```bash
pip install -e .                  # or simply add this folder to PYTHONPATH
pip install matplotlib pytest     # optional: figures and tests
```

## The data

`Data(states, dt, observable=0, obs=None, names=None)`

- `states`: array of shape **(N, T+1, d)**: N trajectories, T+1 times spaced by `dt`, d coordinates.
  A single long trajectory can also be cut into pieces to make N trajectories.
- `observable`: what you want to predict. An integer `i` = the i-th coordinate; or a function of the state
  (e.g. `lambda X: X.mean(-1)` for a spatial mean).
- `obs` (optional): the values of ψ, if they are measured **for longer** than the full states
  (common scenario: a sensor that runs for a long time, full states only at the beginning).
- `data.subset(indices)`: the same data restricted to some trajectories.

You need **hundreds to thousands of trajectories** starting from varied initial states: the model learns
a function of the initial state, so it must see many of them.

## What `KMZ().fit(data)` does

Three automatic stages:

1. **Dictionary.** The model writes its prediction as a combination of functions of the initial state
   (the *dictionary*). It starts from monomials (low dimension) or principal components (high dimension),
   looks at how ψ behaves at long times (convergence to a limit, basins, oscillation, chaos), learns the
   corresponding *invariant functions* (basin function, time average, phase) and keeps only what lowers the
   error on held-out trajectories.
2. **Memory.** The effect of the functions *outside* the dictionary is taken into account by memory kernels
   (Mori–Zwanzig), estimated from correlations. Beyond the window where the states are known, the memory is
   truncated to p terms; p is chosen by validation, among the stable truncations.
3. **Error.** 30 % of the trajectories are used for validation. The error is measured there at each time:
   this is the error bar.

Main options: `KMZ(dictionary="auto"|"base", base=f, candidates=[(name, f)], seed=0, verbose=False)` and
`fit(data, horizon=None, memory_window=None)`. `print(model)` summarises what was learned;
`model.report()` returns all the choices (for archiving).

## Predicting: `model.predict(x0, horizon=None, error="local", method="memory")`

Returns a `Prediction` object: `t`, `value`, `error`, `global_error`, `out_of_domain`, `interval(2)`, `plot()`.

- **Local error bar** (default): root-mean-square error of the 50 validation trajectories closest to x0.
  It is wider near a basin boundary, narrower elsewhere. On the benchmark, 94 % to 99 % of the real errors
  fall within ± 2 bars. `error="global"` gives the average over all x0.
- **Beyond the data**, the prediction continues (through the memory), but the bar is NaN: we do not know.
- If x0 is **far from the training data**, a `KMZWarning` says so.
- `method="projection"`: direct regression of ψ(x(t)) on the dictionary, time by time. It reaches the best
  possible error for this dictionary (the *floor*) but only as long as ψ is observed.

## Evaluating and adjusting the dictionary: `model.evaluate_dictionary()`

```python
ev = model.evaluate_dictionary(extra=[("energy", E)])   # E: X -> (..., 1), optional
print(ev)            # numbers + advice
ev.plot("evaluation.png")
```

The evaluation splits the model error into two parts (they add up in squares, by Pythagoras):

- **the dictionary part (the floor)**: what no model written with these functions can correct;
- **the memory part**: the cost of extrapolating the memory beyond the state window.

It also gives the predicted long-time floor (Prop. 8), the contribution of each block (by removing it),
the gain of each candidate function or of functions you propose, and a list of **advice**, for example
"the error comes from the memory, not from the dictionary: extend the window or use the projection",
"ψ converges to two values: a basin function would help", "chaotic behaviour: beyond t ≈ 8 only the mean
is predictable", "the late mean depends on the initial state: add a conserved quantity". The residual ε is
shown but **must not be used to choose**: it can rise while the error falls.

To add your own functions: `KMZ(candidates=[("energy", E)])` (kept only if they help) or `KMZ(base=f)`
(always kept).

## Basins: `find_basins(data)`

For each trajectory, the state is averaged over the end (last 25 %). If these averages form clearly
separated groups, each group is a basin. The model then learns the **basin function**: a function of the
initial state alone that tells which attractor it goes to (a Koopman invariant function, Prop. 8).

- `res.n_basins`, `res.attractors`, `res.proportions`, `res.labels`
- `res.accuracy` (on validation trajectories), `res.confusion_matrix`
- `res.classify(x0)`, `res.margin(x0)` (small near boundaries)
- `res.map()`, `res.plot("map.png")` (state plane in 2D, slice or PCA plane beyond)

Trajectories must be long enough to reach their attractor. A system with a single attractor, a limit
cycle, chaos or a conservative system gives **a single basin**, with a sentence explaining why.

## Analysing the system: `analyze(data)`

The goal here is no longer to predict but to **understand**. `analyze` gathers what the data say about the
system, in five parts, and writes a Markdown report with its figures:

```python
report = analyze(data)             # or analyze(data, model=model, basins=basins)
print(report)
report.write("analysis/")          # analysis.md + spectrum.png, modes.png, memory.png, error.png, basins.png
```

1. **The data**: number, duration, covered region, effective dimension of the initial states.
2. **The long-time behaviour**: convergence, basins (with a map), oscillation, mixing.
3. **The Koopman modes**: functions of the state that, along trajectories, are simply multiplied by λ at
   each step; each has a frequency and a damping rate. Each mode goes through **two reliability checks**:
   it must satisfy φ(x(t)) ≈ λᵏ φ(x0) on validation trajectories, and stay the same when the computation is
   redone with another lag. Modes that fail are dictionary artefacts and are marked "unreliable". On
   Van der Pol, the limit-cycle mode is found (period 6.64 for a true 6.66, no damping). On damped Duffing,
   no oscillating mode passes the checks: the global dictionary cannot represent the oscillation at the
   bottom of each well, and the tool says so instead of showing a false frequency.
4. **The memory**: how long the unresolved part of the system keeps acting on ψ (decay of the kernels
   Ω_l), and what would be lost without it (comparison with a Markovian model).
5. **The predictability**: horizon beyond which the initial state no longer says anything about ψ,
   dictionary floor.

The report ends with its **limitations**: data coverage, dictionary floor, unreliable modes, untested
noise. **The tool describes; the physical interpretation remains the researcher's**, and the report says
so. To study one basin on its own: `analyze(data.subset(indices))`.

## A serious example: `examples/kuramoto_ring.py`

A ring of 16 coupled phase oscillators (Kuramoto), state of dimension 32, observable = order parameter r.
It has 5 coexisting attractors (twisted states, winding number q = 0, ±1, ±2) and the ground truth is
known, so every output of the tool is checked against it.

    python examples/kuramoto_ring.py             # ~20-60 s
    python examples/kuramoto_ring.py --quick     # ~10-20 s
    python examples/kuramoto_ring.py --noise 0.05
    python examples/kuramoto_ring.py --file my_data.npy --dt 0.1 --observable 0

`my_data.npy` is an array of shape (N trajectories, L+1 times, d). Results go to `examples/results_<name>/`
(summary.md, figures, analysis/). Expected: error bars that match the real error (~99 % coverage), the 5
basins found exactly, but a weak prediction of the basin from x0 alone (close to the trivial baseline).

## Certificate (when the equations are known)

`model.certificate(field)` computes the a posteriori certificate of Theorem F on the autocorrelation
a(t) = ⟨ψ, T(t)ψ⟩. The vector field must be written with the operations of `kmz.taylor` (see
`kmz/systems.py`). It is only useful at short times and does not bound the prediction with memory
(BILAN, § 6).

## Benchmark results (10 systems)

See [`tests/benchmark_results.md`](tests/benchmark_results.md) (regenerated by `python tests/benchmark_report.py`).

## Limitations, honestly

- **The dictionary floor.** The model predicts the best approximation of ψ(x(t)) *within the dictionary*,
  not the exact solution. For chaos (Lorenz, Kuramoto–Sivashinsky), beyond the mixing time, it only
  predicts the mean, which is the best possible from x0.
- **Unstable memory (Nagumo).** When no truncation of the memory is stable, the tool reports it, but beyond
  the state window the error grows and ends **worse than a constant prediction** (Nagumo: 1.6 versus 0.5).
  The error bar shows it. Remedy: `method="projection"` while ψ is observed, or a longer state window.
- **Allen–Cahn**: the basin function would lower the floor, but makes the memory too long for the window;
  the tool does not keep it (and `evaluate_dictionary` explains why).
- **Noise not tested.** All tests use data without measurement noise.
- An error bar only exists where there are validation data (times and initial states).
- The thresholds (2 % gain, convergence, basin separation) are set by hand.

## Layout

```
kmz/
  data.py         data format (Data)
  model.py        KMZ (fit, predict, evaluate_dictionary, certificate) and Prediction
  dictionary.py   coordinates, monomials, learned functions, long-time diagnosis
  memory.py       memory kernels, choice of the truncation
  evaluation.py   dictionary evaluation and advice
  basins.py       find_basins
  analysis.py     analyze: understanding report (invariants, modes, memory, predictability)
  certificate.py  Theorem F certificate
  taylor.py       Taylor series (exact Lie derivatives for the certificate)
  linalg.py       small linear-algebra helpers
  systems.py      the 10 benchmark systems (to generate test data)
tests/            test_unit.py (~1 min), test_benchmark.py (~15 min), benchmark_report.py
tutorial/         damped Duffing tutorial (script, text, figures, analysis report)
```

Running the tests (from this folder): `pytest tests/test_unit.py` (fast); `pytest tests/test_benchmark.py`
(complete); `KMZ_SCALE=0.3 pytest tests/test_benchmark.py` (quick benchmark, less precise).

## Small glossary

- **Observable ψ**: the quantity to predict.
- **Dictionary**: the list of functions of the state with which the model writes its prediction.
- **Floor**: the smallest possible error with this dictionary.
- **Memory**: the correction that accounts for what the dictionary does not see (Mori–Zwanzig).
- **Invariant function**: a function of the state that does not change along trajectories (e.g. "which
  basin we are in"). Adding them to the dictionary lowers the long-time floor.
- **Koopman mode**: a function of the state multiplied by a fixed number λ at each time step; it carries
  a frequency and a damping rate.
- **Validation**: held-out trajectories, never used for learning, that serve to measure the error.
