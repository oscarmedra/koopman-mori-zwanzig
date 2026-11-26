"""Fast interface tests (a few seconds each), on damped Duffing."""
import warnings
import numpy as np
import pytest
from kmz import Data, KMZ, find_basins, analyze, KMZWarning, Prediction
from kmz.systems import simulate, BENCHMARK


@pytest.fixture(scope="module")
def data():
    return simulate("damped_duffing", N=2000, duration=15.0, seed=0)


@pytest.fixture(scope="module")
def model(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", KMZWarning)
        return KMZ(seed=0).fit(data)


def test_data_shapes():
    X = np.random.default_rng(0).normal(size=(10, 6, 3))
    D = Data(X, dt=0.5, observable=2)
    assert (D.N, D.d, D.n_state_steps, D.n_obs_steps) == (10, 3, 5, 5)
    assert np.allclose(D.Y, X[:, :, 2].T)
    D1 = Data(X[0], dt=0.5)                      # a single trajectory (T+1, d)
    assert D1.N == 1
    D2 = Data(X, dt=0.5, observable=lambda Z: Z[..., 0] + Z[..., 1])
    assert np.allclose(D2.Y, (X[..., 0] + X[..., 1]).T)


def test_data_errors():
    X = np.zeros((5, 4, 2)); X[0, 0, 0] = np.nan
    with pytest.raises(ValueError): Data(X, dt=1)
    X = np.zeros((5, 4, 2))
    with pytest.raises(ValueError): Data(X, dt=1, obs=np.ones((5, 8)))   # obs inconsistent with the states
    with pytest.raises(ValueError): Data(X, dt=1, obs=np.zeros((5, 2)))  # obs too short


def test_fit_requires_data():
    with pytest.raises(TypeError): KMZ().fit(np.zeros((100, 10, 2)))


def test_predict_shapes(model, data):
    p = model.predict([0.5, 0.0])
    assert isinstance(p, Prediction) and p.value.shape == (151,) and p.error.shape == (151,)
    assert abs(p.value[0] - 0.5) < 1e-8                 # ψ is in the dictionary: exact at t = 0
    assert np.all(np.isfinite(p.error)) and np.all(p.error >= 0)
    P = model.predict(data.x0[:7], horizon=5.0)
    assert P.value.shape == (51, 7) and P.error.shape == (51, 7)
    low, high = P.interval(2)
    assert np.all(low <= high)


def test_predict_beyond_data(model):
    with pytest.warns(KMZWarning):
        p = model.predict([0.5, 0.0], horizon=25.0)
    assert np.isnan(p.error[-1]) and np.isfinite(p.value[-1])
    with pytest.raises(ValueError):
        model.predict([0.5, 0.0], horizon=25.0, method="projection")


def test_out_of_domain(model):
    with pytest.warns(KMZWarning):
        p = model.predict([8.0, 8.0])
    assert p.out_of_domain


def test_projection_reaches_the_floor(model):
    # the direct projection is the floor: on average, it does not do worse than the model with memory
    assert np.mean(model.proj_error) <= np.mean(model.val_error) * 1.02


def test_dictionary_contains_the_basin(model):
    assert any("basin" in b for b in model.block_names)
    assert model.diag["convergence"] == "strong"


def test_evaluate_dictionary(model):
    E = lambda X: (0.5 * X[..., 1] ** 2 - 0.5 * X[..., 0] ** 2 + 0.25 * X[..., 0] ** 4)[..., None]
    ev = model.evaluate_dictionary(extra=[("energy", E)])
    assert "energy" in ev.trials and len(ev.recommendations) >= 1
    assert np.allclose(ev.model_error ** 2, np.maximum(ev.floor ** 2, ev.model_error ** 2 - ev.memory_part ** 2))
    assert "Advice" in ev.summary()


def test_user_candidates(data):
    # a useful function (indicator of the right basin, controlled "cheat") must be kept
    lim = np.sign(data.states[:, -1, 0])
    from kmz.dictionary import Coordinates, fit_rff
    co = Coordinates(data.x0); f, _ = fit_rff(co(data.x0), lim, np.random.default_rng(0), co)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", KMZWarning)
        m = KMZ(dictionary="base", candidates=[("my_function", lambda X: f(X)[..., None])]).fit(data)
    assert "my_function" in m.block_names


def test_certificate_no_violation():
    D = simulate("damped_duffing", N=1500, duration=3.0, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", KMZWarning)
        m = KMZ(dictionary="base").fit(D)
    c = m.certificate(BENCHMARK["damped_duffing"].F, times=(0.2, 0.5, 1.0, 2.0), m=1500)
    for row in c["rows"]:
        assert row["cert_F"] >= row["err"]
    assert c["useful_horizon"] > 0


def test_basins_duffing(data):
    b = find_basins(data)
    assert b.n_basins == 2 and b.accuracy > 0.95
    assert isinstance(b.classify([1.5, 0.0]), int)
    assert np.sign(b.attractors[b.classify([1.5, 0.0]), 0]) == 1
    c = b.map(n=40)
    assert c["classes"].shape == (40, 40) and set(np.unique(c["classes"])) == {0, 1}


def test_basins_monostable_system():
    D = simulate("van_der_pol", N=1500, duration=30.0, seed=0, subsample=5)
    assert find_basins(D).n_basins == 1


def test_analyze_duffing(data, model, tmp_path):
    r = analyze(data, model=model)
    titles = [t for t, _ in r.sections]
    assert titles == ["The data", "The long-time behaviour", "The Koopman modes", "The memory", "The predictability"]
    assert r.basins.n_basins == 2 and len(r.findings) >= 3
    assert any("interpretation" in l for l in r.limitations)    # the tool reminds that it does not interpret
    path = r.write(str(tmp_path / "analysis"))
    text = open(path).read()
    assert "## Summary" in text and (tmp_path / "analysis" / "basins.png").exists()


def test_analyze_finds_the_van_der_pol_cycle():
    """The Koopman mode of the limit cycle (period ≈ 6.66, no damping) must be found and judged reliable."""
    D = simulate("van_der_pol", N=3000, duration=30.0, seed=0)
    r = analyze(D)
    reliable = [m for m in r.modes if m["reliable"] and m["period"]]
    assert any(abs(m["period"] - 6.66) < 0.3 and abs(m["rate"]) < 0.05 for m in reliable)
