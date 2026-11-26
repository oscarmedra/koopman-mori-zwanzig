"""Tests on the benchmark of the 10 systems (6 ODEs, 4 PDEs). Duration: ~12-15 min on one core.
Faster (less precise): KMZ_SCALE=0.3 pytest tests/test_benchmark.py

Each test checks a claim of the BILAN (recherche/outil/BILAN.md). Known limitations are tested as such:
we check that the tool reports them, not that it overcomes them."""
import pytest
from bench import evaluate, CONFIG, BISTABLE, CHAOS

SYSTEMS = list(CONFIG)
LEARNED = ("basin", "cesaro", "harmonic", "limit")


@pytest.mark.parametrize("name", SYSTEMS)
def test_announced_error_predicts_test_error(name):
    """The validation error (global bar) predicts the error on new trajectories within 20 %."""
    r = evaluate(name); K = r["K"]
    assert abs(r["err_val"][K] - r["err_test"][K]) <= 0.2 * r["err_test"][K]


@pytest.mark.parametrize("name", SYSTEMS)
def test_local_bar_coverage(name):
    """At least 90 % of the test errors fall within ± 2 local error bars."""
    assert evaluate(name)["test_coverage_2s"] >= 0.90


@pytest.mark.parametrize("name", SYSTEMS)
def test_floor_reached_on_the_window(name):
    """Over the state window, the model (full memory) reaches the floor (Thm H(iii))."""
    r = evaluate(name); H = r["H_mem"]
    assert r["err_test"][H] <= 1.10 * r["floor"][H] + 0.01 * r["std"][H]


@pytest.mark.parametrize("name", [s for s in SYSTEMS if s != "nagumo"])
def test_not_worse_than_a_constant(name):
    """At the end of the horizon, the model does at least as well as a constant prediction (within 5 %)."""
    r = evaluate(name); K = r["K"]
    assert r["err_test"][K] <= 1.05 * r["std"][K]


def test_nagumo_known_limitation_reported():
    """Known limitation: for Nagumo, no memory truncation is stable, and the error ends worse than a
    constant. The tool must report it (warning) and the announced error must show it."""
    r = evaluate("nagumo"); K = r["K"]
    assert not r["memory_stable"]
    assert any("stable" in a for a in r["fit_warnings"])
    if r["err_test"][K] > r["std"][K]:
        assert r["err_val"][K] > r["std"][K]     # the announced error warns the user


@pytest.mark.parametrize("name", ["damped_duffing", "nagumo"])
def test_basin_function_kept(name):
    assert any("basin" in b for b in evaluate(name)["blocks"])


@pytest.mark.parametrize("name", sorted(CHAOS))
def test_chaos_no_learned_function(name):
    """For chaos, the invariant function is constant: the tool must not learn anything useless."""
    assert not any(b.startswith(LEARNED) for b in evaluate(name)["blocks"])


def test_van_der_pol_finds_the_phase():
    assert any("harmonic" in b for b in evaluate("van_der_pol")["blocks"])
    assert abs(evaluate("van_der_pol")["model"].diag["periodic_component"]["period"] - 6.66) < 0.3


@pytest.mark.parametrize("name", SYSTEMS)
def test_basins(name):
    """3 bistable systems: 2 basins, ≥ 90 % correctly classified. The 7 others: a single basin."""
    b = evaluate(name)["basins"]
    if name in BISTABLE:
        assert b.n_basins == 2 and b.accuracy >= 0.90
    else:
        assert b.n_basins == 1
