import numpy as np
import pytest

from spx_forecast import evaluate as ev


def test_metrics_on_toy_data():
    actual = np.array([0.01, -0.02, 0.03, 0.0])
    pred = np.array([0.01, -0.01, 0.0, 0.0])
    assert ev.rmse(actual, actual) == 0.0
    assert ev.mae(actual, actual) == 0.0
    assert ev.rmse(actual, pred) == pytest.approx(np.sqrt((0 + 1e-4 + 9e-4 + 0) / 4))
    assert ev.directional_accuracy(actual, pred) == pytest.approx(0.75)


def test_mase_identity():
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 0.01, 500)
    naive = np.zeros_like(actual)
    assert ev.mase(actual, naive, naive) == pytest.approx(1.0)
    # A forecast twice as bad as persistence scores 2.
    worse = 2 * np.abs(actual)
    assert ev.mase(actual, worse, naive) == pytest.approx(2.0, rel=0.2)


def test_diebold_mariano_identical_is_zero():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 0.01, 400)
    stat, p = ev.diebold_mariano(e, e)
    assert stat == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_diebold_mariano_detects_worse_forecast():
    rng = np.random.default_rng(3)
    actual = rng.normal(0, 0.01, 2000)
    good = np.zeros_like(actual)
    bad = rng.normal(0, 0.05, 2000)
    stat, p = ev.diebold_mariano(actual - bad, actual - good)
    assert stat > 1.0 and p < 0.05


def test_max_drawdown():
    eq = np.array([1.0, 1.2, 0.9, 1.1])
    assert ev.max_drawdown(eq) == pytest.approx(0.9 / 1.2 - 1)
