"""The leakage guard: future data must not change past features."""

import numpy as np
import pandas as pd
import pytest

from spx_forecast.features import FEATURES, build_features, make_sequences


@pytest.fixture()
def frame():
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.bdate_range("2010-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": dates,
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1e6, 5e6, n),
    })


def test_no_lookahead(frame):
    """Perturbing data after `cut` must not move any feature on or before it."""
    feats = build_features(frame)
    cut = feats.index[200]

    future = frame.copy()
    mask = future["date"] > cut
    for col in ("open", "high", "low", "close"):
        future.loc[mask, col] *= 1.5
    perturbed = build_features(future)

    a = feats.loc[:cut, FEATURES]
    b = perturbed.loc[:cut, FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_label_is_next_day_return(frame):
    feats = build_features(frame)
    close = frame.set_index("date")["close"]
    # build_features drops warm-up rows, so map by date rather than position.
    when = feats.index[250]
    i = close.index.get_loc(when)
    expected = np.log(close.iloc[i + 1] / close.iloc[i])
    assert feats.loc[when, "target"] == pytest.approx(expected, rel=1e-9)


def test_no_nans_and_ordering(frame):
    feats = build_features(frame)
    assert not feats.isna().any().any()
    assert feats.index.is_monotonic_increasing


def test_sequences_align_with_last_row(frame):
    feats = build_features(frame)
    X = feats[FEATURES].to_numpy()[:50]
    y = feats["target"].to_numpy()[:50]
    seqs, labels = make_sequences(X, y, lookback=10)
    assert seqs.shape == (41, 10, len(FEATURES))
    assert np.allclose(seqs[0, -1], X[9])          # window ends at row 9
    assert labels[0] == pytest.approx(y[9])
    assert len(seqs) == len(labels)
