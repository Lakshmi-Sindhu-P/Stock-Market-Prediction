import numpy as np
import pandas as pd
import pytest

from spx_forecast.data import clean, log_returns, validate


def _frame(**over):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.0, 101.0, 102.0],
        "volume": [1_000, 1_100, 1_200],
    })
    for k, v in over.items():
        df[k] = v
    return df


def test_validate_ok():
    rep = validate(_frame())
    assert rep.rows == 3
    assert rep.start == "2020-01-02"


def test_validate_rejects_duplicate_dates():
    df = _frame()
    df.loc[1, "date"] = df.loc[0, "date"]
    with pytest.raises(ValueError, match="duplicate"):
        validate(df)


def test_validate_rejects_close_outside_range():
    df = _frame()
    df.loc[0, "close"] = 500.0
    with pytest.raises(ValueError, match="outside"):
        validate(df)


def test_clean_drops_bad_rows_and_sorts():
    df = _frame()
    df.loc[2, "close"] = -1.0
    out = clean(df)
    assert len(out) == 2
    assert out["date"].is_monotonic_increasing


def test_log_returns():
    r = log_returns(_frame())
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(np.log(101 / 100))
    assert r.iloc[1] == pytest.approx(np.log(102 / 101))
