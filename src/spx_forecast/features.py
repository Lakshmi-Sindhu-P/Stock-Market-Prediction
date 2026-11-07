"""Causal feature engineering.

Every feature in a row dated t is computed from information available at the
close of t. The label for row t is the return from t to t+1, so a row is
"predictable" only with data a trader could actually have had.

`tests/test_features.py::test_no_lookahead` enforces this by mutating the
future and asserting the past does not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1",      # yesterday's log return
    "ret_5",
    "ret_20",
    "vol_5",      # realised volatility, 5d
    "vol_20",
    "rsi_14",     # Wilder RSI, scaled to [-1, 1]
    "range_pct",  # (high - low) / close, intraday range
    "vol_ratio",  # volume vs its 20d mean
    "ma_ratio",   # close vs 20d SMA (mean-reversion / trend signal)
    "gap",        # overnight gap: open / prev close - 1
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame indexed by date, including the next-day label.

    Columns: FEATURES + 'target' (next-day log return) + 'close' (for price
    reconstruction). Rows where any input is undefined (warm-up) are dropped.
    """
    out = df.copy().set_index("date").sort_index()
    close = out["close"]
    log_close = np.log(close)
    ret = log_close.diff()

    feats = pd.DataFrame(index=out.index)
    feats["ret_1"] = ret
    feats["ret_5"] = log_close.diff(5)
    feats["ret_20"] = log_close.diff(20)
    feats["vol_5"] = ret.rolling(5).std()
    feats["vol_20"] = ret.rolling(20).std()
    feats["rsi_14"] = (_rsi(close, 14) - 50.0) / 50.0
    feats["range_pct"] = (out["high"] - out["low"]) / close
    feats["vol_ratio"] = out["volume"] / out["volume"].rolling(20).mean()
    feats["ma_ratio"] = close / close.rolling(20).mean() - 1.0
    feats["gap"] = out["open"] / close.shift(1) - 1.0

    # Label: the return realised on the *next* bar.
    feats["target"] = ret.shift(-1)
    feats["close"] = close

    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows: window ending at row i predicts label y[i]."""
    n = len(X) - lookback + 1
    if n <= 0:
        raise ValueError(f"not enough rows ({len(X)}) for lookback={lookback}")
    windows = np.lib.stride_tricks.sliding_window_view(X, (lookback, X.shape[1]))
    seqs = windows[:, 0, :, :].copy()
    return seqs, y[lookback - 1:]


def historical_vol(ret: pd.Series, window: int = 20) -> float:
    """Annualised volatility, used for reporting only."""
    return float(ret.rolling(window).std().iloc[-1] * np.sqrt(252))
