"""Walk-forward backtest: refit on an expanding window, predict the next block."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .evaluate import summarise
from .features import FEATURES


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict
    folds: list[dict] = field(default_factory=list)


def _predict_block(model, X_all: np.ndarray, s: int, e: int) -> np.ndarray:
    """Predict rows [s, e). Sequence models need `lookback - 1` rows of context."""
    if not model.is_sequence:
        return model.predict(X_all[s:e])
    ctx = max(0, s - model.lookback + 1)
    preds = model.predict(X_all[ctx:e])
    if len(preds) != e - s:
        raise RuntimeError(f"{model.name}: expected {e - s} predictions, got {len(preds)}")
    return preds


def run_backtest(frame: pd.DataFrame, model_classes: dict, test_start: str = config.TEST_START,
                 refit_every: int = config.REFIT_EVERY) -> BacktestResult:
    """Rolling-origin evaluation with strictly chronological, non-overlapping blocks.

    Row i holds features known at the close of day i and the label for day i+1.
    Training for a block ending at `s` uses rows [0, s) only, so no test-period
    information reaches the model or its scaler.
    """
    X_all = frame[FEATURES].to_numpy(dtype=np.float64)
    y_all = frame["target"].to_numpy(dtype=np.float64)
    n = len(frame)

    start = int(np.searchsorted(frame.index.to_numpy(), np.datetime64(test_start)))
    if start <= 200:
        raise ValueError(f"test_start {test_start} leaves too few training rows")

    preds = {name: np.full(n - start, np.nan) for name in model_classes}
    folds = []
    for s in range(start, n, refit_every):
        e = min(s + refit_every, n)
        X_train, y_train = X_all[:s], y_all[:s]
        for name, cls in model_classes.items():
            model = cls().fit(X_train, y_train)
            preds[name][s - start:e - start] = _predict_block(model, X_all, s, e)
        folds.append({"start": str(frame.index[s].date()),
                      "end": str(frame.index[e - 1].date()), "rows": e - s})

    test = frame.iloc[start:]
    out = pd.DataFrame({"close": test["close"].to_numpy(),
                        "actual": test["target"].to_numpy()}, index=test.index)
    for name in model_classes:
        out[name] = preds[name]

    actual = out["actual"].to_numpy()
    metrics = summarise(actual, {k: out[k].to_numpy() for k in model_classes})
    metrics["_meta"] = {
        "test_start": str(out.index[0].date()),
        "test_end": str(out.index[-1].date()),
        "test_rows": int(len(out)),
        "train_rows_at_first_fold": int(start),
        "lookback": config.LOOKBACK,
        "refit_every": refit_every,
        "folds": folds,
    }
    return BacktestResult(out, metrics, folds)
