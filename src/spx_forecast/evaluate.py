"""Error metrics and tests that keep the comparison honest."""

from __future__ import annotations

import math

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def r2(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 against the mean of the *actual* series."""
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1 - ss_res / ss_tot)


def mase(actual: np.ndarray, pred: np.ndarray, naive: np.ndarray) -> float:
    """Mean absolute scaled error against the persistence forecast.

    < 1 means the model beats "tomorrow = today"; > 1 means it is worse.
    """
    denom = np.mean(np.abs(actual - naive))
    return float(np.mean(np.abs(actual - pred)) / denom) if denom else float("nan")


def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    """Share of days where the sign of the predicted return is right.

    Days the model calls 'flat' (exactly zero) are counted as wrong; there are
    only a handful and it keeps the metric conservative.
    """
    return float(np.mean(np.sign(actual) == np.sign(pred)))


def sharpe(returns: np.ndarray, periods: int = 252) -> float:
    sd = np.std(returns, ddof=1)
    return float(np.mean(returns) / sd * np.sqrt(periods)) if sd else float("nan")


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def cagr(equity: np.ndarray, periods_per_year: int = 252) -> float:
    years = len(equity) / periods_per_year
    return float(equity[-1] ** (1 / years) - 1) if years > 0 else float("nan")


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test: is forecaster A's loss different from B's?

    Returns (statistic, two-sided p-value). Uses squared-error loss with a
    Newey-West correction for h-step overlapping forecasts.
    """
    d = err_a ** 2 - err_b ** 2
    T = len(d)
    d_bar = d.mean()
    # Newey-West long-run variance
    gamma0 = np.sum((d - d_bar) ** 2) / T
    lrv = gamma0
    for lag in range(1, h):
        gamma = np.sum((d[lag:] - d_bar) * (d[:-lag] - d_bar)) / T
        lrv += 2 * (1 - lag / h) * gamma
    if lrv <= 0:
        # Identical forecasts: differences are exactly zero, so there is no
        # evidence of a difference. Report the null result rather than NaN.
        return (0.0, 1.0) if np.allclose(d, 0.0) else (float("nan"), float("nan"))
    stat = d_bar / np.sqrt(lrv / T)
    # Normal approximation is standard for DM at these sample sizes.
    p = float(2 * (1 - 0.5 * (1 + math.erf(abs(stat) / math.sqrt(2)))))
    return float(stat), p


def summarise(actual: np.ndarray, preds: dict[str, np.ndarray],
              naive: np.ndarray | None = None) -> dict:
    """All metrics for a dict of {model_name: predictions}, in return space."""
    naive = np.zeros_like(actual) if naive is None else naive
    out = {}
    for name, pred in preds.items():
        out[name] = {
            "rmse_bp": rmse(actual, pred) * 1e4,
            "mae_bp": mae(actual, pred) * 1e4,
            "r2": r2(actual, pred),
            "mase": mase(actual, pred, naive),
            "direction_acc": directional_accuracy(actual, pred),
            "n": int(len(actual)),
        }
        if name == "persistence":
            out[name]["direction_acc"] = None  # flat forecast has no direction
        else:
            stat, p = diebold_mariano(actual - pred, actual - naive)
            out[name]["dm_vs_naive"] = {"stat": stat, "p_value": p}
    return out
