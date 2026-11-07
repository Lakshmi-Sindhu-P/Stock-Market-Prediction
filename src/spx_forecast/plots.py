"""Result figures (matplotlib only, saved as PNG for the README)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

COST_BP = 1.0  # one basis point per position change


def _strategy_returns(pred: np.ndarray, actual: np.ndarray, cost_bp: float = COST_BP) -> np.ndarray:
    """Long when the forecast is positive, flat otherwise, minus turnover cost."""
    pos = (pred > 0).astype(float)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    return pos * actual - turnover * cost_bp / 1e4


def equity(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.exp(np.cumsum(_strategy_returns(pred, actual)))


def plot_all(preds: pd.DataFrame, metrics: dict, out_dir=config.FIGURES) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    models = [c for c in preds.columns if c not in ("close", "actual")]
    actual = preds["actual"].to_numpy()

    # 1. Equity curves -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    bh = np.exp(np.cumsum(actual))
    ax.plot(preds.index, bh, label="Buy & hold", lw=1.6, color="black")
    colors = {"persistence": "tab:gray", "drift": "tab:orange",
              "ridge": "tab:green", "lstm": "tab:blue"}
    for m in models:
        if m == "persistence":
            continue
        ax.plot(preds.index, equity(preds[m].to_numpy(), actual),
                label=f"{m} signal", lw=1.2, color=colors.get(m))
    ax.set_yscale("log")
    ax.set_title("Out-of-sample growth of $1 (1 bp cost per position change)")
    ax.set_ylabel("Equity (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "equity_curves.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    # 2. Rolling RMSE --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    for m in ("persistence", "lstm"):
        if m in preds:
            err = (preds[m] - preds["actual"]) ** 2
            ax.plot(preds.index, np.sqrt(err.rolling(120).mean()) * 1e4, label=m, lw=1.1)
    ax.set_title("Rolling 120-day RMSE (basis points)")
    ax.set_ylabel("bp")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "rolling_rmse.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    # 3. Prediction quality --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if "lstm" in preds:
        axes[0].hexbin(preds["lstm"] * 1e4, preds["actual"] * 1e4, gridsize=45,
                       cmap="Blues", mincnt=1)
        axes[0].set_xlabel("predicted next-day return (bp)")
        axes[0].set_ylabel("realised next-day return (bp)")
        r = np.corrcoef(preds["lstm"], preds["actual"])[0, 1]
        axes[0].set_title(f"LSTM: predicted vs realised (corr {r:+.3f})")
    axes[1].hist((preds["actual"] - preds["lstm"]) * 1e4, bins=80, alpha=0.75)
    axes[1].set_title("LSTM error distribution (bp)")
    axes[1].set_xlabel("error (bp)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "prediction_quality.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))
    return written
