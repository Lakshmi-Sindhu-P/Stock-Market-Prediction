"""Streamlit dashboard for the walk-forward results.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from spx_forecast import config  # noqa: E402
from spx_forecast.features import FEATURES  # noqa: E402
from spx_forecast.plots import equity  # noqa: E402

st.set_page_config(page_title="S&P 500 Forecast — walk-forward", layout="wide")


@st.cache_data(show_spinner=False)
def load() -> tuple[pd.DataFrame, dict]:
    if not config.PREDICTIONS_CSV.exists():
        st.error("No results yet. Run `python scripts/run_backtest.py` first.")
        st.stop()
    preds = pd.read_csv(config.PREDICTIONS_CSV, parse_dates=["date"], index_col="date")
    metrics = json.loads(config.METRICS_JSON.read_text())
    return preds, metrics


preds, metrics = load()
models = [m for m in ("persistence", "drift", "ridge", "lstm") if m in preds.columns]
meta = metrics["_meta"]

st.title("S&P 500 — next-day return forecasts")
st.caption(
    f"Walk-forward test {meta['test_start']} → {meta['test_end']} · "
    f"{meta['test_rows']:,} trading days · {len(meta['folds'])} folds · "
    f"refit every {meta['refit_every']} days"
)

st.info(
    "Daily index returns are close to unpredictable. The honest benchmark is the "
    "naive forecast (tomorrow = today): anything that fails to beat it has no edge, "
    "however good the price chart looks.",
    icon="ℹ️",
)

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    model = st.selectbox("Model", models, index=len(models) - 1)
    window = st.select_slider(
        "Chart window (trading days)",
        options=[60, 120, 250, 500, len(preds)],
        value=250,
    )
    st.divider()
    st.caption("Metrics are computed over the full out-of-sample window.")

# --- metric cards ----------------------------------------------------------
m = metrics[model]
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{m['mae_bp']:.2f} bp", help="Mean absolute error, next-day log return")
mase_delta = m["mase"] - 1
c2.metric("MASE", f"{m['mase']:.4f}", delta=f"{mase_delta:+.2%} vs naive",
          delta_color="inverse", help="< 1 beats the naive forecast")
c3.metric("Hit rate", "—" if m["direction_acc"] is None else f"{m['direction_acc']:.1%}")
dm = m.get("dm_vs_naive") or {}
c4.metric("Diebold–Mariano p", "—" if not dm else f"{dm['p_value']:.3f}",
          help="p < 0.05 means a statistically real difference from the naive forecast")

# --- charts ----------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Predicted vs realised return")
    tail = preds.tail(window)
    st.line_chart(
        pd.DataFrame({
            "realised": tail["actual"] * 1e4,
            "predicted": tail[model] * 1e4,
        }),
        y_label="basis points",
        height=320,
    )
    st.caption("Predicted returns are deliberately small — the model regresses to the mean.")

with right:
    st.subheader("Signal equity vs buy & hold")
    eq = pd.DataFrame({
        "buy & hold": np.exp(np.cumsum(preds["actual"])),
        f"{model} signal": equity(preds[model].to_numpy(), preds["actual"].to_numpy()),
    })
    st.line_chart(eq, height=320, y_label="growth of $1 (log)")
    st.caption("Long when the forecast is positive, flat otherwise; 1 bp per switch.")

# --- table -----------------------------------------------------------------
st.subheader("All models")
rows = []
for name in models:
    d = metrics[name]
    dm = d.get("dm_vs_naive") or {}
    rows.append({
        "model": name,
        "RMSE (bp)": round(d["rmse_bp"], 2),
        "MAE (bp)": round(d["mae_bp"], 2),
        "MASE": round(d["mase"], 4),
        "R²": round(d["r2"], 4),
        "hit rate": None if d["direction_acc"] is None else round(d["direction_acc"], 3),
        "DM p": None if not dm else round(dm["p_value"], 3),
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with st.expander("How the backtest works"):
    st.markdown(
        f"""
- **Target**: next-day log return of the S&P 500 close.
- **Features** ({len(FEATURES)}): trailing returns, realised volatility, RSI, intraday range,
  volume ratio, distance from the 20-day mean, overnight gap — all computed from
  data available at the close.
- **Split**: strictly chronological. The model is refit every
  {meta['refit_every']} trading days on every observation before the block and
  predicts the block without peeking. Scalers are fit inside each fold.
- **LSTM**: {meta['lookback']}-day sequence of features → 32-unit LSTM → dropout → 1 output,
  early-stopped on a chronological validation tail.
- **Costs**: 1 bp per position change in the equity chart.
        """
    )
