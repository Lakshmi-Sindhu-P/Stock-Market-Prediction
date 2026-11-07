"""Data acquisition and validation.

Source: Yahoo Finance public chart endpoint for ^GSPC (S&P 500 index, daily).
The index needs no dividend/split adjustment, so `close` is used directly —
this sidesteps the adjusted-vs-unadjusted trap in the original notebook.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import RAW_CSV, RAW_META, TICKER

YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
)

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DataReport:
    rows: int
    start: str
    end: str
    missing_volume: int
    max_gap_days: int


def fetch_daily(ticker: str = TICKER, start: str = "1970-01-01",
                retries: int = 3) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo's chart endpoint."""
    p1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.now(timezone.utc).timestamp())
    url = YAHOO_URL.format(ticker=urllib.parse.quote(ticker), p1=p1, p2=p2)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
            result = payload["chart"]["result"][0]
            quote = result["indicators"]["quote"][0]
            df = pd.DataFrame({
                "date": pd.to_datetime(result["timestamp"], unit="s").date,
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"],
            })
            return df[COLUMNS]
        except Exception as exc:  # network hiccup or rate limit
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {ticker}: {last_err}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and enforce dtypes/ordering."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    out["volume"] = out["volume"].fillna(0).astype("int64")
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype("float64")
    out = out.drop_duplicates(subset="date", keep="last")
    return out.sort_values("date").reset_index(drop=True)


def validate(df: pd.DataFrame) -> DataReport:
    """Fail loudly on the problems that silently corrupt a backtest."""
    if df.empty:
        raise ValueError("empty dataset")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("dates are not sorted ascending")
    if df["date"].duplicated().any():
        raise ValueError("duplicate dates")
    if (df["close"] <= 0).any():
        raise ValueError("non-positive close price")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("NaN in price columns")
    # High/low sanity: a close outside [low, high] means the row is corrupt.
    bad = (df["close"] > df["high"] * 1.001) | (df["close"] < df["low"] * 0.999)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have close outside [low, high]")

    gaps = df["date"].diff().dt.days
    return DataReport(
        rows=len(df),
        start=str(df["date"].iloc[0].date()),
        end=str(df["date"].iloc[-1].date()),
        missing_volume=int((df["volume"] == 0).sum()),
        max_gap_days=int(np.nanmax(gaps.fillna(0))),
    )


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} missing — run scripts/fetch_data.py")
    return pd.read_csv(RAW_CSV, parse_dates=["date"])


def save_raw(df: pd.DataFrame, report: DataReport) -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    RAW_META.write_text(json.dumps({
        "source": "Yahoo Finance chart API",
        "ticker": TICKER,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": report.rows,
        "start": report.start,
        "end": report.end,
        "missing_volume_rows": report.missing_volume,
        "max_gap_calendar_days": report.max_gap_days,
    }, indent=2) + "\n")


def log_returns(df: pd.DataFrame) -> pd.Series:
    """Daily log returns, indexed by date. First observation is dropped."""
    ret = np.log(df.set_index("date")["close"]).diff().dropna()
    ret.name = "log_ret"
    return ret
