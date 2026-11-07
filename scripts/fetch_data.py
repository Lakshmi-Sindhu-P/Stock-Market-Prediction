#!/usr/bin/env python
"""Download the S&P 500 daily series and validate it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spx_forecast.data import clean, fetch_daily, save_raw, validate  # noqa: E402


def main() -> int:
    raw = fetch_daily()
    df = clean(raw)
    report = validate(df)
    save_raw(df, report)
    print(f"saved {report.rows} rows  {report.start} -> {report.end}")
    print(f"zero-volume rows: {report.missing_volume}   max calendar gap: {report.max_gap_days}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
