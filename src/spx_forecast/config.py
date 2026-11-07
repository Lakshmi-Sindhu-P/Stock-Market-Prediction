"""Project-wide constants and paths. Keep this small — no config framework."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TICKER = "^GSPC"
RAW_CSV = ROOT / "data" / "raw" / "spx_daily.csv"
RAW_META = ROOT / "data" / "raw" / "spx_daily.meta.json"
PREDICTIONS_CSV = ROOT / "results" / "predictions.csv"
METRICS_JSON = ROOT / "results" / "metrics.json"
FIGURES = ROOT / "results" / "figures"

SEED = 7

# Walk-forward setup. 2015+ is out-of-sample; models refit once a year.
TEST_START = "2015-01-01"
REFIT_EVERY = 252  # trading days

# Feature / model hyper-parameters (deliberately small: ~14k daily rows)
LOOKBACK = 20          # sequence length fed to the LSTM
HIDDEN_UNITS = 32
DROPOUT = 0.2
MAX_EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 128
