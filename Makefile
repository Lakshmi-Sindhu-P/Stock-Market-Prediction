.PHONY: setup data backtest test app clean

VENV ?= .venv
PY := $(VENV)/bin/python

setup:            ## create the environment and install the package
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PY) -e ".[dev]"

data:             ## refresh the S&P 500 series from Yahoo Finance
	$(PY) scripts/fetch_data.py

backtest:         ## full walk-forward run (writes results/ and reports/RESULTS.md)
	$(PY) scripts/run_backtest.py

test:             ## leakage + metric tests
	$(PY) -m pytest

app:              ## dashboard on http://localhost:8501
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -rf results/figures/*.png
