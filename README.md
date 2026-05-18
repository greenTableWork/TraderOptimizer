# TraderOptimizer

TraderOptimizer is a small, verbose Optuna training loop for producing
TraderCore-style strategy config JSON.

The first supported strategy is `ConstantStepOffset`, because the current Trader
workspace already has local SQLite bars and CSO config files. The optimizer reads
`TraderLab/Data/tws_historical.sqlite`, tries CSO hyperparameters with Optuna,
runs a simple local simulator, and writes:

- `best_config.json`: a TraderCore-compatible strategy config.
- `best_summary.json`: the best trial, metrics, and data window.
- `trials.csv`: every Optuna trial and its metrics.
- `optuna-study.db`: the Optuna SQLite study.

This is intentionally simple. Use it to search parameter ranges quickly, then
validate promising configs with the real TraderCore `BackTester`.

## Setup

From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Run a quick optimization

```bash
trader-optimizer optimize \
  --trader-root .. \
  --symbol AAPL \
  --bar-size "10 secs" \
  --trials 50 \
  --max-bars 50000
```

The command is verbose by default. It prints the data source, bar window, Optuna
study path, best score, best config path, and train/validation metrics.

Use `--max-bars 0` if you want to run against the full matching SQLite series.
That can be much slower for the two-year `10 secs` scrape.

## Output config

The generated config uses the same core fields as TraderCore CSO configs:

```json
{
  "strategy_type": "ConstantStepOffset",
  "baseline": 276.85,
  "stepDelta": 1.0,
  "executionLimitOffset": 40.0,
  "stateTransitionThreshold": 0.25,
  "orderQuantityInUSD": 100.0,
  "orderQuantity": 1,
  "contract": {
    "symbol": "AAPL",
    "secType": "STOCK",
    "currency": "USD",
    "exchange": "BACKTESTER"
  },
  "price_contract": {
    "symbol": "AAPL",
    "secType": "STOCK",
    "currency": "USD",
    "exchange": "BACKTESTER"
  },
  "ledgerPath": "data/TraderLedger/CSO_AAPL_OPTIMIZED.sqlite",
  "ledgerContextCollection": "CSO_AAPL_OPTIMIZED_context"
}
```

## Validate with TraderCore BackTester

After optimization, run the generated config through the real BackTester:

```bash
cd ../TraderLab
scripts/run_tradercore_backtest.sh --skip-build -- \
  --sqlite-data Data/tws_historical.sqlite \
  --strategy-config /absolute/path/to/TraderOptimizer/runs/.../best_config.json \
  --bar-size "10 secs" \
  --what-to-show TRADES \
  --use-rth 1
```

The local optimizer is fast and inspectable, but the C++ BackTester remains the
source of truth for fills, ledger writing, and runtime behavior.

## Optimize existing strategy configs

To discover the checked-in backtesting and stock-stress configs and generate an
optimized config for each one:

```bash
trader-optimizer optimize-existing \
  --trader-root .. \
  --trials 25 \
  --max-bars 5000 \
  --output-dir runs/batch_existing
```

The batch command writes one folder per strategy plus:

- `runs/batch_existing/batch_summary.json`
- `runs/batch_existing/batch_summary.csv`

The current discovery path covers:

- `TraderCore/configs/backtesting/**/*.json`
- `TraderLab/configs/backtests/ibkr_stock_stress/*.json`

It supports `ConstantStepOffset`, `MovingAverageCross`, `TechnicalSignal`, and
`PortfolioAllocation` configs with local SQLite bars. Missing data or unsupported
configs are recorded as skipped in the batch summary.

## Notebook report

The executed notebook report is:

```text
notebooks/TraderOptimizer_Batch_Results.ipynb
```

It loads `runs/batch_existing`, ranks generated configs by objective and return,
shows strategy-family coverage, renders a compact return chart, and prints the
best generated config JSON.

## Hyperparameters

Optuna currently tunes:

- `baseline_quantile`: converts the training close-price distribution into a
  baseline.
- `step_delta_pct`: converts a percent of baseline into `stepDelta`.
- `execution_steps`: controls `executionLimitOffset`.
- `threshold_pct_of_step`: controls `stateTransitionThreshold`.
- `order_quantity_usd`: controls `orderQuantityInUSD`.

The objective is a blended train/validation return score with penalties for
configs that do not trade or that finish with too much marked open inventory.
That prevents the study from picking a config only because it holds a large
unclosed position at the final bar.
