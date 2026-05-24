# TraderOptimizer

TraderOptimizer is a small, verbose Optuna training loop for producing
TraderCore-style strategy config JSON.

The optimizer reads `public.historical_bars` from PostgreSQL, tries strategy
hyperparameters with Optuna, runs a simple local simulator, and writes:

- `best_config.json`: a TraderCore-compatible strategy config.
- `best_summary.json`: the best trial, metrics, and data window.
- `optimizer_runs`, `optimizer_trials`, and `optimizer_fills` rows in PostgreSQL.
- `optimizer_batch_results` rows for batch optimization summaries.
- Optuna study tables in PostgreSQL.
- JSON config and summary artifacts for review next to each run.

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

Use `--max-bars 0` if you want to run against the full matching PostgreSQL series.
That can be much slower for the two-year `10 secs` scrape.
Use `--pg-host`, `--pg-port`, `--pg-database`, `--pg-user`, and
`--pg-password` to override the default local `trader` database. You can also
pass `--pg-conninfo`; when doing that, pass `--optuna-storage-url` too because
Optuna needs a SQLAlchemy PostgreSQL URL.

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
  "ledgerPath": "data/TraderLedger/CSO_AAPL_OPTIMIZED",
  "ledgerContextCollection": "CSO_AAPL_OPTIMIZED_context"
}
```

## Validate with TraderCore BackTester

After optimization, run the generated config through the real BackTester:

```bash
cd ../TraderLab
scripts/run_tradercore_backtest.sh --skip-build -- \
  --pg-database trader \
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
  --workers 4 \
  --output-dir runs/batch_existing \
  --plan-path reports/batch_optimization_plan.md \
  --export-config-dir optimized_configs/batch_existing
```

The batch command writes one folder per strategy plus:

- `runs/batch_existing/batch_summary.json`
- `optimizer_runs`, `optimizer_trials`, and `optimizer_batch_results` rows in PostgreSQL

Candidate strategies run concurrently by default, up to 4 workers. Pass
`--workers 1` for serial execution or a larger value when PostgreSQL and the
local machine can absorb more parallel studies.

The current discovery path covers:

- `TraderCore/configs/backtesting/**/*.json`
- `TraderLab/configs/backtests/ibkr_stock_stress/*.json`

It supports `ConstantStepOffset`, `MovingAverageCross`, `TechnicalSignal`, and
`PortfolioAllocation` configs with PostgreSQL bars. Missing data or unsupported
configs are recorded as skipped in the batch summary.

To focus only on the new non-CSO strategy suite:

```bash
trader-optimizer optimize-existing \
  --trader-root .. \
  --exclude-strategy-type ConstantStepOffset \
  --trials 50 \
  --max-bars 5000 \
  --workers 4 \
  --output-dir runs/non_cso_existing \
  --plan-path reports/non_cso_optimization_plan.md \
  --export-config-dir optimized_configs/non_cso
```

That writes a generated optimization plan plus stable config files:

- `reports/non_cso_optimization_plan.md`
- `optimized_configs/non_cso/*.optimized.json`
- `optimized_configs/non_cso/index.json`

## Notebook report

The executed notebook report is:

```text
notebooks/TraderOptimizer_Batch_Results.ipynb
```

It loads `runs/batch_existing`, ranks generated configs by objective and return,
shows strategy-family coverage, renders a compact return chart, and prints the
best generated config JSON.

## Hyperparameters

For `ConstantStepOffset`, Optuna tunes:

- `baseline_quantile`: converts the training close-price distribution into a
  baseline.
- `step_delta_pct`: converts a percent of baseline into `stepDelta`.
- `execution_steps`: controls `executionLimitOffset`.
- `threshold_pct_of_step`: controls `stateTransitionThreshold`.
- `order_quantity_usd`: controls `orderQuantityInUSD`.

For non-CSO configs, Optuna tunes the strategy-specific fields that TraderCore
already parses:

- `MovingAverageCross`: `fastWindow`, `slowWindow`, `orderQuantity`, and
  derived `orderQuantityInUSD`.
- `TechnicalSignal` TS-002/003/004/005: the relevant signal windows,
  thresholds, ATR switch, and `orderQuantity`.
- `PortfolioAllocation` QS-001/QS-002/PAIRS-001: volatility, momentum, pair
  z-score, and gross exposure controls.

The objective is a blended train/validation excess-return score against a
buy-and-hold benchmark for the same symbol set, with penalties for configs that
do not trade or that finish with too much marked open inventory. Batch summaries
record strategy return, buy-and-hold return, and excess return. Only configs
that beat buy-and-hold over the full simulated window are exported.
