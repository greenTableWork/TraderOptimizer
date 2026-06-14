# TraderOptimizer

TraderOptimizer is a small, verbose Optuna training loop for producing
TraderCore-style strategy config JSON.

The optimizer reads `public.historical_bars` from PostgreSQL, tries strategy
hyperparameters with Optuna, validates generated configs with the real
TraderCore `BackTester`, and stores run details in PostgreSQL:

- `best_config.json`: a TraderCore-compatible strategy config.
- `best_summary.json`: the best trial, metrics, and data window.
- `optimizer_runs`, `optimizer_trials`, and `optimizer_fills` rows in PostgreSQL.
- `optimizer_batch_results` rows for batch optimization summaries.
- Optuna study tables in PostgreSQL.
- JSON config and summary artifacts remain next to each run for review.

The Optuna search still uses fast internal scoring to propose candidates, but
generated configs are BackTester-gated by default before they are exported.

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
study path, best score, best config path, train/validation metrics, and writes a
`backtester` validation payload into `best_summary.json`.

Use `--max-bars 0` if you want to run against the full matching PostgreSQL series.
That can be much slower for the two-year `10 secs` scrape.
Use `--start-utc` and `--end-utc` to run a smaller BackTester validation window
when full-window validation is too expensive.
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

## BackTester validation

By default, every generated config is validated with TraderCore `BackTester`.
The validation writes a temporary BackTester JSON config next to the run under
`backtester/`, then records the BackTester summary path, generated run config,
and benchmark comparisons in `best_summary.json`.

```bash
trader-optimizer optimize \
  --trader-root .. \
  --symbol AAPL \
  --bar-size "10 secs" \
  --trials 25 \
  --max-bars 5000 \
  --skip-backtester-build
```

The BackTester validation gate requires all of these to pass:

- positive strategy return after modeled fees,
- strategy return beats SPX over the same validation window,
- strategy return beats buying and holding the same stock symbols over the same
  validation window.

Use `--no-backtester-validation` only for development checks where exported
configs are not being considered for promotion.

## Optimize existing strategy configs

To discover the checked-in backtesting and stock-stress configs and generate an
optimized config for each one:

```bash
trader-optimizer optimize-existing \
  --trader-root .. \
  --trials 25 \
  --max-bars 5000 \
  --skip-backtester-build \
  --workers 4 \
  --output-dir runs/batch_existing \
  --plan-path reports/batch_optimization_plan.md \
  --export-config-dir ../TraderCore/TraderLogicConfigs/TraderOptimizer/optimized_configs/batch_existing
```

The batch command writes one folder per strategy plus:

- `runs/batch_existing/batch_summary.json`
- `optimizer_runs`, `optimizer_trials`, and `optimizer_batch_results` rows in PostgreSQL

Detailed trial, fill, and batch metrics are stored in PostgreSQL tables rather
than per-run detail files.

Candidate strategies run concurrently by default, up to 4 workers. Pass
`--workers 1` for serial execution or a larger value when PostgreSQL and the
local machine can absorb more parallel studies.
BackTester validation also runs for each generated config, and only configs with
a passing BackTester benchmark status are copied to `--export-config-dir`.

The current discovery path covers:

- `TraderCore/TraderLogicConfigs/TraderCore/configs/backtesting/**/*.json`
- `TraderCore/TraderLogicConfigs/TraderLab/configs/backtests/**/*.json`

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
  --skip-backtester-build \
  --workers 4 \
  --output-dir runs/non_cso_existing \
  --plan-path reports/non_cso_optimization_plan.md \
  --export-config-dir ../TraderCore/TraderLogicConfigs/TraderOptimizer/optimized_configs/non_cso
```

That writes a generated optimization plan plus stable config files:

- `reports/non_cso_optimization_plan.md`
- `../TraderCore/TraderLogicConfigs/TraderOptimizer/optimized_configs/non_cso/*.optimized.json`
- `../TraderCore/TraderLogicConfigs/TraderOptimizer/optimized_configs/non_cso/index.json`

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

The proposal objective is a blended train/validation excess-return score against
a buy-and-hold benchmark for the same symbol set, with penalties for configs
that do not trade or that finish with too much marked open inventory. Batch
summaries record the BackTester-gated strategy return, same-stock hold return,
and excess return. Only configs that pass the BackTester benchmark gates are
exported.

## Strategy tuning profile

Generated configs, `best_summary.json`, batch summaries, export indexes, and
benchmark champions include `strategyTuningProfile`. This is the shared metadata
surface for describing what a strategy is tuned for and what the optimizer is
trying to promote.

The profile categorizes:

- direction, including expected up/down behavior and `curveSlopeSeverity`.
  The fallback severity scale defaults to `3`, but historical region
  extraction can now use instrument-normalized slope thresholds so a high-beta
  stock, an index, and a crypto symbol are not forced onto the same absolute
  slope scale. When bars are loaded, the profile also records observed slope
  direction and computed slope severity from the current optimization window;
- volatility, split into individual instrument or basket volatility and
  market-volatility regime inputs; loaded bars include realized volatility
  evidence by symbol, and loaded futures bars add market-direction evidence;
- index futures direction, with ES/NQ/YM/RTY proxy candidates loaded from
  `historical_bars` when available so futures can confirm or reject the
  instrument direction;
- options trade data, modeled as a 3D probability map over expiration,
  moneyness, and time for direction and momentum probabilities. If PostgreSQL
  has an `option_trades`, `options_trades`, `historical_option_trades`, or
  `historical_options_trades` table with underlying, time, side, expiry,
  moneyness or strike plus spot, price or premium, and volume columns, the map
  is populated from those trades;
- trade volume plus orderbook integration, using `historical_bars.volume` and
  the pending `codex/l2-orderbook-ingestion` strategy hook for L2 imbalance
  features. Loaded bar volume records relative volume, price-volume
  correlation, a volume-backed direction bias, and the initial fusion direction
  that orderbook imbalance will adjust once L2 events are wired.

## Tuning region extraction

Use `scripts/categorize_historical_bar_regions.py` to turn PostgreSQL
`historical_bars` into backtest-ready regions. The `direction` mode labels day,
week, and month buckets with one category such as
`up_slope_3`, `down_slope_4`, or `flat_slope_1`. Buckets are UTC by default;
pass `--bucket-timezone America/New_York` for NYSE-style equity calendar
buckets. These labels describe the input market region; they are not benchmark
pass/fail statuses.

Example:

```bash
python scripts/build_instrument_slope_severity_config.py \
  --symbol AAPL \
  --bar-size "10 secs" \
  --period day \
  --period week \
  --period month \
  --bucket-timezone America/New_York \
  --output runs/tuning_regions/aapl_slope_severity_config.json

python scripts/categorize_historical_bar_regions.py \
  --subcategory direction \
  --symbol AAPL \
  --bar-size "10 secs" \
  --period day \
  --period week \
  --period month \
  --bucket-timezone America/New_York \
  --slope-severity-config runs/tuning_regions/aapl_slope_severity_config.json \
  --direction up \
  --direction down \
  --output runs/tuning_regions/aapl_direction_regions.jsonl \
  --summary-output runs/tuning_regions/aapl_direction_regions_summary.json
```

Each JSONL row includes `startUtc`, `endUtc`, `category`, `direction`,
`curveSlopeSeverity`, the optional `curveSlopeSeverityThresholds` used to
normalize the severity for that instrument, OHLCV data, return, linear slope,
and a stable `regionId` derived from the symbol, subcategory, bucket window,
category, and proxy fields. The `backtestRegion` object can be fed into a later
BackTester window-selection step.

`build_instrument_slope_severity_config.py` computes absolute `linearSlopePct`
samples for each requested symbol/period and writes the 20/40/60/80 percentile
boundaries as severity 1-4 upper bounds; severity 5 is anything above the fourth
boundary. Use `--quantile` four times to choose a different split. The output
schema is `instrument_slope_severity_config.v1`, and the region creator accepts
it with `--slope-severity-config` for `direction`,
`index-futures-direction`, and `volume-orderbook` runs. If no matching entry is
found for a symbol/period/bar profile, the code falls back to the legacy
absolute thresholds: `< 0.0025`, `< 0.01`, `< 0.03`, `< 0.07`, otherwise
severity 5.

Reruns are duplicate-aware. By default, the script scans prior
`*regions.jsonl` and `*regions.csv` files next to the output path and skips any
generated region whose `regionId` already exists. It also removes duplicate
regions inside the current run, such as a repeated `--symbol`. Use
`--dedupe-path` to scan additional files or directories, or
`--no-dedupe-existing` when you intentionally want a full regenerated artifact.
The summary JSON records generated, skipped, and written region counts under
`dedupe`.

The script also supports the second subcategory, `volatility`. This mode labels
the same day/week/month windows by individual realized volatility and can include
a market proxy such as `SPX` or `ES` when matching bars exist. To normalize
those labels per instrument, first build a volatility regime config:

```bash
python scripts/build_instrument_volatility_regime_config.py \
  --symbol AAPL \
  --symbol SPX \
  --bar-size "10 secs" \
  --period day \
  --period week \
  --period month \
  --bucket-timezone America/New_York \
  --output runs/tuning_regions/aapl_spx_volatility_regime_config.json
```

Then pass that config to the region creator:

```bash
python scripts/categorize_historical_bar_regions.py \
  --subcategory volatility \
  --symbol AAPL \
  --market-symbol SPX \
  --bar-size "10 secs" \
  --period day \
  --period week \
  --period month \
  --bucket-timezone America/New_York \
  --volatility-regime-config runs/tuning_regions/aapl_spx_volatility_regime_config.json \
  --output runs/tuning_regions/aapl_volatility_regions.jsonl \
  --summary-output runs/tuning_regions/aapl_volatility_regions_summary.json
```

Volatility categories are `individual_low_vol`, `individual_medium_vol`, or
`individual_high_vol` when no market proxy is loaded. With a proxy, categories
combine both sides, for example `individual_high_market_low_vol`. Defaults are
`low < 0.01`, `medium < 0.03`, otherwise `high`; override them with
`--low-volatility-threshold-pct` and `--high-volatility-threshold-pct`.
`build_instrument_volatility_regime_config.py` computes realized-volatility
samples for each requested symbol/period and writes the one-third/two-thirds
quantile boundaries as `lowMax` and `mediumMax`; anything above `mediumMax` is
`high`. The output schema is `instrument_volatility_regime_config.v1`, and the
region rows include `individualVolatilityRegimeThresholds` plus
`marketVolatilityRegimeThresholds` when normalized thresholds were supplied.

The third subcategory is `index-futures-direction`. It compares the symbol's
direction with a futures proxy over the same bucket. Pass `--futures-symbol` to
force a proxy, or omit it to try the configured proxy list for the symbol, such
as `NQ` then `ES` for large-cap technology symbols:

```bash
python scripts/categorize_historical_bar_regions.py \
  --subcategory index-futures-direction \
  --symbol AAPL \
  --futures-symbol ES \
  --bar-size "10 secs" \
  --period day \
  --period week \
  --period month \
  --bucket-timezone America/New_York \
  --output runs/tuning_regions/aapl_index_futures_regions.jsonl \
  --summary-output runs/tuning_regions/aapl_index_futures_regions_summary.json
```

Index futures categories include `aligned_up`, `aligned_down`,
`conflicting_symbol_up_future_down`, `conflicting_symbol_down_future_up`, and
neutral cases such as `symbol_flat_future_up`. Use `--futures-alignment aligned`
or `--futures-alignment conflicting` to emit only those backtest windows.

The fourth subcategory is `options-probability-map`. It loads option trades from
PostgreSQL when a compatible table exists, aggregates the trades inside each
day/week/month bar bucket, and embeds a compact 3D probability map over
expiration days, strike moneyness, and trade-time bucket:

```bash
python scripts/categorize_historical_bar_regions.py \
  --subcategory options-probability-map \
  --symbol AAPL \
  --bar-size "10 secs" \
  --period day \
  --period week \
  --period month \
  --bucket-timezone America/New_York \
  --output runs/tuning_regions/aapl_options_probability_regions.jsonl \
  --summary-output runs/tuning_regions/aapl_options_probability_regions_summary.json
```

Options categories include `options_up_momentum_high`,
`options_down_momentum_medium`, and `options_neutral_momentum_low`. The default
momentum thresholds are `< 0.15` for low, `< 0.45` for medium, and otherwise
high. Override them with `--low-options-momentum-threshold` and
`--high-options-momentum-threshold`, or filter emitted windows with
`--options-momentum-regime high`. Compatible option-trade tables are discovered
from `option_trades`, `options_trades`, `historical_option_trades`, or
`historical_options_trades`; pass `--option-trade-table` to override the
candidate list.

The fifth subcategory is `volume-orderbook`. It uses `historical_bars.volume`
to compute relative volume versus the other emitted buckets for the same
period, price-volume correlation, a volume-backed direction vote, and a
volume-only fusion direction. Orderbook fields are present now with
`awaiting_l2_orderbook_ingestion` so the pending L2 imbalance branch can fill in
`bid_ask_imbalance`, `book_pressure`, and `depth_slope` later:

```bash
python scripts/categorize_historical_bar_regions.py \
  --subcategory volume-orderbook \
  --symbol AAPL \
  --bar-size "10 secs" \
  --period day \
  --period week \
  --period month \
  --bucket-timezone America/New_York \
  --output runs/tuning_regions/aapl_volume_orderbook_regions.jsonl \
  --summary-output runs/tuning_regions/aapl_volume_orderbook_regions_summary.json
```

Volume/orderbook categories include `volume_up_high_orderbook_pending`,
`volume_down_normal_orderbook_pending`, and
`volume_neutral_low_orderbook_pending`. Defaults are `low <= 0.75`,
`normal < 1.25`, otherwise `high`, with an up/down volume vote only when
relative volume is at least `1.1` and price direction is not flat. Override
these with `--low-relative-volume-threshold`,
`--high-relative-volume-threshold`, and `--volume-direction-threshold`, or
filter emitted windows with `--volume-regime high` and `--volume-direction up`.

## Regime vector generation

Use `scripts/build_regime_vectors.py` to create one current regime vector per
instrument. The script reads all requested PostgreSQL symbols, learns
instrument-normalized slope and volatility thresholds from the loaded history,
then emits a vector from the latest bucket for the selected normalization
period:

```bash
python scripts/build_regime_vectors.py \
  --all-symbols \
  --bar-size "10 secs" \
  --bucket-timezone America/New_York \
  --normalization-period day \
  --max-bars 5000 \
  --output runs/regime_vectors/all_tickers_regime_vectors.jsonl \
  --summary-output runs/regime_vectors/all_tickers_regime_vectors_summary.json
```

Core fields include direction, slope severity, instrument and market volatility
regime, volatility spread, futures alignment, relative volume, and volume
direction. Advanced fields are marked separately: `regimePersistence`,
`covarianceStress`, `momentumHorizonRegime`, `liquidityOrderFlowRegime`,
`optionsSurfaceRegime`, `distributionClusterId`, and `changePointConfidence`.
The research-corpus note for those advanced parameters lives at
`reports/regime_corpus_bfs_20260531/advanced_regime_parameters.md`.

## Regime tuning universe

Use `scripts/build_regime_tuning_universe.py` to turn regime vectors into
optimizer tasks. The default `matching-symbol` scope emits only tasks that can
run against an existing checked-in strategy config for that ticker. The `all`
scope emits the full ticker-by-strategy universe and marks tasks that need a
retargeted config before they can run:

```bash
python scripts/build_regime_tuning_universe.py \
  --trader-root .. \
  --regime-vectors runs/regime_vectors/all_tickers_regime_vectors.jsonl \
  --strategy-scope matching-symbol \
  --trials 25 \
  --max-bars 0 \
  --output runs/regime_tuning_universe/matching_strategy_universe.jsonl \
  --summary-output runs/regime_tuning_universe/matching_strategy_universe_summary.json
```

Use `scripts/generate_regime_strategy_candidates.py` when the checked-in
strategy configs are too small for a broad regime sweep. It writes deterministic
non-CSO candidates under `runs/`, skips equivalent strategy-family/symbol search
spaces that already exist, and can be safely rerun against the same output
directory:

```bash
python scripts/generate_regime_strategy_candidates.py \
  --trader-root .. \
  --regime-vectors runs/regime_vectors/all_tickers_regime_vectors.jsonl \
  --count 100 \
  --output-dir runs/strategy_candidate_universe/plus100_non_cso_supported
```

Then include both the checked-in configs and generated pack when building the
expanded supported universe:

```bash
python scripts/build_regime_tuning_universe.py \
  --trader-root .. \
  --regime-vectors runs/regime_vectors/all_tickers_regime_vectors.jsonl \
  --config-glob 'TraderCore/TraderLogicConfigs/TraderCore/configs/backtesting/**/*.json' \
  --config-glob 'TraderCore/TraderLogicConfigs/TraderLab/configs/backtests/**/*.json' \
  --config-glob 'TraderOptimizer/runs/strategy_candidate_universe/plus100_non_cso_supported/**/*.json' \
  --optimizer-supported-only \
  --exclude-strategy-type ConstantStepOffset \
  --strategy-scope all \
  --trials 25 \
  --max-bars 0 \
  --output runs/regime_tuning_universe/plus100_non_cso_supported_universe.jsonl
```

Each task stores the ticker, vector start/end duration, regime cell
(`directionSign`, `instrumentVolatilityRegime`, `marketVolatilityRegime`,
`volumeRegime` by default), strategy identity, whether retargeting is required,
and the exact `trader-optimizer optimize-existing` command for the existing
config case.

## Live regime shadow detection

Use `detect-live-regimes` after creating current regime vectors to produce a
causal, smoothed live regime state. This is the bridge between the research
corpus, historical optimizer results, and the live runtime: corpus-backed
detector specs explain each signal, the active regime cell is compatible with
the optimizer universe, and the runtime side is restricted to observing or
selecting BackTester-gated configs.

```bash
trader-optimizer detect-live-regimes \
  --regime-vectors runs/regime_vectors/all_tickers_regime_vectors.jsonl \
  --state-input runs/live_regimes/latest_state.json \
  --state-output runs/live_regimes/latest_state.json \
  --output runs/live_regimes/current_live_regime_detections.jsonl \
  --summary-output runs/live_regimes/current_live_regime_summary.json
```

The detector applies hysteresis before switching active cells. By default, a
new raw cell must appear three times in a row unless
`changePointConfidence >= 0.80`. Use `--write-postgres` to persist the shadow
state into `live_regime_vectors`, `regime_vector_history`,
`regime_transition_events`, and `strategy_selection_decisions`.

`strategySelection` remains `no_validated_config` unless you provide a JSONL
strategy map with `--strategy-map`; every candidate in that map must already be
BackTester-gated with `validationStatus: "ok"`. The live path does not optimize
or promote configs.

Build that map from a completed regime tuning universe run:

```bash
trader-optimizer build-strategy-map \
  --run-summary runs/regime_tuning_universe/latest_run_summary.json \
  --output runs/strategy_regime_maps/latest_strategy_map.jsonl \
  --summary-output runs/strategy_regime_maps/latest_strategy_map_summary.json
```

The map joins each passing optimizer result back to its regime cell, exported
config path, BackTester gate status, same-stock excess return, and SPX excess
return. Use `--write-postgres` to upsert the same rows into
`strategy_regime_config_map`.
