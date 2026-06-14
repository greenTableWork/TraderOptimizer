# TraderOptimizer Config Optimization Plan

This plan is generated from the current Trader strategy config JSON files.
It describes what Optuna will tune, which PostgreSQL data profile will be used, and where the resulting artifacts will be written.

## Run Settings

- PostgreSQL data: `postgresql://vrajpandya@127.0.0.1:5432/trader`
- Optuna storage: `postgresql+psycopg2://vrajpandya@127.0.0.1:5432/trader`
- Output directory: `/Users/vrajpandya/.openclaw/workspace/Trader/TraderOptimizer/runs/regime_tuning_smoke/concurrent_precreated_20260601`
- Exported configs: not requested
- Trials per config: `2`
- Max bars per symbol: `0`
- Start UTC: `2026-05-29T13:30:00+00:00`
- End UTC: `2026-05-29T19:59:50+00:00`
- Strategy budget: `config default`
- Train fraction: `0.7`
- Preferred bar size: `10 secs`
- Workers: `6`

## Search Spaces

- `MovingAverageCross`: `fastWindow`, `slowWindow`, `orderQuantity`, and derived `orderQuantityInUSD`.
- `TechnicalSignal` TS-002 EMA cross: `fastWindow`, `slowWindow`, and `orderQuantity`.
- `TechnicalSignal` TS-003 Bollinger breakout: `middleWindow`, `trendWindow`, `bandStddev`, and `orderQuantity`.
- `TechnicalSignal` TS-004 opening range breakout: `openingRangeBars`, `useAtrStop`, `atrWindow`, and `orderQuantity`.
- `TechnicalSignal` TS-005 RSI divergence: `rsiPeriod`, `divergenceLookback`, and `orderQuantity`.
- `PortfolioAllocation` QS-001 volatility targeting: `targetVolatility`, `volatilityWindow`, and `maxGrossExposure`.
- `PortfolioAllocation` QS-002 momentum factor: `momentumLookback`, `momentumLegSize`, and `maxGrossExposure`.
- `PortfolioAllocation` PAIRS-001 equity pairs: `pairWindow`, `pairEntryZ`, `pairExitZ`, and `maxGrossExposure`.

## Strategy Tuning Categories

- `direction`: expected up/down behavior plus `curveSlopeSeverity`, defaulting to `3`.
- `volatility`: individual instrument/basket volatility plus planned market-volatility regime inputs.
- `indexFuturesDirection`: ES/NQ/YM/RTY proxy direction where futures can inform the instrument.
- `optionsProbabilityMap3d`: options-trade probability map over expiry, moneyness, and time.
- `tradeVolumeOrderbook`: current bar volume plus L2 orderbook imbalance from `codex/l2-orderbook-ingestion`.

## Objective

The objective blends train and validation excess return versus a buy-and-hold benchmark for the same symbol set, then penalizes open inventory, drawdown, and no-trade configurations. Every exported config is validated with TraderCore BackTester and must have positive return, beat SPX, and beat same-stock buy-and-hold over the BackTester validation window.

## Strategy Coverage

| Config | Type | Variant | Symbols | Data profile | Tuned fields | Categories | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `constant_step_offset_aapl_tws_postgres` | `ConstantStepOffset` | `CSO` | AAPL | AAPL: 10 secs TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd | direction, volatility, indexFuturesDirection, optionsProbabilityMap3d, tradeVolumeOrderbook | up_or_down_around_baseline / mean_reversion_band / slope 3 |
| `moving_average_cross_aapl_tws_postgres` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD | direction, volatility, indexFuturesDirection, optionsProbabilityMap3d, tradeVolumeOrderbook | up / long_flat / slope 3 |
| `matrend001_single_ma_aapl_postgres` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD | direction, volatility, indexFuturesDirection, optionsProbabilityMap3d, tradeVolumeOrderbook | up / long_flat / slope 3 |
| `matrend002_triple_ma_aapl_postgres` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD | direction, volatility, indexFuturesDirection, optionsProbabilityMap3d, tradeVolumeOrderbook | up / long_flat / slope 3 |
| `mw001_moving_average_cross_aapl_postgres` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD | direction, volatility, indexFuturesDirection, optionsProbabilityMap3d, tradeVolumeOrderbook | up / long_flat / slope 3 |
| `qs001_volatility_targeting_top_stocks_postgres` | `PortfolioAllocation` | `QS-001` | AAPL, AMD, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA | AAPL: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure | direction, volatility, indexFuturesDirection, optionsProbabilityMap3d, tradeVolumeOrderbook | up_scaled_by_volatility / adaptive_long_exposure / slope 3 |

## Validation Path

1. Inspect the generated `best_summary.json`, `backtester` payload, and PostgreSQL `optimizer_trials` rows for each strategy.
2. Promote only configs with a passing BackTester validation status.
3. Use smaller `--max-bars`, `--start-utc`, or `--end-utc` windows when the BackTester validation cost is too high.
