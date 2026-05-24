# TraderOptimizer Config Optimization Plan

This plan is generated from the current Trader strategy config JSON files.
It describes what Optuna will tune, which PostgreSQL data profile will be used, and where the resulting artifacts will be written.

## Run Settings

- PostgreSQL data: `historical_bars` in the local `trader` database
- Optuna storage: PostgreSQL
- Output directory: `/Users/vrajpandya/.openclaw/workspace/Trader/TraderOptimizer/runs/non_cso_existing`
- Exported configs: `/Users/vrajpandya/.openclaw/workspace/Trader/TraderOptimizer/optimized_configs/non_cso`
- Trials per config: `50`
- Max bars per symbol: `5000`
- Train fraction: `0.7`
- Preferred bar size: `auto`

## Search Spaces

- `MovingAverageCross`: `fastWindow`, `slowWindow`, `orderQuantity`, and derived `orderQuantityInUSD`.
- `TechnicalSignal` TS-002 EMA cross: `fastWindow`, `slowWindow`, and `orderQuantity`.
- `TechnicalSignal` TS-003 Bollinger breakout: `middleWindow`, `trendWindow`, `bandStddev`, and `orderQuantity`.
- `TechnicalSignal` TS-004 opening range breakout: `openingRangeBars`, `useAtrStop`, `atrWindow`, and `orderQuantity`.
- `TechnicalSignal` TS-005 RSI divergence: `rsiPeriod`, `divergenceLookback`, and `orderQuantity`.
- `PortfolioAllocation` QS-001 volatility targeting: `targetVolatility`, `volatilityWindow`, and `maxGrossExposure`.
- `PortfolioAllocation` QS-002 momentum factor: `momentumLookback`, `momentumLegSize`, and `maxGrossExposure`.
- `PortfolioAllocation` PAIRS-001 equity pairs: `pairWindow`, `pairEntryZ`, `pairExitZ`, and `maxGrossExposure`.

## Objective

The objective blends train and validation simulated return while penalizing open inventory, drawdown, and no-trade configurations. The simulator includes the local stock commission model and uses close-price fills for the non-CSO strategy families.

## Strategy Coverage

| Config | Type | Variant | Symbols | Data profile | Tuned fields |
| --- | --- | --- | --- | --- | --- |
| `moving_average_cross_aapl_tws_postgres` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 1 min TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_moving_average_cross_aapl_postgres` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 1 min TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `pairs001_equity_pairs_postgres` | `PortfolioAllocation` | `PAIRS-001` | GOOGL, GOOG, AMD, NVDA, AAPL, MSFT | GOOGL: 1 min TRADES rth=1<br>GOOG: 1 min TRADES rth=1<br>AMD: 1 min TRADES rth=1<br>NVDA: 1 min TRADES rth=1<br>AAPL: 1 min TRADES rth=1<br>MSFT: 1 min TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `qs001_volatility_targeting_top_stocks_postgres` | `PortfolioAllocation` | `QS-001` | AAPL, AMD, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA | AAPL: 1 min TRADES rth=1<br>AMD: 1 min TRADES rth=1<br>AMZN: 1 min TRADES rth=1<br>AVGO: 1 min TRADES rth=1<br>GOOG: 1 min TRADES rth=1<br>GOOGL: 1 min TRADES rth=1<br>META: 1 min TRADES rth=1<br>MSFT: 1 min TRADES rth=1<br>NVDA: 1 min TRADES rth=1<br>TSLA: 1 min TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs002_momentum_factor_top_stocks_postgres` | `PortfolioAllocation` | `QS-002` | AAPL, AMD, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA | AAPL: 1 min TRADES rth=1<br>AMD: 1 min TRADES rth=1<br>AMZN: 1 min TRADES rth=1<br>AVGO: 1 min TRADES rth=1<br>GOOG: 1 min TRADES rth=1<br>GOOGL: 1 min TRADES rth=1<br>META: 1 min TRADES rth=1<br>MSFT: 1 min TRADES rth=1<br>NVDA: 1 min TRADES rth=1<br>TSLA: 1 min TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `ts002_ema_cross_tsla_postgres` | `TechnicalSignal` | `TS-002` | TSLA | TSLA: 1 min TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts003_bollinger_breakout_amd_postgres` | `TechnicalSignal` | `TS-003` | AMD | AMD: 1 min TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts004_opening_range_breakout_aapl_postgres` | `TechnicalSignal` | `TS-004` | AAPL | AAPL: 1 min TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts005_rsi_divergence_goog_postgres` | `TechnicalSignal` | `TS-005` | GOOG | GOOG: 1 min TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |

## Validation Path

1. Inspect the generated `best_summary.json` and PostgreSQL `optimizer_trials` rows for each strategy.
2. Run promising `best_config.json` files through TraderCore `BackTester`; the Python simulator is a fast search harness, not the execution source of truth.
3. Promote only configs that survive the C++ backtest with acceptable fees, trade count, and drawdown.
