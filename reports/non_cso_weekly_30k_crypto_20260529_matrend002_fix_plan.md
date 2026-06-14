# TraderOptimizer Config Optimization Plan

This plan is generated from the current Trader strategy config JSON files.
It describes what Optuna will tune, which PostgreSQL data profile will be used, and where the resulting artifacts will be written.

## Run Settings

- PostgreSQL data: `postgresql://vrajpandya@127.0.0.1:5432/trader`
- Optuna storage: `postgresql+psycopg2://vrajpandya@127.0.0.1:5432/trader`
- Output directory: `/Users/vrajpandya/.openclaw/workspace/Trader/TraderOptimizer/runs/non_cso_weekly_30k_crypto_20260529_matrend002_fix`
- Exported configs: not requested
- Trials per config: `25`
- Max bars per symbol: `0`
- Start UTC: `2026-05-26T13:30:00+00:00`
- End UTC: `2026-05-29T19:59:50+00:00`
- Strategy budget: `30000.0`
- Train fraction: `0.7`
- Preferred bar size: `1 min`
- Workers: `2`

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

The objective blends train and validation excess return versus a buy-and-hold benchmark for the same symbol set, then penalizes open inventory, drawdown, and no-trade configurations. Every exported config is validated with TraderCore BackTester and must have positive return, beat SPX, and beat same-stock buy-and-hold over the BackTester validation window.

## Strategy Coverage

| Config | Type | Variant | Symbols | Data profile | Tuned fields |
| --- | --- | --- | --- | --- | --- |
| `matrend002_btc_weekly_30k_crypto` | `MovingAverageCross` | `MovingAverageCross` | BTC | BTC: 1 min AGGTRADES rth=0 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_ltc_weekly_30k_crypto` | `MovingAverageCross` | `MovingAverageCross` | LTC | LTC: 1 min AGGTRADES rth=0 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |

## Validation Path

1. Inspect the generated `best_summary.json`, `backtester` payload, and PostgreSQL `optimizer_trials` rows for each strategy.
2. Promote only configs with a passing BackTester validation status.
3. Use smaller `--max-bars`, `--start-utc`, or `--end-utc` windows when the BackTester validation cost is too high.
