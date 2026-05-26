# TraderOptimizer Config Optimization Plan

This plan is generated from the current Trader strategy config JSON files.
It describes what Optuna will tune, which PostgreSQL data profile will be used, and where the resulting artifacts will be written.

## Run Settings

- PostgreSQL data: `postgresql://vrajpandya@127.0.0.1:5432/trader`
- Optuna storage: `postgresql+psycopg2://vrajpandya@127.0.0.1:5432/trader`
- Output directory: `/Users/vrajpandya/.openclaw/workspace/Trader/TraderOptimizer/runs/buy_hold_concurrent_20260524T0225Z`
- Exported configs: `/Users/vrajpandya/.openclaw/workspace/Trader/TraderCore/TraderLogicConfigs/TraderOptimizer/optimized_configs/buy_hold_concurrent`
- Trials per config: `40`
- Max bars per symbol: `5000`
- Train fraction: `0.7`
- Preferred bar size: `auto`
- Workers: `4`

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

The objective blends train and validation excess return versus a buy-and-hold benchmark for the same symbol set, then penalizes open inventory, drawdown, and no-trade configurations. Only configs that beat buy-and-hold over the full simulated window are exported. The simulator includes the local stock commission model and uses close-price fills for the non-CSO strategy families.

## Strategy Coverage

| Config | Type | Variant | Symbols | Data profile | Tuned fields |
| --- | --- | --- | --- | --- | --- |
| `constant_step_offset_aapl_tws_postgres` | `ConstantStepOffset` | `CSO` | AAPL | AAPL: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_amd_tws_postgres` | `ConstantStepOffset` | `CSO` | AMD | AMD: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_ltc_tws_postgres` | `ConstantStepOffset` | `CSO` | LTC | LTC: 5 mins AGGTRADES rth=0 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `moving_average_cross_aapl_tws_postgres` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 1 min TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_moving_average_cross_aapl_postgres` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 1 min TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `pairs001_equity_pairs_postgres` | `PortfolioAllocation` | `PAIRS-001` | GOOGL, GOOG, AMD, NVDA, AAPL, MSFT | GOOGL: 1 min TRADES rth=1<br>GOOG: 1 min TRADES rth=1<br>AMD: 1 min TRADES rth=1<br>NVDA: 1 min TRADES rth=1<br>AAPL: 1 min TRADES rth=1<br>MSFT: 1 min TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `qs001_volatility_targeting_top_stocks_postgres` | `PortfolioAllocation` | `QS-001` | AAPL, AMD, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA | AAPL: 1 min TRADES rth=1<br>AMD: 1 min TRADES rth=1<br>AMZN: 1 min TRADES rth=1<br>AVGO: 1 min TRADES rth=1<br>GOOG: 1 min TRADES rth=1<br>GOOGL: 1 min TRADES rth=1<br>META: 1 min TRADES rth=1<br>MSFT: 1 min TRADES rth=1<br>NVDA: 1 min TRADES rth=1<br>TSLA: 1 min TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs002_momentum_factor_top_stocks_postgres` | `PortfolioAllocation` | `QS-002` | AAPL, AMD, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA | AAPL: 1 min TRADES rth=1<br>AMD: 1 min TRADES rth=1<br>AMZN: 1 min TRADES rth=1<br>AVGO: 1 min TRADES rth=1<br>GOOG: 1 min TRADES rth=1<br>GOOGL: 1 min TRADES rth=1<br>META: 1 min TRADES rth=1<br>MSFT: 1 min TRADES rth=1<br>NVDA: 1 min TRADES rth=1<br>TSLA: 1 min TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `ts002_ema_cross_tsla_postgres` | `TechnicalSignal` | `TS-002` | TSLA | TSLA: 1 min TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts003_bollinger_breakout_amd_postgres` | `TechnicalSignal` | `TS-003` | AMD | AMD: 1 min TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts004_opening_range_breakout_aapl_postgres` | `TechnicalSignal` | `TS-004` | AAPL | AAPL: 1 min TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts005_rsi_divergence_goog_postgres` | `TechnicalSignal` | `TS-005` | GOOG | GOOG: 1 min TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `constant_step_offset_aapl_tws` | `ConstantStepOffset` | `CSO` | AAPL | AAPL: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_amd_tws` | `ConstantStepOffset` | `CSO` | AMD | AMD: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_amzn_tws` | `ConstantStepOffset` | `CSO` | AMZN | AMZN: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_avgo_tws` | `ConstantStepOffset` | `CSO` | AVGO | AVGO: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_brk_b_tws` | `ConstantStepOffset` | `CSO` | BRK.B | BRK.B: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_cost_tws` | `ConstantStepOffset` | `CSO` | COST | COST: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_goog_tws` | `ConstantStepOffset` | `CSO` | GOOG | GOOG: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_googl_tws` | `ConstantStepOffset` | `CSO` | GOOGL | GOOGL: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_intc_tws` | `ConstantStepOffset` | `CSO` | INTC | INTC: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_jnj_tws` | `ConstantStepOffset` | `CSO` | JNJ | JNJ: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_jpm_tws` | `ConstantStepOffset` | `CSO` | JPM | JPM: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_lly_tws` | `ConstantStepOffset` | `CSO` | LLY | LLY: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_meta_tws` | `ConstantStepOffset` | `CSO` | META | META: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_msft_tws` | `ConstantStepOffset` | `CSO` | MSFT | MSFT: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_mu_tws` | `ConstantStepOffset` | `CSO` | MU | MU: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_nvda_tws` | `ConstantStepOffset` | `CSO` | NVDA | NVDA: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_tsla_tws` | `ConstantStepOffset` | `CSO` | TSLA | TSLA: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_v_tws` | `ConstantStepOffset` | `CSO` | V | V: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_wmt_tws` | `ConstantStepOffset` | `CSO` | WMT | WMT: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |
| `constant_step_offset_xom_tws` | `ConstantStepOffset` | `CSO` | XOM | XOM: 1 min TRADES rth=1 | baseline_quantile, step_delta_pct, execution_steps, threshold_pct_of_step, order_quantity_usd |

## Validation Path

1. Inspect the generated `best_summary.json` and PostgreSQL `optimizer_trials` rows for each strategy.
2. Run promising `best_config.json` files through TraderCore `BackTester`; the Python simulator is a fast search harness, not the execution source of truth.
3. Promote only configs that survive the C++ backtest with acceptable fees, trade count, and drawdown.
