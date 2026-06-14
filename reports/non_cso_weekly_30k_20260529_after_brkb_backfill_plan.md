# TraderOptimizer Config Optimization Plan

This plan is generated from the current Trader strategy config JSON files.
It describes what Optuna will tune, which PostgreSQL data profile will be used, and where the resulting artifacts will be written.

## Run Settings

- PostgreSQL data: `postgresql://vrajpandya@127.0.0.1:5432/trader`
- Optuna storage: `postgresql+psycopg2://vrajpandya@127.0.0.1:5432/trader`
- Output directory: `/Users/vrajpandya/.openclaw/workspace/Trader/TraderOptimizer/runs/non_cso_weekly_30k_20260529_after_brkb_backfill`
- Exported configs: not requested
- Trials per config: `25`
- Max bars per symbol: `0`
- Start UTC: `2026-05-26T13:30:00+00:00`
- End UTC: `2026-05-29T19:59:50+00:00`
- Strategy budget: `30000.0`
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

## Objective

The objective blends train and validation excess return versus a buy-and-hold benchmark for the same symbol set, then penalizes open inventory, drawdown, and no-trade configurations. Every exported config is validated with TraderCore BackTester and must have positive return, beat SPX, and beat same-stock buy-and-hold over the BackTester validation window.

## Strategy Coverage

| Config | Type | Variant | Symbols | Data profile | Tuned fields |
| --- | --- | --- | --- | --- | --- |
| `matrend001_aapl_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_amd_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AMD | AMD: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_amzn_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AMZN | AMZN: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_avgo_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AVGO | AVGO: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_brk_b_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | BRK.B | BRK.B: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_cost_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | COST | COST: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_goog_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | GOOG | GOOG: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_googl_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | GOOGL | GOOGL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_intc_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | INTC | INTC: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_jnj_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | JNJ | JNJ: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_jpm_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | JPM | JPM: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_lly_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | LLY | LLY: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_meta_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | META | META: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_msft_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | MSFT | MSFT: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_mu_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | MU | MU: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_nvda_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | NVDA | NVDA: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_tsla_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | TSLA | TSLA: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_v_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | V | V: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_wmt_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | WMT | WMT: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend001_xom_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | XOM | XOM: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_aapl_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_amd_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AMD | AMD: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_amzn_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AMZN | AMZN: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_avgo_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AVGO | AVGO: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_brk_b_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | BRK.B | BRK.B: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_cost_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | COST | COST: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_goog_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | GOOG | GOOG: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_googl_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | GOOGL | GOOGL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_intc_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | INTC | INTC: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_jnj_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | JNJ | JNJ: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_jpm_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | JPM | JPM: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_lly_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | LLY | LLY: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_meta_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | META | META: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_msft_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | MSFT | MSFT: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_mu_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | MU | MU: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_nvda_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | NVDA | NVDA: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_tsla_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | TSLA | TSLA: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_v_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | V | V: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_wmt_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | WMT | WMT: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `matrend002_xom_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | XOM | XOM: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_aapl_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AAPL | AAPL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_amd_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AMD | AMD: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_amzn_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AMZN | AMZN: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_avgo_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | AVGO | AVGO: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_brk_b_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | BRK.B | BRK.B: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_cost_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | COST | COST: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_goog_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | GOOG | GOOG: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_googl_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | GOOGL | GOOGL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_intc_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | INTC | INTC: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_jnj_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | JNJ | JNJ: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_jpm_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | JPM | JPM: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_lly_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | LLY | LLY: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_meta_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | META | META: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_msft_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | MSFT | MSFT: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_mu_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | MU | MU: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_nvda_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | NVDA | NVDA: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_tsla_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | TSLA | TSLA: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_v_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | V | V: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_wmt_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | WMT | WMT: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `mw001_xom_weekly_30k_ibkr` | `MovingAverageCross` | `MovingAverageCross` | XOM | XOM: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity, orderQuantityInUSD |
| `pairs001_aapl_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | AAPL, MSFT, AMZN, GOOGL, AVGO, GOOG | AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_amd_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | AMD, XOM, WMT, V, JNJ, INTC | AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_amzn_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | AMZN, GOOGL, AVGO, GOOG, META, TSLA | AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_avgo_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | AVGO, GOOG, META, TSLA, BRK.B, JPM | AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_brk_b_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | BRK.B, JPM, LLY, MU, AMD, XOM | BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_cost_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | COST, NVDA, AAPL, MSFT, AMZN, GOOGL | COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_goog_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | GOOG, META, TSLA, BRK.B, JPM, LLY | GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_googl_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | GOOGL, AVGO, GOOG, META, TSLA, BRK.B | GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_intc_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | INTC, COST, NVDA, AAPL, MSFT, AMZN | INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_jnj_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | JNJ, INTC, COST, NVDA, AAPL, MSFT | JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_jpm_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | JPM, LLY, MU, AMD, XOM, WMT | JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_lly_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | LLY, MU, AMD, XOM, WMT, V | LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_meta_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | META, TSLA, BRK.B, JPM, LLY, MU | META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_msft_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | MSFT, AMZN, GOOGL, AVGO, GOOG, META | MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_mu_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | MU, AMD, XOM, WMT, V, JNJ | MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_nvda_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | NVDA, AAPL, MSFT, AMZN, GOOGL, AVGO | NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_tsla_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | TSLA, BRK.B, JPM, LLY, MU, AMD | TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_v_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | V, JNJ, INTC, COST, NVDA, AAPL | V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_wmt_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | WMT, V, JNJ, INTC, COST, NVDA | WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `pairs001_xom_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `PAIRS-001` | XOM, WMT, V, JNJ, INTC, COST | XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1 | pairWindow, pairEntryZ, pairExitZ, maxGrossExposure |
| `qs001_aapl_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | AAPL, MSFT, AMZN, GOOGL, AVGO, GOOG | AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_amd_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | AMD, XOM, WMT, V, JNJ, INTC | AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_amzn_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | AMZN, GOOGL, AVGO, GOOG, META, TSLA | AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_avgo_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | AVGO, GOOG, META, TSLA, BRK.B, JPM | AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_brk_b_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | BRK.B, JPM, LLY, MU, AMD, XOM | BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_cost_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | COST, NVDA, AAPL, MSFT, AMZN, GOOGL | COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_goog_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | GOOG, META, TSLA, BRK.B, JPM, LLY | GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_googl_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | GOOGL, AVGO, GOOG, META, TSLA, BRK.B | GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_intc_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | INTC, COST, NVDA, AAPL, MSFT, AMZN | INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_jnj_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | JNJ, INTC, COST, NVDA, AAPL, MSFT | JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_jpm_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | JPM, LLY, MU, AMD, XOM, WMT | JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_lly_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | LLY, MU, AMD, XOM, WMT, V | LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_meta_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | META, TSLA, BRK.B, JPM, LLY, MU | META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_msft_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | MSFT, AMZN, GOOGL, AVGO, GOOG, META | MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_mu_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | MU, AMD, XOM, WMT, V, JNJ | MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_nvda_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | NVDA, AAPL, MSFT, AMZN, GOOGL, AVGO | NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_tsla_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | TSLA, BRK.B, JPM, LLY, MU, AMD | TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_v_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | V, JNJ, INTC, COST, NVDA, AAPL | V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_wmt_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | WMT, V, JNJ, INTC, COST, NVDA | WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs001_xom_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-001` | XOM, WMT, V, JNJ, INTC, COST | XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1 | targetVolatility, volatilityWindow, maxGrossExposure |
| `qs002_aapl_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | AAPL, MSFT, AMZN, GOOGL, AVGO, GOOG | AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_amd_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | AMD, XOM, WMT, V, JNJ, INTC | AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_amzn_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | AMZN, GOOGL, AVGO, GOOG, META, TSLA | AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_avgo_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | AVGO, GOOG, META, TSLA, BRK.B, JPM | AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_brk_b_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | BRK.B, JPM, LLY, MU, AMD, XOM | BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_cost_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | COST, NVDA, AAPL, MSFT, AMZN, GOOGL | COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_goog_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | GOOG, META, TSLA, BRK.B, JPM, LLY | GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_googl_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | GOOGL, AVGO, GOOG, META, TSLA, BRK.B | GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_intc_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | INTC, COST, NVDA, AAPL, MSFT, AMZN | INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_jnj_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | JNJ, INTC, COST, NVDA, AAPL, MSFT | JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_jpm_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | JPM, LLY, MU, AMD, XOM, WMT | JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_lly_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | LLY, MU, AMD, XOM, WMT, V | LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_meta_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | META, TSLA, BRK.B, JPM, LLY, MU | META: 10 secs TRADES rth=1<br>TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_msft_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | MSFT, AMZN, GOOGL, AVGO, GOOG, META | MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1<br>GOOG: 10 secs TRADES rth=1<br>META: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_mu_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | MU, AMD, XOM, WMT, V, JNJ | MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1<br>XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_nvda_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | NVDA, AAPL, MSFT, AMZN, GOOGL, AVGO | NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1<br>MSFT: 10 secs TRADES rth=1<br>AMZN: 10 secs TRADES rth=1<br>GOOGL: 10 secs TRADES rth=1<br>AVGO: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_tsla_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | TSLA, BRK.B, JPM, LLY, MU, AMD | TSLA: 10 secs TRADES rth=1<br>BRK.B: 10 secs TRADES rth=1<br>JPM: 10 secs TRADES rth=1<br>LLY: 10 secs TRADES rth=1<br>MU: 10 secs TRADES rth=1<br>AMD: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_v_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | V, JNJ, INTC, COST, NVDA, AAPL | V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1<br>AAPL: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_wmt_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | WMT, V, JNJ, INTC, COST, NVDA | WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1<br>NVDA: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `qs002_xom_basket_weekly_30k_ibkr` | `PortfolioAllocation` | `QS-002` | XOM, WMT, V, JNJ, INTC, COST | XOM: 10 secs TRADES rth=1<br>WMT: 10 secs TRADES rth=1<br>V: 10 secs TRADES rth=1<br>JNJ: 10 secs TRADES rth=1<br>INTC: 10 secs TRADES rth=1<br>COST: 10 secs TRADES rth=1 | momentumLookback, momentumLegSize, maxGrossExposure |
| `ts002_aapl_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | AAPL | AAPL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_amd_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | AMD | AMD: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_amzn_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | AMZN | AMZN: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_avgo_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | AVGO | AVGO: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_brk_b_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | BRK.B | BRK.B: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_cost_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | COST | COST: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_goog_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | GOOG | GOOG: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_googl_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | GOOGL | GOOGL: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_intc_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | INTC | INTC: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_jnj_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | JNJ | JNJ: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_jpm_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | JPM | JPM: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_lly_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | LLY | LLY: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_meta_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | META | META: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_msft_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | MSFT | MSFT: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_mu_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | MU | MU: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_nvda_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | NVDA | NVDA: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_tsla_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | TSLA | TSLA: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_v_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | V | V: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_wmt_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | WMT | WMT: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts002_xom_weekly_30k_ibkr` | `TechnicalSignal` | `TS-002` | XOM | XOM: 10 secs TRADES rth=1 | fastWindow, slowWindow, orderQuantity |
| `ts003_aapl_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | AAPL | AAPL: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_amd_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | AMD | AMD: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_amzn_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | AMZN | AMZN: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_avgo_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | AVGO | AVGO: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_brk_b_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | BRK.B | BRK.B: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_cost_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | COST | COST: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_goog_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | GOOG | GOOG: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_googl_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | GOOGL | GOOGL: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_intc_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | INTC | INTC: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_jnj_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | JNJ | JNJ: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_jpm_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | JPM | JPM: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_lly_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | LLY | LLY: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_meta_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | META | META: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_msft_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | MSFT | MSFT: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_mu_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | MU | MU: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_nvda_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | NVDA | NVDA: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_tsla_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | TSLA | TSLA: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_v_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | V | V: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_wmt_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | WMT | WMT: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts003_xom_weekly_30k_ibkr` | `TechnicalSignal` | `TS-003` | XOM | XOM: 10 secs TRADES rth=1 | middleWindow, trendWindow, bandStddev, orderQuantity |
| `ts004_aapl_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | AAPL | AAPL: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_amd_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | AMD | AMD: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_amzn_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | AMZN | AMZN: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_avgo_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | AVGO | AVGO: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_brk_b_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | BRK.B | BRK.B: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_cost_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | COST | COST: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_goog_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | GOOG | GOOG: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_googl_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | GOOGL | GOOGL: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_intc_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | INTC | INTC: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_jnj_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | JNJ | JNJ: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_jpm_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | JPM | JPM: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_lly_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | LLY | LLY: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_meta_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | META | META: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_msft_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | MSFT | MSFT: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_mu_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | MU | MU: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_nvda_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | NVDA | NVDA: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_tsla_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | TSLA | TSLA: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_v_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | V | V: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_wmt_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | WMT | WMT: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts004_xom_weekly_30k_ibkr` | `TechnicalSignal` | `TS-004` | XOM | XOM: 10 secs TRADES rth=1 | openingRangeBars, useAtrStop, atrWindow, orderQuantity |
| `ts005_aapl_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | AAPL | AAPL: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_amd_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | AMD | AMD: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_amzn_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | AMZN | AMZN: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_avgo_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | AVGO | AVGO: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_brk_b_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | BRK.B | BRK.B: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_cost_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | COST | COST: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_goog_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | GOOG | GOOG: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_googl_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | GOOGL | GOOGL: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_intc_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | INTC | INTC: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_jnj_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | JNJ | JNJ: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_jpm_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | JPM | JPM: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_lly_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | LLY | LLY: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_meta_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | META | META: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_msft_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | MSFT | MSFT: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_mu_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | MU | MU: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_nvda_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | NVDA | NVDA: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_tsla_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | TSLA | TSLA: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_v_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | V | V: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_wmt_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | WMT | WMT: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |
| `ts005_xom_weekly_30k_ibkr` | `TechnicalSignal` | `TS-005` | XOM | XOM: 10 secs TRADES rth=1 | rsiPeriod, divergenceLookback, orderQuantity |

## Validation Path

1. Inspect the generated `best_summary.json`, `backtester` payload, and PostgreSQL `optimizer_trials` rows for each strategy.
2. Promote only configs with a passing BackTester validation status.
3. Use smaller `--max-bars`, `--start-utc`, or `--end-utc` windows when the BackTester validation cost is too high.
