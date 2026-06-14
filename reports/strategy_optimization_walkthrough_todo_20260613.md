# Strategy Optimization Walkthrough Todo - 2026-06-13

Purpose: use this checklist to walk through what happens to each strategy from
candidate discovery through Optuna tuning, BackTester validation, benchmark
gating, export, and regime mapping.

## 1. Pick the walkthrough scope

- [ ] Choose one single-symbol strategy candidate to trace end to end.
- [ ] Choose one portfolio strategy candidate to trace end to end.
- [ ] Choose one failed candidate from a recent run and classify the failure.
- [ ] Choose one passing/exported candidate and verify why it passed.
- [ ] Decide whether this walkthrough is for a normal batch run, a regime
  universe run, or a live-regime selection path.

Suggested initial examples:

- [ ] `MovingAverageCross` for AAPL or MSFT.
- [ ] `TechnicalSignal` TS-002 EMA cross for TSLA or AAPL.
- [ ] `TechnicalSignal` TS-003 Bollinger breakout for AMD.
- [ ] `TechnicalSignal` TS-004 opening range breakout for AAPL.
- [ ] `TechnicalSignal` TS-005 RSI divergence for GOOG.
- [ ] `PortfolioAllocation` QS-001 volatility targeting.
- [ ] `PortfolioAllocation` QS-002 momentum factor.
- [ ] `PortfolioAllocation` PAIRS-001 equity pairs.

## 2. Confirm inputs before optimization

- [ ] Confirm PostgreSQL connection settings: `TRADER_PG_CONNINFO`, `PGHOST`,
  `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`.
- [ ] Confirm `historical_bars` has enough rows for every candidate symbol.
- [ ] Confirm the intended `bar_size`, `what_to_show`, and `use_rth` profile.
- [ ] Confirm the optimization time window: automatic latest bars,
  `--start-utc`/`--end-utc`, or regime-vector window.
- [ ] Confirm `--max-bars`: use `0` only when intentionally running the full
  matching PostgreSQL series.
- [ ] Confirm the strategy config file is discovered by the intended
  `--config-glob`.
- [ ] Confirm unsupported strategy types and CSO are excluded when the run is
  meant to be non-CSO only.
- [ ] Confirm `BackTester` is already built once before high-concurrency runs.

## 3. Candidate discovery flow

- [ ] Open the source strategy JSON.
- [ ] Record `strategy_type`.
- [ ] Record variant field:
  - [ ] `signal_type` for `TechnicalSignal`.
  - [ ] `allocation_type` for `PortfolioAllocation`.
  - [ ] `trendMode` for `MovingAverageCross` variants.
- [ ] Record symbols from `price_contract`/`contract` or `contracts`.
- [ ] Confirm duplicate search spaces are not being run accidentally.
- [ ] Confirm generated candidate packs live under `runs/` and checked-in
  stable configs live under `TraderLogicConfigs`.

## 4. Per-strategy search spaces

- [ ] `ConstantStepOffset`
  - [ ] Tune baseline/step parameters only when CSO is explicitly in scope.
  - [ ] Usually exclude from non-CSO strategy-suite and regime-universe runs.

- [ ] `MovingAverageCross`
  - [ ] Tune `fastWindow`.
  - [ ] Tune `slowWindow`.
  - [ ] Tune `orderQuantity`.
  - [ ] Derive `orderQuantityInUSD`.
  - [ ] For `MATREND-002`, also tune `middleWindow`.
  - [ ] Check whether the optimizer surrogate and BackTester behavior match the
    intended trend mode.

- [ ] `TechnicalSignal` TS-002 EMA cross
  - [ ] Tune `fastWindow`.
  - [ ] Tune `slowWindow`.
  - [ ] Tune `orderQuantity`.
  - [ ] Check train and validation excess return versus same-stock hold.

- [ ] `TechnicalSignal` TS-003 Bollinger breakout
  - [ ] Tune `middleWindow`.
  - [ ] Tune `trendWindow`.
  - [ ] Tune `bandStddev`.
  - [ ] Tune `orderQuantity`.
  - [ ] Check whether breakouts trade enough to avoid no-trade penalties.

- [ ] `TechnicalSignal` TS-004 opening range breakout
  - [ ] Tune `openingRangeBars`.
  - [ ] Tune `useAtrStop`.
  - [ ] Tune `atrWindow`.
  - [ ] Tune `orderQuantity`.
  - [ ] Confirm session/window assumptions are compatible with `use_rth`.

- [ ] `TechnicalSignal` TS-005 RSI divergence
  - [ ] Tune `rsiPeriod`.
  - [ ] Tune `divergenceLookback`.
  - [ ] Tune `orderQuantity`.
  - [ ] Check whether the optimized config creates enough signals.

- [ ] `PortfolioAllocation` QS-001 volatility targeting
  - [ ] Tune `targetVolatility`.
  - [ ] Tune `volatilityWindow`.
  - [ ] Tune `maxGrossExposure`.
  - [ ] Confirm all portfolio symbols have aligned historical bars.

- [ ] `PortfolioAllocation` QS-002 momentum factor
  - [ ] Tune `momentumLookback`.
  - [ ] Tune `momentumLegSize`.
  - [ ] Tune `maxGrossExposure`.
  - [ ] Confirm leg size is valid for the symbol universe.

- [ ] `PortfolioAllocation` PAIRS-001 equity pairs
  - [ ] Tune `pairWindow`.
  - [ ] Tune `pairEntryZ`.
  - [ ] Tune `pairExitZ`.
  - [ ] Tune `maxGrossExposure`.
  - [ ] Confirm configured pairs are present in `contracts`.

## 5. Optimization objective walkthrough

- [ ] Split bars into train and validation windows.
- [ ] Simulate candidate behavior with the optimizer-side surrogate.
- [ ] Compute train excess return versus buy-and-hold.
- [ ] Compute validation excess return versus buy-and-hold.
- [ ] Apply open-inventory penalty.
- [ ] Apply drawdown penalty.
- [ ] Apply no-trade penalty.
- [ ] Confirm Optuna study name and storage.
- [ ] Inspect best trial parameters.
- [ ] Inspect all trials for obvious degenerate search behavior.

## 6. BackTester validation walkthrough

- [ ] Locate generated `best_config.json`.
- [ ] Locate generated BackTester run config.
- [ ] Confirm ledgers/report paths are isolated for concurrent validation.
- [ ] Confirm `BackTester` binary path and build preset.
- [ ] Run or inspect the BackTester validation output.
- [ ] Confirm `backtester.status == "ok"` before considering promotion.
- [ ] If BackTester fails, classify the failure:
  - [ ] Config parse/schema failure.
  - [ ] Missing market data.
  - [ ] Runtime exception/crash.
  - [ ] Ledger/report path collision.
  - [ ] Strategy produced no usable trades.
  - [ ] Benchmark gate failure.

## 7. Promotion benchmarks

- [ ] Positive return gate: strategy return is greater than zero after modeled
  fees.
- [ ] SPX gate: strategy beats SPX over the same validation window.
- [ ] Same-stock hold gate: strategy beats buy-and-hold for the same symbol set.
- [ ] Record absolute strategy return.
- [ ] Record SPX return.
- [ ] Record same-stock hold return.
- [ ] Record excess return versus SPX.
- [ ] Record excess return versus same-stock hold.
- [ ] Reject configs that pass optimizer surrogate scoring but fail promotion
  benchmarks.

## 8. Export and mapping walkthrough

- [ ] Confirm exported configs are written only for benchmark-passing results.
- [ ] Open exported `index.json`.
- [ ] Confirm source config path, optimized config path, summary path, and
  BackTester output path.
- [ ] For regime-universe runs, join run result to:
  - [ ] Universe task.
  - [ ] Regime cell.
  - [ ] Batch summary.
  - [ ] Exported config index.
- [ ] Confirm `strategy_regime_config_map` rows include ticker, strategy,
  variant, regime cell, validation status, and benchmark deltas.

## 9. Failure review questions

- [ ] Did the optimizer fail, or did the candidate simply fail benchmarks?
- [ ] Did PostgreSQL data loading choose the intended bar profile?
- [ ] Did the candidate have enough bars after start/end filtering?
- [ ] Did the optimizer produce trades in both train and validation windows?
- [ ] Did BackTester agree with the optimizer-side surrogate directionally?
- [ ] Did the strategy beat same-stock hold but fail SPX, or the other way
  around?
- [ ] Was the run polluted by duplicate configs with the same symbol and search
  space?
- [ ] Did high concurrency create file-path, ledger, database, or build
  contention?

## 10. Commands to keep handy

```bash
cd /Users/vrajpandya/ws/Trader/TraderOptimizer

.venv/bin/trader-optimizer optimize-existing \
  --trader-root .. \
  --exclude-strategy-type ConstantStepOffset \
  --trials 25 \
  --max-bars 5000 \
  --workers 4 \
  --output-dir runs/walkthrough_batch \
  --plan-path reports/walkthrough_batch_plan.md \
  --export-config-dir ../TraderCore/TraderLogicConfigs/TraderOptimizer/optimized_configs/walkthrough_batch
```

```bash
cd /Users/vrajpandya/ws/Trader/TraderCore
cmake --build --preset debug --target BackTester
```

```bash
cd /Users/vrajpandya/ws/Trader/TraderOptimizer
.venv/bin/pytest -q
```

## 11. Artifacts to open for each walked strategy

- [ ] Source strategy config JSON.
- [ ] Generated optimization plan markdown.
- [ ] Per-candidate output directory.
- [ ] `best_config.json`.
- [ ] `best_summary.json`.
- [ ] `batch_summary.json`.
- [ ] BackTester run config.
- [ ] BackTester output summary JSON.
- [ ] Exported optimized config, if any.
- [ ] Export `index.json`, if any.
- [ ] PostgreSQL `optimizer_runs` row.
- [ ] PostgreSQL `optimizer_trials` rows.
- [ ] PostgreSQL `optimizer_batch_results` row.

## 12. Walkthrough output we want at the end

- [ ] One plain-English sequence diagram for a single-symbol strategy.
- [ ] One plain-English sequence diagram for a portfolio strategy.
- [ ] A table of each strategy family and its tuned parameters.
- [ ] A table of failure categories with examples from recent logs.
- [ ] A clear answer for where benchmark gates are enforced.
- [ ] A clear answer for where regime cells enter the tuning/mapping process.
- [ ] A short list of code changes needed if we find gaps during the walkthrough.
