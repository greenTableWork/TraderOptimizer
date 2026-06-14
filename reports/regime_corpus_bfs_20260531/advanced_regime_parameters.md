# Advanced Regime Vector Parameters

Generated: 2026-06-01

These parameters extend the core regime vector. They are advanced because they
need either multi-asset context, online state estimation, options surfaces, or
orderbook integration beyond a single OHLCV window.

| Parameter | Domain or range | Current implementation status | Paper support |
| --- | --- | --- | --- |
| `regimePersistence` | Float score `[0, 1]`, plus matching/state window counts | Heuristic from recent rolling direction windows | Markov and sticky-HMM regime papers: Ang/Timmermann, Ang/Bekaert, Werge, HMM market identification |
| `covarianceStress` | `low`, `normal`, `high`, `not_loaded`; correlation `[-1, 1]`; beta unbounded | Active when a market proxy has aligned bars | Bucci/Ciciretti realized-covariance regimes; cross-asset regime-switching allocation papers |
| `momentumHorizonRegime` | `continuation_up`, `continuation_down`, `reversal_risk`, `mixed`, `flat` | Active from short/medium/long window returns | Time-Series Momentum; Value and Momentum Everywhere; factor-switching HMM papers |
| `liquidityOrderFlowRegime` | `low`, `normal`, `high`, `unknown` plus volume/orderbook evidence status | Volume-only now; L2 orderbook fields pending `codex/l2-orderbook-ingestion` | Order-flow imbalance HMM paper; Market liquidity and funding liquidity |
| `optionsSurfaceRegime` | `bullish`, `bearish`, `neutral`, `not_loaded` | Marked `not_loaded` until IV/skew/term-structure inputs are available for all symbols | Open gap; needs dedicated options-regime seed set |
| `distributionClusterId` | String cluster ID, eventually model-assigned | Shape bucket now; pending Wasserstein/signature clustering model | Wasserstein market-regime clustering; online MMD/signature detection; sliced Wasserstein k-means |
| `changePointConfidence` | Float `[0, 1]` plus direction `stable`, `vol_up`, `vol_down` | Heuristic from recent-vs-prior volatility ratio | Online MMD/signature regime-change detection; hybrid regime-switch classification papers |

The core vector remains:

```text
directionSign
slopeSeverity
instrumentVolatilityRegime
marketVolatilityRegime
volatilitySpread
indexFuturesDirection
indexFuturesAlignment
relativeVolume
volumeRegime
volumeDirection
```

The advanced vector is intentionally stored separately from benchmark status.
These are market-state descriptors for selecting or analyzing tuning windows,
not pass/fail conditions for strategy promotion.
