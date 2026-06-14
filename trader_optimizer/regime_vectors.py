from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log, sqrt
from pathlib import Path
from statistics import mean, pstdev

from trader_optimizer.config import write_json
from trader_optimizer.data import Bar, DataProfile
from trader_optimizer.market_features import ORDERBOOK_INTEGRATION_BRANCH, index_futures_for_symbol
from trader_optimizer.series_math import (
    correlation as _correlation,
    deltas as _deltas,
    full_window_slope_pct as _full_window_slope_pct,
    returns_from_bars as _returns,
)
from trader_optimizer.slope_severity import slope_severity_from_slope
from trader_optimizer.volatility_regime import volatility_regime_from_realized


REGIME_VECTOR_SCHEMA = "instrument_regime_vector.v1"
DEFAULT_FLAT_THRESHOLD_PCT = 0.0025
DEFAULT_LOW_RELATIVE_VOLUME_THRESHOLD = 0.75
DEFAULT_HIGH_RELATIVE_VOLUME_THRESHOLD = 1.25


ADVANCED_REGIME_PARAMETERS: tuple[str, ...] = (
    "regimePersistence",
    "covarianceStress",
    "momentumHorizonRegime",
    "liquidityOrderFlowRegime",
    "optionsSurfaceRegime",
    "distributionClusterId",
    "changePointConfidence",
)


REGIME_VECTOR_DOMAINS: dict[str, object] = {
    "directionSign": ["up", "down", "flat"],
    "slopeSeverity": {"type": "integer", "range": [1, 5]},
    "instrumentVolatilityRegime": ["low", "medium", "high"],
    "marketVolatilityRegime": ["low", "medium", "high", "not_loaded"],
    "volatilitySpread": {"type": "float", "range": "unbounded"},
    "indexFuturesAlignment": ["aligned", "conflicting", "neutral_or_unknown", "not_loaded"],
    "regimePersistence": {"type": "float", "range": [0.0, 1.0], "advanced": True},
    "covarianceStress": ["low", "normal", "high", "not_loaded"],
    "momentumHorizonRegime": [
        "continuation_up",
        "continuation_down",
        "reversal_risk",
        "mixed",
        "flat",
    ],
    "liquidityOrderFlowRegime": ["low", "normal", "high", "unknown"],
    "optionsSurfaceRegime": ["bullish", "bearish", "neutral", "not_loaded"],
    "distributionClusterId": {"type": "string", "advanced": True},
    "changePointConfidence": {"type": "float", "range": [0.0, 1.0], "advanced": True},
}


@dataclass(frozen=True)
class RegimeVectorContext:
    symbol: str
    bars: Sequence[Bar]
    profile: DataProfile
    market_symbol: str | None = None
    market_bars: Sequence[Bar] | None = None
    futures_symbol: str | None = None
    futures_bars: Sequence[Bar] | None = None
    slope_severity_thresholds: Sequence[float] | Mapping[str, float] | None = None
    volatility_regime_thresholds: Sequence[float] | Mapping[str, float] | None = None
    market_volatility_regime_thresholds: Sequence[float] | Mapping[str, float] | None = None


def build_regime_vector(
    context: RegimeVectorContext,
    *,
    flat_threshold_pct: float = DEFAULT_FLAT_THRESHOLD_PCT,
    generated_utc: str | None = None,
) -> dict[str, object]:
    bars = sorted(context.bars, key=lambda bar: _parse_utc(bar.timestamp_utc))
    if len(bars) < 2:
        raise ValueError(f"Need at least two bars to build a regime vector for {context.symbol}")
    closes = [float(bar.close) for bar in bars]
    returns = _returns(bars)
    slope_pct = _full_window_slope_pct(closes)
    direction_sign = _direction_from_slope(slope_pct, flat_threshold_pct)
    realized_volatility = _realized_volatility_pct(bars)

    market_stats = _market_volatility_stats(
        context.market_symbol,
        context.market_bars,
        context.market_volatility_regime_thresholds,
    )
    futures_stats = _futures_stats(
        context.symbol,
        direction_sign,
        context.futures_symbol,
        context.futures_bars,
        flat_threshold_pct=flat_threshold_pct,
    )
    volume_stats = _volume_stats(bars)
    vector = {
        "schema": REGIME_VECTOR_SCHEMA,
        "generatedUtc": generated_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "symbol": context.symbol,
        "barSize": context.profile.bar_size,
        "whatToShow": context.profile.what_to_show,
        "useRth": context.profile.use_rth,
        "startUtc": bars[0].timestamp_utc,
        "endUtc": bars[-1].timestamp_utc,
        "bars": len(bars),
        "core": {
            "directionSign": direction_sign,
            "slopeSeverity": slope_severity_from_slope(
                slope_pct,
                context.slope_severity_thresholds,
            ),
            "linearSlopePct": slope_pct,
            "windowReturnPct": closes[-1] / closes[0] - 1.0 if closes[0] else 0.0,
            "instrumentVolatilityRegime": volatility_regime_from_realized(
                realized_volatility,
                context.volatility_regime_thresholds,
            ),
            "realizedVolatilityPct": realized_volatility,
            "marketVolatilityRegime": market_stats["regime"],
            "marketRealizedVolatilityPct": market_stats["realizedVolatilityPct"],
            "volatilitySpread": (
                realized_volatility - float(market_stats["realizedVolatilityPct"])
                if market_stats["realizedVolatilityPct"] is not None
                else None
            ),
            "indexFuturesDirection": futures_stats["direction"],
            "indexFuturesAlignment": futures_stats["alignment"],
            "indexFuturesSymbol": futures_stats["symbol"],
            "relativeVolume": volume_stats.get("relativeVolume"),
            "volumeRegime": volume_stats.get("volumeRegime", "unknown"),
            "volumeDirection": volume_stats.get("volumeDirection", "neutral"),
        },
        "advanced": {
            "regimePersistence": _regime_persistence(bars, direction_sign, flat_threshold_pct),
            "covarianceStress": _covariance_stress(bars, context.market_bars),
            "momentumHorizonRegime": _momentum_horizon_regime(closes, flat_threshold_pct),
            "liquidityOrderFlowRegime": _liquidity_orderflow_regime(volume_stats),
            "optionsSurfaceRegime": {
                "status": "not_loaded",
                "regime": "not_loaded",
                "requiredInputs": [
                    "implied_volatility_rank",
                    "skew",
                    "term_structure",
                    "call_put_pressure",
                ],
            },
            "distributionClusterId": _distribution_cluster_id(returns, realized_volatility),
            "changePointConfidence": _change_point_confidence(bars),
        },
        "domains": REGIME_VECTOR_DOMAINS,
        "advancedParameters": list(ADVANCED_REGIME_PARAMETERS),
        "method": {
            "core": "historical_bars_current_window",
            "advanced": "heuristic_from_historical_bars_until_dedicated_paper_models_are_implemented",
            "orderbookIntegrationBranch": ORDERBOOK_INTEGRATION_BRANCH,
        },
    }
    return vector


def write_regime_vectors_jsonl(path: Path, vectors: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vector in vectors:
            handle.write(json.dumps(vector, sort_keys=True))
            handle.write("\n")


def write_regime_vector_summary(
    path: Path,
    vectors: Sequence[Mapping[str, object]],
    *,
    skipped_symbols: Sequence[Mapping[str, object]] = (),
) -> None:
    core_counts: dict[str, dict[str, int]] = {
        "directionSign": {},
        "instrumentVolatilityRegime": {},
        "marketVolatilityRegime": {},
        "indexFuturesAlignment": {},
        "volumeRegime": {},
    }
    for vector in vectors:
        core = vector.get("core")
        if not isinstance(core, Mapping):
            continue
        for field in core_counts:
            value = str(core.get(field, "unknown"))
            core_counts[field][value] = core_counts[field].get(value, 0) + 1
    summary = {
        "schema": "instrument_regime_vector_summary.v1",
        "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "vectorSchema": REGIME_VECTOR_SCHEMA,
        "vectors": len(vectors),
        "symbols": sorted(str(vector.get("symbol")) for vector in vectors),
        "advancedParameters": list(ADVANCED_REGIME_PARAMETERS),
        "domains": REGIME_VECTOR_DOMAINS,
        "counts": {field: dict(sorted(values.items())) for field, values in core_counts.items()},
        "skippedSymbols": list(skipped_symbols),
    }
    write_json(path, summary)


def _market_volatility_stats(
    market_symbol: str | None,
    market_bars: Sequence[Bar] | None,
    thresholds: Sequence[float] | Mapping[str, float] | None,
) -> dict[str, object]:
    if not market_symbol or not market_bars or len(market_bars) < 2:
        return {
            "symbol": market_symbol,
            "regime": "not_loaded",
            "realizedVolatilityPct": None,
        }
    realized = _realized_volatility_pct(market_bars)
    return {
        "symbol": market_symbol,
        "regime": volatility_regime_from_realized(realized, thresholds),
        "realizedVolatilityPct": realized,
    }


def _futures_stats(
    symbol: str,
    direction_sign: str,
    futures_symbol: str | None,
    futures_bars: Sequence[Bar] | None,
    *,
    flat_threshold_pct: float,
) -> dict[str, object]:
    if not futures_symbol or not futures_bars or len(futures_bars) < 2:
        return {
            "symbol": None,
            "candidateSymbols": index_futures_for_symbol(symbol),
            "direction": "not_loaded",
            "alignment": "not_loaded",
        }
    futures_closes = [float(bar.close) for bar in futures_bars]
    futures_direction = _direction_from_slope(
        _full_window_slope_pct(futures_closes),
        flat_threshold_pct,
    )
    directional_values = {"up", "down"}
    if direction_sign in directional_values and futures_direction in directional_values:
        alignment = "aligned" if direction_sign == futures_direction else "conflicting"
    else:
        alignment = "neutral_or_unknown"
    return {
        "symbol": futures_symbol,
        "candidateSymbols": index_futures_for_symbol(symbol),
        "direction": futures_direction,
        "alignment": alignment,
    }


def _volume_stats(bars: Sequence[Bar]) -> dict[str, object]:
    volumes = [float(bar.volume) for bar in bars if bar.volume is not None]
    if len(volumes) < 2 or len(volumes) != len(bars):
        return {"status": "not_loaded"}
    average_volume = mean(volumes)
    latest_volume = volumes[-1]
    relative_volume = latest_volume / average_volume if average_volume > 0 else 0.0
    returns = _returns(bars)
    price_volume_correlation = _correlation(returns, _deltas(volumes))
    volume_regime = _volume_regime(relative_volume)
    direction = _direction_from_slope(
        _full_window_slope_pct([float(bar.close) for bar in bars]),
        DEFAULT_FLAT_THRESHOLD_PCT,
    )
    volume_direction = (
        direction if relative_volume >= 1.1 and direction in {"up", "down"} else "neutral"
    )
    return {
        "status": "active",
        "averageVolume": average_volume,
        "latestVolume": latest_volume,
        "relativeVolume": relative_volume,
        "volumeRegime": volume_regime,
        "volumeDirection": volume_direction,
        "priceVolumeCorrelation": price_volume_correlation,
    }


def _regime_persistence(
    bars: Sequence[Bar],
    current_direction: str,
    flat_threshold_pct: float,
    *,
    max_windows: int = 20,
) -> dict[str, object]:
    if current_direction == "flat" or len(bars) < 4:
        return {"status": "active", "score": 0.0, "matchingWindows": 0, "windowCount": 0}
    directions: list[str] = []
    for idx in range(2, len(bars) + 1):
        window = bars[max(0, idx - 4):idx]
        if len(window) < 2:
            continue
        direction = _direction_from_slope(
            _full_window_slope_pct([float(bar.close) for bar in window]),
            flat_threshold_pct,
        )
        directions.append(direction)
    recent = directions[-max_windows:]
    matching = 0
    for direction in reversed(recent):
        if direction != current_direction:
            break
        matching += 1
    return {
        "status": "active",
        "score": matching / len(recent) if recent else 0.0,
        "matchingWindows": matching,
        "windowCount": len(recent),
    }


def _covariance_stress(
    bars: Sequence[Bar],
    market_bars: Sequence[Bar] | None,
) -> dict[str, object]:
    if not market_bars or len(market_bars) < 3 or len(bars) < 3:
        return {"status": "not_loaded", "regime": "not_loaded"}
    left, right = _aligned_returns(bars, market_bars)
    if len(left) < 2:
        return {"status": "not_loaded", "regime": "not_loaded"}
    correlation = _correlation(left, right)
    beta = _beta(left, right)
    abs_correlation = abs(correlation)
    if abs_correlation >= 0.80 or abs(beta) >= 1.5:
        regime = "high"
    elif abs_correlation >= 0.50 or abs(beta) >= 0.75:
        regime = "normal"
    else:
        regime = "low"
    return {
        "status": "active",
        "regime": regime,
        "correlation": correlation,
        "beta": beta,
        "observations": len(left),
    }


def _momentum_horizon_regime(
    closes: Sequence[float],
    flat_threshold_pct: float,
) -> dict[str, object]:
    horizons = {
        "short": _window_return(closes, 30),
        "medium": _window_return(closes, 120),
        "long": _window_return(closes, 390),
    }
    active = {
        key: value
        for key, value in horizons.items()
        if value is not None
    }
    if not active:
        return {"status": "not_loaded", "regime": "flat", "returns": horizons}
    directions = {
        key: _direction_from_return(value, flat_threshold_pct)
        for key, value in active.items()
    }
    non_flat = {value for value in directions.values() if value != "flat"}
    if not non_flat:
        regime = "flat"
    elif len(non_flat) == 1:
        only_direction = next(iter(non_flat))
        regime = f"continuation_{only_direction}"
    elif directions.get("short") != directions.get("long"):
        regime = "reversal_risk"
    else:
        regime = "mixed"
    return {
        "status": "active",
        "regime": regime,
        "returns": horizons,
        "directions": directions,
    }


def _liquidity_orderflow_regime(volume_stats: Mapping[str, object]) -> dict[str, object]:
    if volume_stats.get("status") != "active":
        return {
            "status": "volume_not_loaded",
            "regime": "unknown",
            "orderbookStatus": "awaiting_l2_orderbook_ingestion",
        }
    return {
        "status": "volume_only",
        "regime": volume_stats.get("volumeRegime", "unknown"),
        "relativeVolume": volume_stats.get("relativeVolume"),
        "priceVolumeCorrelation": volume_stats.get("priceVolumeCorrelation"),
        "orderbookStatus": "awaiting_l2_orderbook_ingestion",
        "orderbookIntegrationBranch": ORDERBOOK_INTEGRATION_BRANCH,
    }


def _distribution_cluster_id(
    returns: Sequence[float],
    realized_volatility: float,
) -> dict[str, object]:
    if len(returns) < 3:
        return {"status": "not_loaded", "clusterId": "insufficient_returns"}
    skew = _skewness(returns)
    tail_ratio = _tail_ratio(returns)
    skew_bucket = "right_skew" if skew > 0.25 else "left_skew" if skew < -0.25 else "balanced"
    tail_bucket = "tail_heavy" if tail_ratio > 2.0 else "tail_normal"
    volatility_bucket = volatility_regime_from_realized(realized_volatility)
    return {
        "status": "active",
        "clusterId": f"vol_{volatility_bucket}_{skew_bucket}_{tail_bucket}",
        "skewness": skew,
        "tailRatio": tail_ratio,
        "method": "shape_bucket_pending_wasserstein_or_signature_clustering",
    }


def _change_point_confidence(
    bars: Sequence[Bar],
    *,
    window: int = 60,
) -> dict[str, object]:
    if len(bars) < window * 2:
        return {
            "status": "insufficient_window",
            "confidence": 0.0,
            "direction": "stable",
        }
    previous = bars[-window * 2:-window]
    recent = bars[-window:]
    previous_vol = _realized_volatility_pct(previous)
    recent_vol = _realized_volatility_pct(recent)
    if previous_vol <= 0 or recent_vol <= 0:
        confidence = 0.0
    else:
        confidence = min(1.0, abs(log(recent_vol / previous_vol)) / log(3.0))
    if confidence < 0.20:
        direction = "stable"
    elif recent_vol > previous_vol:
        direction = "vol_up"
    else:
        direction = "vol_down"
    return {
        "status": "active",
        "confidence": confidence,
        "direction": direction,
        "previousVolatilityPct": previous_vol,
        "recentVolatilityPct": recent_vol,
    }


def _realized_volatility_pct(bars: Sequence[Bar]) -> float:
    returns = _returns(bars)
    return pstdev(returns) * sqrt(len(returns)) if returns else 0.0


def _direction_from_slope(slope_pct: float, flat_threshold_pct: float) -> str:
    if slope_pct > flat_threshold_pct:
        return "up"
    if slope_pct < -flat_threshold_pct:
        return "down"
    return "flat"


def _direction_from_return(return_pct: float, flat_threshold_pct: float) -> str:
    if return_pct > flat_threshold_pct:
        return "up"
    if return_pct < -flat_threshold_pct:
        return "down"
    return "flat"


def _volume_regime(relative_volume: float) -> str:
    if relative_volume <= DEFAULT_LOW_RELATIVE_VOLUME_THRESHOLD:
        return "low"
    if relative_volume < DEFAULT_HIGH_RELATIVE_VOLUME_THRESHOLD:
        return "normal"
    return "high"


def _aligned_returns(
    bars: Sequence[Bar],
    market_bars: Sequence[Bar],
) -> tuple[list[float], list[float]]:
    by_timestamp = {bar.timestamp_utc: bar for bar in market_bars}
    aligned_bars: list[Bar] = []
    aligned_market_bars: list[Bar] = []
    for bar in bars:
        market_bar = by_timestamp.get(bar.timestamp_utc)
        if market_bar is None:
            continue
        aligned_bars.append(bar)
        aligned_market_bars.append(market_bar)
    return _returns(aligned_bars), _returns(aligned_market_bars)


def _beta(left: Sequence[float], right: Sequence[float]) -> float:
    count = min(len(left), len(right))
    if count < 2:
        return 0.0
    x_values = [float(value) for value in left[-count:]]
    y_values = [float(value) for value in right[-count:]]
    y_mean = mean(y_values)
    x_mean = mean(x_values)
    covariance = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    market_variance = sum((value - y_mean) ** 2 for value in y_values)
    return covariance / market_variance if market_variance > 0 else 0.0


def _window_return(closes: Sequence[float], window: int) -> float | None:
    if len(closes) < 2:
        return None
    if len(closes) <= window:
        first = closes[0]
    else:
        first = closes[-window]
    last = closes[-1]
    return last / first - 1.0 if first else 0.0


def _skewness(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    avg = mean(values)
    stddev = pstdev(values)
    if stddev <= 0:
        return 0.0
    return mean(((value - avg) / stddev) ** 3 for value in values)


def _tail_ratio(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    sorted_abs = sorted(abs(float(value)) for value in values)
    median_abs = sorted_abs[len(sorted_abs) // 2]
    if median_abs <= 0:
        return 0.0
    tail_index = max(0, int(len(sorted_abs) * 0.95) - 1)
    return sorted_abs[tail_index] / median_abs


def _parse_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
