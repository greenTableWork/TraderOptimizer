from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev

from trader_optimizer.data import Bar
from trader_optimizer.series_math import (
    correlation as _correlation,
    deltas as _deltas,
    full_window_slope_pct as _full_window_slope_pct,
    returns_from_bars as _returns,
)


DEFAULT_CURVE_SLOPE_SEVERITY = 3
ORDERBOOK_INTEGRATION_BRANCH = "codex/l2-orderbook-ingestion"
TECH_HEAVY_SYMBOLS = {
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "GOOG",
    "GOOGL",
    "INTC",
    "META",
    "MSFT",
    "NVDA",
    "QQQ",
    "TSLA",
}
SMALL_CAP_SYMBOLS = {"IWM", "RUT"}
DOW_SYMBOLS = {"DIA", "DJI"}


@dataclass(frozen=True)
class OptionTrade:
    underlying: str
    trade_time_utc: str
    expiration_days: int
    strike_moneyness: float
    side: str
    premium: float
    volume: float
    open_interest: float | None = None
    implied_volatility: float | None = None


def build_market_feature_summary(
    symbol_bars: Mapping[str, Sequence[Bar]],
    *,
    index_futures_bars: Mapping[str, Sequence[Bar]] | None = None,
    option_trades: Sequence[OptionTrade] | None = None,
) -> dict[str, object]:
    direction_by_symbol = {
        symbol: _direction_features(bars)
        for symbol, bars in symbol_bars.items()
        if len(bars) >= 2
    }
    volatility_by_symbol = {
        symbol: _volatility_features(bars)
        for symbol, bars in symbol_bars.items()
        if len(bars) >= 2
    }
    volume_by_symbol = {
        symbol: features
        for symbol, bars in symbol_bars.items()
        if (features := _volume_features(bars)) is not None
    }
    futures_direction = {
        symbol: _direction_features(bars)
        for symbol, bars in (index_futures_bars or {}).items()
        if len(bars) >= 2
    }
    options_map = build_options_probability_map(option_trades or ())

    return {
        "direction": {
            "aggregate": _aggregate_direction(direction_by_symbol),
            "bySymbol": direction_by_symbol,
        },
        "volatility": {
            "bySymbol": volatility_by_symbol,
            "marketByFuture": futures_direction,
        },
        "volume": {
            "bySymbol": volume_by_symbol,
            "aggregate": _aggregate_volume(volume_by_symbol),
        },
        "tradeVolumeOrderbook": _trade_volume_orderbook_fusion(volume_by_symbol),
        "indexFuturesDirection": {
            "status": "active" if futures_direction else "not_loaded",
            "aggregate": _aggregate_direction(futures_direction),
            "byFuture": futures_direction,
            "instrumentAlignment": _futures_instrument_alignment(
                direction_by_symbol,
                futures_direction,
            ),
        },
        "optionsProbabilityMap3d": options_map,
    }


def build_options_probability_map(
    trades: Sequence[OptionTrade],
) -> dict[str, object]:
    cells: dict[tuple[str, str, str, str], dict[str, float | str]] = defaultdict(
        lambda: {
            "callPremium": 0.0,
            "putPremium": 0.0,
            "callVolume": 0.0,
            "putVolume": 0.0,
            "openInterest": 0.0,
            "impliedVolatilityWeighted": 0.0,
            "impliedVolatilityWeight": 0.0,
        }
    )
    for trade in trades:
        side = trade.side.upper()
        expiration_bucket = _expiration_bucket(trade.expiration_days)
        moneyness_bucket = _moneyness_bucket(trade.strike_moneyness)
        time_bucket = _time_bucket(trade.trade_time_utc)
        key = (trade.underlying.upper(), expiration_bucket, moneyness_bucket, time_bucket)
        cell = cells[key]
        if side.startswith("C"):
            cell["callPremium"] = float(cell["callPremium"]) + float(trade.premium)
            cell["callVolume"] = float(cell["callVolume"]) + float(trade.volume)
        elif side.startswith("P"):
            cell["putPremium"] = float(cell["putPremium"]) + float(trade.premium)
            cell["putVolume"] = float(cell["putVolume"]) + float(trade.volume)
        cell["openInterest"] = float(cell["openInterest"]) + float(trade.open_interest or 0.0)
        if trade.implied_volatility is not None and trade.volume > 0:
            cell["impliedVolatilityWeighted"] = float(
                cell["impliedVolatilityWeighted"]
            ) + float(trade.implied_volatility) * float(trade.volume)
            cell["impliedVolatilityWeight"] = float(
                cell["impliedVolatilityWeight"]
            ) + float(trade.volume)

    output_cells: list[dict[str, object]] = []
    aggregate_by_underlying: dict[str, dict[str, float]] = defaultdict(
        lambda: {"callWeight": 0.0, "putWeight": 0.0, "cells": 0.0}
    )
    for (underlying, expiration_bucket, moneyness_bucket, time_bucket), raw in sorted(
        cells.items()
    ):
        call_weight = float(raw["callPremium"]) + float(raw["callVolume"])
        put_weight = float(raw["putPremium"]) + float(raw["putVolume"])
        total_weight = call_weight + put_weight
        up_probability = call_weight / total_weight if total_weight > 0 else 0.5
        down_probability = put_weight / total_weight if total_weight > 0 else 0.5
        momentum_probability = abs(up_probability - down_probability)
        iv_weight = float(raw["impliedVolatilityWeight"])
        aggregate_by_underlying[underlying]["callWeight"] += call_weight
        aggregate_by_underlying[underlying]["putWeight"] += put_weight
        aggregate_by_underlying[underlying]["cells"] += 1
        output_cells.append(
            {
                "underlying": underlying,
                "expirationDaysBucket": expiration_bucket,
                "strikeMoneynessBucket": moneyness_bucket,
                "tradeTimeBucket": time_bucket,
                "upProbability": up_probability,
                "downProbability": down_probability,
                "momentumProbability": momentum_probability,
                "momentumDirection": _direction_from_probability(up_probability),
                "callPremium": raw["callPremium"],
                "putPremium": raw["putPremium"],
                "callVolume": raw["callVolume"],
                "putVolume": raw["putVolume"],
                "openInterest": raw["openInterest"],
                "impliedVolatility": (
                    float(raw["impliedVolatilityWeighted"]) / iv_weight
                    if iv_weight > 0
                    else None
                ),
            }
        )

    return {
        "status": "active" if output_cells else "not_loaded",
        "axes": ["expiration_days", "strike_moneyness", "trade_time_bucket"],
        "predictionTargets": ["direction", "momentum"],
        "aggregateByUnderlying": _aggregate_options_by_underlying(aggregate_by_underlying),
        "cells": output_cells,
    }


def index_futures_for_symbol(symbol: str) -> list[str]:
    normalized_symbol = symbol.upper()
    if normalized_symbol in TECH_HEAVY_SYMBOLS:
        return ["NQ", "ES"]
    if normalized_symbol in SMALL_CAP_SYMBOLS:
        return ["RTY", "ES"]
    if normalized_symbol in DOW_SYMBOLS:
        return ["YM", "ES"]
    return ["ES"]


def index_futures_for_symbols(symbols: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        for future_symbol in index_futures_for_symbol(symbol):
            if future_symbol in seen:
                continue
            seen.add(future_symbol)
            ordered.append(future_symbol)
    return ordered


def _direction_features(bars: Sequence[Bar]) -> dict[str, object]:
    closes = [float(bar.close) for bar in bars]
    full_window_slope_pct = _full_window_slope_pct(closes)
    window_return_pct = closes[-1] / closes[0] - 1.0 if closes[0] else 0.0
    direction = _direction_from_slope(full_window_slope_pct)
    return {
        "firstTimestamp": bars[0].timestamp_utc,
        "lastTimestamp": bars[-1].timestamp_utc,
        "bars": len(bars),
        "windowReturnPct": window_return_pct,
        "linearSlopePct": full_window_slope_pct,
        "predictedDirection": direction,
        "curveSlopeSeverity": _slope_severity(full_window_slope_pct),
        "curveSlopeSeverityBaseline": DEFAULT_CURVE_SLOPE_SEVERITY,
    }


def _volatility_features(bars: Sequence[Bar]) -> dict[str, object]:
    returns = _returns(bars)
    if not returns:
        return {
            "returnStddev": 0.0,
            "realizedVolatilityPct": 0.0,
            "regime": "unknown",
        }
    stddev = pstdev(returns)
    realized = stddev * sqrt(len(returns))
    return {
        "returnStddev": stddev,
        "realizedVolatilityPct": realized,
        "regime": _volatility_regime(realized),
    }


def _volume_features(bars: Sequence[Bar]) -> dict[str, object] | None:
    volumes = [float(bar.volume) for bar in bars if bar.volume is not None]
    if len(volumes) < 2 or len(volumes) != len(bars):
        return None
    average_volume = mean(volumes)
    latest_volume = volumes[-1]
    relative_volume = latest_volume / average_volume if average_volume > 0 else 0.0
    direction = _direction_features(bars)
    price_volume_correlation = _correlation(_returns(bars), _deltas(volumes))
    volume_direction = "neutral"
    if relative_volume >= 1.1 and direction["predictedDirection"] in {"up", "down"}:
        volume_direction = str(direction["predictedDirection"])
    return {
        "latestVolume": latest_volume,
        "averageVolume": average_volume,
        "relativeVolume": relative_volume,
        "priceVolumeCorrelation": price_volume_correlation,
        "predictedDirection": volume_direction,
        "confidence": min(1.0, abs(price_volume_correlation) * min(relative_volume, 2.0)),
    }


def _aggregate_direction(features: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if not features:
        return {
            "predictedDirection": "unknown",
            "curveSlopeSeverity": DEFAULT_CURVE_SLOPE_SEVERITY,
        }
    primary_symbol, primary_features = next(iter(features.items()))
    return {
        "primarySymbol": primary_symbol,
        "predictedDirection": primary_features["predictedDirection"],
        "curveSlopeSeverity": primary_features["curveSlopeSeverity"],
        "averageWindowReturnPct": mean(
            float(value["windowReturnPct"]) for value in features.values()
        ),
    }


def _aggregate_volume(features: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if not features:
        return {"predictedDirection": "unknown", "confidence": 0.0}
    strongest_symbol, strongest = max(
        features.items(),
        key=lambda item: float(item[1].get("confidence") or 0.0),
    )
    return {
        "primarySymbol": strongest_symbol,
        "predictedDirection": strongest.get("predictedDirection", "unknown"),
        "confidence": strongest.get("confidence", 0.0),
    }


def _trade_volume_orderbook_fusion(
    volume_features: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    aggregate = _aggregate_volume(volume_features)
    has_volume = bool(volume_features)
    return {
        "status": "volume_only" if has_volume else "not_loaded",
        "volumeDirection": aggregate.get("predictedDirection", "unknown"),
        "fusionDirection": aggregate.get("predictedDirection", "unknown"),
        "confidence": aggregate.get("confidence", 0.0),
        "orderbookStatus": "awaiting_l2_orderbook_ingestion",
        "integrationBranch": ORDERBOOK_INTEGRATION_BRANCH,
        "requiredOrderbookFeatures": [
            "bid_ask_imbalance",
            "book_pressure",
            "depth_slope",
        ],
        "fusionMethod": (
            "use current volume as the first direction vote, then raise or lower "
            "confidence with L2 orderbook imbalance once orderbook events are wired"
        ),
    }


def _futures_instrument_alignment(
    instrument_features: Mapping[str, Mapping[str, object]],
    future_features: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not instrument_features or not future_features:
        return {"status": "not_loaded"}
    instrument = _aggregate_direction(instrument_features)
    futures = _aggregate_direction(future_features)
    instrument_direction = str(instrument.get("predictedDirection", "unknown"))
    futures_direction = str(futures.get("predictedDirection", "unknown"))
    directional_values = {"up", "down"}
    if instrument_direction in directional_values and futures_direction in directional_values:
        alignment = "aligned" if instrument_direction == futures_direction else "conflicting"
    else:
        alignment = "neutral_or_unknown"
    confidence = (
        min(
            float(instrument.get("curveSlopeSeverity", DEFAULT_CURVE_SLOPE_SEVERITY)),
            float(futures.get("curveSlopeSeverity", DEFAULT_CURVE_SLOPE_SEVERITY)),
        )
        / 5.0
    )
    return {
        "status": "active",
        "predictedInstrumentDirection": instrument_direction,
        "indexFuturesDirection": futures_direction,
        "alignment": alignment,
        "confidence": confidence,
    }


def _aggregate_options_by_underlying(
    aggregate_by_underlying: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, object]]:
    aggregates: dict[str, dict[str, object]] = {}
    for underlying, raw in sorted(aggregate_by_underlying.items()):
        call_weight = float(raw["callWeight"])
        put_weight = float(raw["putWeight"])
        total_weight = call_weight + put_weight
        up_probability = call_weight / total_weight if total_weight > 0 else 0.5
        down_probability = put_weight / total_weight if total_weight > 0 else 0.5
        aggregates[underlying] = {
            "upProbability": up_probability,
            "downProbability": down_probability,
            "momentumProbability": abs(up_probability - down_probability),
            "momentumDirection": _direction_from_probability(up_probability),
            "cells": int(raw["cells"]),
        }
    return aggregates


def _direction_from_slope(slope_pct: float) -> str:
    if slope_pct > 0.0025:
        return "up"
    if slope_pct < -0.0025:
        return "down"
    return "flat"


def _slope_severity(slope_pct: float) -> int:
    magnitude = abs(slope_pct)
    if magnitude < 0.0025:
        return 1
    if magnitude < 0.01:
        return 2
    if magnitude < 0.03:
        return 3
    if magnitude < 0.07:
        return 4
    return 5


def _volatility_regime(realized_volatility_pct: float) -> str:
    if realized_volatility_pct < 0.01:
        return "low"
    if realized_volatility_pct < 0.03:
        return "medium"
    return "high"


def _expiration_bucket(expiration_days: int) -> str:
    if expiration_days <= 7:
        return "0_7"
    if expiration_days <= 30:
        return "8_30"
    if expiration_days <= 90:
        return "31_90"
    return "91_plus"


def _moneyness_bucket(strike_moneyness: float) -> str:
    if strike_moneyness < 0.95:
        return "deep_itm_put_side"
    if strike_moneyness < 1.0:
        return "near_itm"
    if strike_moneyness <= 1.05:
        return "near_otm"
    return "far_otm_call_side"


def _time_bucket(trade_time_utc: str) -> str:
    return trade_time_utc[:13] + ":00:00+00:00" if len(trade_time_utc) >= 13 else trade_time_utc


def _direction_from_probability(up_probability: float) -> str:
    if up_probability > 0.55:
        return "up"
    if up_probability < 0.45:
        return "down"
    return "neutral"
