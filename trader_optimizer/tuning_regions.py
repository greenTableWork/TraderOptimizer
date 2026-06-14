from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trader_optimizer.config import write_json
from trader_optimizer.data import Bar
from trader_optimizer.market_features import (
    ORDERBOOK_INTEGRATION_BRANCH,
    OptionTrade,
    build_options_probability_map,
)
from trader_optimizer.series_math import (
    correlation as _correlation,
    deltas as _deltas,
    full_window_slope_pct as _full_window_slope_pct,
    returns_from_bars as _returns,
)
from trader_optimizer.slope_severity import (
    slope_severity_from_slope,
    slope_severity_thresholds_to_dict,
    thresholds_for_period,
)
from trader_optimizer.volatility_regime import (
    volatility_regime_from_realized,
    volatility_regime_thresholds_to_dict,
    volatility_thresholds_for_period,
)


Period = Literal["day", "week", "month"]
Direction = Literal["up", "down", "flat"]
VolatilityRegime = Literal["low", "medium", "high"]
FuturesAlignment = Literal["aligned", "conflicting", "neutral_or_unknown"]
MomentumRegime = Literal["low", "medium", "high"]
VolumeRegime = Literal["low", "normal", "high"]
VolumeDirection = Literal["up", "down", "neutral"]
DEFAULT_PERIODS: tuple[Period, ...] = ("day", "week", "month")
DEFAULT_DIRECTIONS: tuple[Direction, ...] = ("up", "down", "flat")
DEFAULT_VOLATILITY_REGIMES: tuple[VolatilityRegime, ...] = ("low", "medium", "high")
DEFAULT_FUTURES_ALIGNMENTS: tuple[FuturesAlignment, ...] = (
    "aligned",
    "conflicting",
    "neutral_or_unknown",
)
DEFAULT_MOMENTUM_REGIMES: tuple[MomentumRegime, ...] = ("low", "medium", "high")
DEFAULT_VOLUME_REGIMES: tuple[VolumeRegime, ...] = ("low", "normal", "high")
DEFAULT_VOLUME_DIRECTIONS: tuple[VolumeDirection, ...] = ("up", "down", "neutral")
DEFAULT_FLAT_THRESHOLD_PCT = 0.0025
DEFAULT_LOW_VOLATILITY_THRESHOLD_PCT = 0.01
DEFAULT_HIGH_VOLATILITY_THRESHOLD_PCT = 0.03
DEFAULT_LOW_MOMENTUM_THRESHOLD = 0.15
DEFAULT_HIGH_MOMENTUM_THRESHOLD = 0.45
DEFAULT_LOW_RELATIVE_VOLUME_THRESHOLD = 0.75
DEFAULT_HIGH_RELATIVE_VOLUME_THRESHOLD = 1.25
DEFAULT_VOLUME_DIRECTION_THRESHOLD = 1.1
DEFAULT_CURVE_SLOPE_SEVERITY = 3
DEFAULT_BUCKET_TIMEZONE = "UTC"
CSV_FIELD_ORDER = [
    "regionId",
    "symbol",
    "tuningSubcategory",
    "period",
    "bucket",
    "category",
    "barSize",
    "whatToShow",
    "useRth",
    "direction",
    "curveSlopeSeverity",
    "curveSlopeSeverityBaseline",
    "curveSlopeSeverityThresholds",
    "futuresSymbol",
    "futuresDirection",
    "futuresCurveSlopeSeverity",
    "futuresCurveSlopeSeverityThresholds",
    "futuresAlignment",
    "futuresReturnPct",
    "futuresLinearSlopePct",
    "individualVolatilityRegime",
    "individualVolatilityRegimeThresholds",
    "marketSymbol",
    "marketVolatilityRegime",
    "marketVolatilityRegimeThresholds",
    "volatilitySpreadPct",
    "optionsDirection",
    "optionsMomentumRegime",
    "optionsUpProbability",
    "optionsDownProbability",
    "optionsMomentumProbability",
    "optionsCellCount",
    "optionTradeCount",
    "optionCallPremium",
    "optionPutPremium",
    "optionCallVolume",
    "optionPutVolume",
    "volumeRegime",
    "volumeDirection",
    "relativeVolume",
    "periodAverageVolume",
    "averageBarVolume",
    "latestBarVolume",
    "priceVolumeCorrelation",
    "fusionDirection",
    "fusionConfidence",
    "orderbookStatus",
    "orderbookIntegrationBranch",
    "orderbookBidAskImbalance",
    "orderbookBookPressure",
    "orderbookDepthSlope",
    "startUtc",
    "endUtc",
    "bucketTimezone",
    "bucketStartUtc",
    "bucketEndUtc",
    "bucketStartLocal",
    "bucketEndLocal",
    "bars",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returnPct",
    "linearSlopePct",
    "returnStddev",
    "realizedVolatilityPct",
    "marketReturnPct",
    "marketReturnStddev",
    "marketRealizedVolatilityPct",
]
REGION_ID_VERSION = "tuning-region-v1"
REGION_ID_FIELDS = (
    "symbol",
    "tuningSubcategory",
    "period",
    "bucketTimezone",
    "bucket",
    "bucketStartUtc",
    "bucketEndUtc",
    "startUtc",
    "endUtc",
    "category",
    "barSize",
    "whatToShow",
    "useRth",
    "marketSymbol",
    "futuresSymbol",
)


@dataclass(frozen=True)
class DirectionRegion:
    symbol: str
    period: Period
    bucket: str
    category: str
    direction: str
    curve_slope_severity: int
    curve_slope_severity_baseline: int
    curve_slope_severity_thresholds: tuple[float, float, float, float] | None
    start_utc: str
    end_utc: str
    bucket_timezone: str
    bucket_start_utc: str
    bucket_end_utc: str
    bucket_start_local: str
    bucket_end_local: str
    bars: int
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    return_pct: float
    linear_slope_pct: float
    tuning_subcategory: str = "direction"
    bar_size: str | None = None
    what_to_show: str | None = None
    use_rth: int | None = None

    def to_dict(self) -> dict[str, object]:
        return _with_region_id({
            "symbol": self.symbol,
            "tuningSubcategory": self.tuning_subcategory,
            "period": self.period,
            "bucket": self.bucket,
            "category": self.category,
            "barSize": self.bar_size,
            "whatToShow": self.what_to_show,
            "useRth": self.use_rth,
            "direction": self.direction,
            "curveSlopeSeverity": self.curve_slope_severity,
            "curveSlopeSeverityBaseline": self.curve_slope_severity_baseline,
            "curveSlopeSeverityThresholds": (
                slope_severity_thresholds_to_dict(self.curve_slope_severity_thresholds)
                if self.curve_slope_severity_thresholds is not None
                else None
            ),
            "startUtc": self.start_utc,
            "endUtc": self.end_utc,
            "bucketTimezone": self.bucket_timezone,
            "bucketStartUtc": self.bucket_start_utc,
            "bucketEndUtc": self.bucket_end_utc,
            "bucketStartLocal": self.bucket_start_local,
            "bucketEndLocal": self.bucket_end_local,
            "bars": self.bars,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "returnPct": self.return_pct,
            "linearSlopePct": self.linear_slope_pct,
            "backtestRegion": {
                "startUtc": self.start_utc,
                "endUtc": self.end_utc,
                "category": self.category,
                "period": self.period,
            },
        })


@dataclass(frozen=True)
class VolatilityRegion:
    symbol: str
    period: Period
    bucket: str
    category: str
    individual_volatility_regime: str
    individual_volatility_regime_thresholds: tuple[float, float] | None
    market_symbol: str | None
    market_volatility_regime: str | None
    market_volatility_regime_thresholds: tuple[float, float] | None
    volatility_spread_pct: float | None
    start_utc: str
    end_utc: str
    bucket_timezone: str
    bucket_start_utc: str
    bucket_end_utc: str
    bucket_start_local: str
    bucket_end_local: str
    bars: int
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    return_pct: float
    return_stddev: float
    realized_volatility_pct: float
    market_return_pct: float | None = None
    market_return_stddev: float | None = None
    market_realized_volatility_pct: float | None = None
    tuning_subcategory: str = "volatility"
    bar_size: str | None = None
    what_to_show: str | None = None
    use_rth: int | None = None

    def to_dict(self) -> dict[str, object]:
        return _with_region_id({
            "symbol": self.symbol,
            "tuningSubcategory": self.tuning_subcategory,
            "period": self.period,
            "bucket": self.bucket,
            "category": self.category,
            "barSize": self.bar_size,
            "whatToShow": self.what_to_show,
            "useRth": self.use_rth,
            "individualVolatilityRegime": self.individual_volatility_regime,
            "individualVolatilityRegimeThresholds": (
                volatility_regime_thresholds_to_dict(
                    self.individual_volatility_regime_thresholds
                )
                if self.individual_volatility_regime_thresholds is not None
                else None
            ),
            "marketSymbol": self.market_symbol,
            "marketVolatilityRegime": self.market_volatility_regime,
            "marketVolatilityRegimeThresholds": (
                volatility_regime_thresholds_to_dict(
                    self.market_volatility_regime_thresholds
                )
                if self.market_volatility_regime_thresholds is not None
                else None
            ),
            "volatilitySpreadPct": self.volatility_spread_pct,
            "startUtc": self.start_utc,
            "endUtc": self.end_utc,
            "bucketTimezone": self.bucket_timezone,
            "bucketStartUtc": self.bucket_start_utc,
            "bucketEndUtc": self.bucket_end_utc,
            "bucketStartLocal": self.bucket_start_local,
            "bucketEndLocal": self.bucket_end_local,
            "bars": self.bars,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "returnPct": self.return_pct,
            "returnStddev": self.return_stddev,
            "realizedVolatilityPct": self.realized_volatility_pct,
            "marketReturnPct": self.market_return_pct,
            "marketReturnStddev": self.market_return_stddev,
            "marketRealizedVolatilityPct": self.market_realized_volatility_pct,
            "backtestRegion": {
                "startUtc": self.start_utc,
                "endUtc": self.end_utc,
                "category": self.category,
                "period": self.period,
            },
        })


@dataclass(frozen=True)
class IndexFuturesDirectionRegion:
    symbol: str
    futures_symbol: str
    period: Period
    bucket: str
    category: str
    direction: str
    futures_direction: str
    futures_alignment: str
    curve_slope_severity: int
    futures_curve_slope_severity: int
    curve_slope_severity_thresholds: tuple[float, float, float, float] | None
    futures_curve_slope_severity_thresholds: tuple[float, float, float, float] | None
    start_utc: str
    end_utc: str
    bucket_timezone: str
    bucket_start_utc: str
    bucket_end_utc: str
    bucket_start_local: str
    bucket_end_local: str
    bars: int
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    return_pct: float
    linear_slope_pct: float
    futures_return_pct: float
    futures_linear_slope_pct: float
    tuning_subcategory: str = "index_futures_direction"
    bar_size: str | None = None
    what_to_show: str | None = None
    use_rth: int | None = None

    def to_dict(self) -> dict[str, object]:
        return _with_region_id({
            "symbol": self.symbol,
            "tuningSubcategory": self.tuning_subcategory,
            "period": self.period,
            "bucket": self.bucket,
            "category": self.category,
            "barSize": self.bar_size,
            "whatToShow": self.what_to_show,
            "useRth": self.use_rth,
            "direction": self.direction,
            "curveSlopeSeverity": self.curve_slope_severity,
            "futuresSymbol": self.futures_symbol,
            "futuresDirection": self.futures_direction,
            "futuresCurveSlopeSeverity": self.futures_curve_slope_severity,
            "curveSlopeSeverityThresholds": (
                slope_severity_thresholds_to_dict(self.curve_slope_severity_thresholds)
                if self.curve_slope_severity_thresholds is not None
                else None
            ),
            "futuresCurveSlopeSeverityThresholds": (
                slope_severity_thresholds_to_dict(self.futures_curve_slope_severity_thresholds)
                if self.futures_curve_slope_severity_thresholds is not None
                else None
            ),
            "futuresAlignment": self.futures_alignment,
            "startUtc": self.start_utc,
            "endUtc": self.end_utc,
            "bucketTimezone": self.bucket_timezone,
            "bucketStartUtc": self.bucket_start_utc,
            "bucketEndUtc": self.bucket_end_utc,
            "bucketStartLocal": self.bucket_start_local,
            "bucketEndLocal": self.bucket_end_local,
            "bars": self.bars,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "returnPct": self.return_pct,
            "linearSlopePct": self.linear_slope_pct,
            "futuresReturnPct": self.futures_return_pct,
            "futuresLinearSlopePct": self.futures_linear_slope_pct,
            "backtestRegion": {
                "startUtc": self.start_utc,
                "endUtc": self.end_utc,
                "category": self.category,
                "period": self.period,
            },
        })


@dataclass(frozen=True)
class OptionsProbabilityRegion:
    symbol: str
    period: Period
    bucket: str
    category: str
    options_direction: str
    options_momentum_regime: str
    options_up_probability: float
    options_down_probability: float
    options_momentum_probability: float
    options_cell_count: int
    option_trade_count: int
    option_call_premium: float
    option_put_premium: float
    option_call_volume: float
    option_put_volume: float
    options_probability_map_3d: dict[str, object]
    start_utc: str
    end_utc: str
    bucket_timezone: str
    bucket_start_utc: str
    bucket_end_utc: str
    bucket_start_local: str
    bucket_end_local: str
    bars: int
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    return_pct: float
    tuning_subcategory: str = "options_probability_map_3d"
    bar_size: str | None = None
    what_to_show: str | None = None
    use_rth: int | None = None

    def to_dict(self) -> dict[str, object]:
        return _with_region_id({
            "symbol": self.symbol,
            "tuningSubcategory": self.tuning_subcategory,
            "period": self.period,
            "bucket": self.bucket,
            "category": self.category,
            "barSize": self.bar_size,
            "whatToShow": self.what_to_show,
            "useRth": self.use_rth,
            "optionsDirection": self.options_direction,
            "optionsMomentumRegime": self.options_momentum_regime,
            "optionsUpProbability": self.options_up_probability,
            "optionsDownProbability": self.options_down_probability,
            "optionsMomentumProbability": self.options_momentum_probability,
            "optionsCellCount": self.options_cell_count,
            "optionTradeCount": self.option_trade_count,
            "optionCallPremium": self.option_call_premium,
            "optionPutPremium": self.option_put_premium,
            "optionCallVolume": self.option_call_volume,
            "optionPutVolume": self.option_put_volume,
            "optionsProbabilityMap3d": self.options_probability_map_3d,
            "startUtc": self.start_utc,
            "endUtc": self.end_utc,
            "bucketTimezone": self.bucket_timezone,
            "bucketStartUtc": self.bucket_start_utc,
            "bucketEndUtc": self.bucket_end_utc,
            "bucketStartLocal": self.bucket_start_local,
            "bucketEndLocal": self.bucket_end_local,
            "bars": self.bars,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "returnPct": self.return_pct,
            "backtestRegion": {
                "startUtc": self.start_utc,
                "endUtc": self.end_utc,
                "category": self.category,
                "period": self.period,
            },
        })


@dataclass(frozen=True)
class VolumeOrderbookRegion:
    symbol: str
    period: Period
    bucket: str
    category: str
    direction: str
    curve_slope_severity: int
    curve_slope_severity_thresholds: tuple[float, float, float, float] | None
    volume_regime: str
    volume_direction: str
    relative_volume: float
    period_average_volume: float
    average_bar_volume: float
    latest_bar_volume: float
    price_volume_correlation: float
    fusion_direction: str
    fusion_confidence: float
    orderbook_status: str
    orderbook_integration_branch: str
    orderbook_bid_ask_imbalance: float | None
    orderbook_book_pressure: float | None
    orderbook_depth_slope: float | None
    start_utc: str
    end_utc: str
    bucket_timezone: str
    bucket_start_utc: str
    bucket_end_utc: str
    bucket_start_local: str
    bucket_end_local: str
    bars: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    return_pct: float
    linear_slope_pct: float
    tuning_subcategory: str = "trade_volume_orderbook"
    bar_size: str | None = None
    what_to_show: str | None = None
    use_rth: int | None = None

    def to_dict(self) -> dict[str, object]:
        return _with_region_id({
            "symbol": self.symbol,
            "tuningSubcategory": self.tuning_subcategory,
            "period": self.period,
            "bucket": self.bucket,
            "category": self.category,
            "barSize": self.bar_size,
            "whatToShow": self.what_to_show,
            "useRth": self.use_rth,
            "direction": self.direction,
            "curveSlopeSeverity": self.curve_slope_severity,
            "curveSlopeSeverityThresholds": (
                slope_severity_thresholds_to_dict(self.curve_slope_severity_thresholds)
                if self.curve_slope_severity_thresholds is not None
                else None
            ),
            "volumeRegime": self.volume_regime,
            "volumeDirection": self.volume_direction,
            "relativeVolume": self.relative_volume,
            "periodAverageVolume": self.period_average_volume,
            "averageBarVolume": self.average_bar_volume,
            "latestBarVolume": self.latest_bar_volume,
            "priceVolumeCorrelation": self.price_volume_correlation,
            "fusionDirection": self.fusion_direction,
            "fusionConfidence": self.fusion_confidence,
            "orderbookStatus": self.orderbook_status,
            "orderbookIntegrationBranch": self.orderbook_integration_branch,
            "orderbookBidAskImbalance": self.orderbook_bid_ask_imbalance,
            "orderbookBookPressure": self.orderbook_book_pressure,
            "orderbookDepthSlope": self.orderbook_depth_slope,
            "startUtc": self.start_utc,
            "endUtc": self.end_utc,
            "bucketTimezone": self.bucket_timezone,
            "bucketStartUtc": self.bucket_start_utc,
            "bucketEndUtc": self.bucket_end_utc,
            "bucketStartLocal": self.bucket_start_local,
            "bucketEndLocal": self.bucket_end_local,
            "bars": self.bars,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "returnPct": self.return_pct,
            "linearSlopePct": self.linear_slope_pct,
            "tradeVolumeOrderbook": {
                "status": "volume_only",
                "volumeDirection": self.volume_direction,
                "fusionDirection": self.fusion_direction,
                "confidence": self.fusion_confidence,
                "orderbookStatus": self.orderbook_status,
                "integrationBranch": self.orderbook_integration_branch,
                "requiredOrderbookFeatures": [
                    "bid_ask_imbalance",
                    "book_pressure",
                    "depth_slope",
                ],
            },
            "backtestRegion": {
                "startUtc": self.start_utc,
                "endUtc": self.end_utc,
                "category": self.category,
                "period": self.period,
            },
        })


TuningRegion = (
    DirectionRegion
    | VolatilityRegion
    | IndexFuturesDirectionRegion
    | OptionsProbabilityRegion
    | VolumeOrderbookRegion
)


def region_id_for_row(row: Mapping[str, object]) -> str:
    identity = {
        field: _normalize_region_id_value(row.get(field))
        for field in REGION_ID_FIELDS
        if row.get(field) not in (None, "")
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{REGION_ID_VERSION}:{digest}"


def region_id_for_region(region: TuningRegion) -> str:
    return str(region.to_dict()["regionId"])


def filter_duplicate_regions(
    regions: Sequence[TuningRegion],
    *,
    existing_region_ids: Iterable[str] = (),
) -> tuple[list[TuningRegion], list[str]]:
    seen = set(existing_region_ids)
    output: list[TuningRegion] = []
    skipped_region_ids: list[str] = []
    for region in regions:
        region_id = region_id_for_region(region)
        if region_id in seen:
            skipped_region_ids.append(region_id)
            continue
        seen.add(region_id)
        output.append(region)
    return output, skipped_region_ids


def load_region_ids(paths: Iterable[Path]) -> set[str]:
    region_ids: set[str] = set()
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        if path.suffix.lower() == ".csv":
            region_ids.update(_load_csv_region_ids(path))
        else:
            region_ids.update(_load_jsonl_region_ids(path))
    return region_ids


def categorize_direction_regions(
    symbol: str,
    bars: Sequence[Bar],
    *,
    periods: Sequence[Period] = DEFAULT_PERIODS,
    directions: Sequence[Direction] = DEFAULT_DIRECTIONS,
    min_bars: int = 2,
    flat_threshold_pct: float = DEFAULT_FLAT_THRESHOLD_PCT,
    slope_severity_thresholds: (
        Mapping[str, Sequence[float] | Mapping[str, float]]
        | Sequence[float]
        | Mapping[str, float]
        | None
    ) = None,
    bucket_timezone: str = DEFAULT_BUCKET_TIMEZONE,
) -> list[DirectionRegion]:
    bucket_zone = _bucket_zone(bucket_timezone)
    allowed_directions = set(directions)
    sorted_bars = sorted(bars, key=lambda bar: _parse_utc(bar.timestamp_utc))
    regions: list[DirectionRegion] = []
    for period in periods:
        buckets: dict[str, list[Bar]] = {}
        bucket_bounds: dict[str, tuple[datetime, datetime, datetime, datetime]] = {}
        for bar in sorted_bars:
            timestamp = _parse_utc(bar.timestamp_utc)
            bucket, bucket_start_local, bucket_end_local = _period_bucket(
                timestamp,
                period,
                bucket_zone,
            )
            buckets.setdefault(bucket, []).append(bar)
            bucket_bounds[bucket] = (
                bucket_start_local,
                bucket_end_local,
                bucket_start_local.astimezone(UTC),
                bucket_end_local.astimezone(UTC),
            )

        for bucket, bucket_bars in sorted(buckets.items()):
            if len(bucket_bars) < min_bars:
                continue
            (
                bucket_start_local,
                bucket_end_local,
                bucket_start_utc,
                bucket_end_utc,
            ) = bucket_bounds[bucket]
            region = _direction_region(
                symbol,
                period,
                bucket,
                bucket_bars,
                bucket_timezone=bucket_timezone,
                bucket_start_utc=bucket_start_utc,
                bucket_end_utc=bucket_end_utc,
                bucket_start_local=bucket_start_local,
                bucket_end_local=bucket_end_local,
                flat_threshold_pct=flat_threshold_pct,
                slope_severity_thresholds=thresholds_for_period(
                    slope_severity_thresholds,
                    period,
                ),
            )
            if region.direction in allowed_directions:
                regions.append(region)
    return regions


def categorize_volatility_regions(
    symbol: str,
    bars: Sequence[Bar],
    *,
    market_symbol: str | None = None,
    market_bars: Sequence[Bar] | None = None,
    periods: Sequence[Period] = DEFAULT_PERIODS,
    regimes: Sequence[VolatilityRegime] = DEFAULT_VOLATILITY_REGIMES,
    min_bars: int = 2,
    low_threshold_pct: float = DEFAULT_LOW_VOLATILITY_THRESHOLD_PCT,
    high_threshold_pct: float = DEFAULT_HIGH_VOLATILITY_THRESHOLD_PCT,
    volatility_regime_thresholds: (
        Mapping[str, Sequence[float] | Mapping[str, float]]
        | Sequence[float]
        | Mapping[str, float]
        | None
    ) = None,
    market_volatility_regime_thresholds: (
        Mapping[str, Sequence[float] | Mapping[str, float]]
        | Sequence[float]
        | Mapping[str, float]
        | None
    ) = None,
    bucket_timezone: str = DEFAULT_BUCKET_TIMEZONE,
) -> list[VolatilityRegion]:
    bucket_zone = _bucket_zone(bucket_timezone)
    allowed_regimes = set(regimes)
    sorted_bars = sorted(bars, key=lambda bar: _parse_utc(bar.timestamp_utc))
    sorted_market_bars = (
        sorted(market_bars, key=lambda bar: _parse_utc(bar.timestamp_utc))
        if market_bars
        else None
    )
    regions: list[VolatilityRegion] = []
    for period in periods:
        buckets, bucket_bounds = _bucket_bars(sorted_bars, period, bucket_zone)
        market_buckets = (
            _bucket_bars(sorted_market_bars, period, bucket_zone)[0]
            if sorted_market_bars is not None
            else {}
        )
        for bucket, bucket_bars in sorted(buckets.items()):
            if len(bucket_bars) < min_bars:
                continue
            market_bucket_bars = market_buckets.get(bucket)
            if market_bucket_bars is not None and len(market_bucket_bars) < min_bars:
                market_bucket_bars = None
            (
                bucket_start_local,
                bucket_end_local,
                bucket_start_utc,
                bucket_end_utc,
            ) = bucket_bounds[bucket]
            region = _volatility_region(
                symbol,
                period,
                bucket,
                bucket_bars,
                bucket_timezone=bucket_timezone,
                bucket_start_utc=bucket_start_utc,
                bucket_end_utc=bucket_end_utc,
                bucket_start_local=bucket_start_local,
                bucket_end_local=bucket_end_local,
                market_symbol=market_symbol,
                market_bars=market_bucket_bars,
                low_threshold_pct=low_threshold_pct,
                high_threshold_pct=high_threshold_pct,
                volatility_regime_thresholds=volatility_thresholds_for_period(
                    volatility_regime_thresholds,
                    period,
                ),
                market_volatility_regime_thresholds=volatility_thresholds_for_period(
                    market_volatility_regime_thresholds,
                    period,
                ),
            )
            if region.individual_volatility_regime in allowed_regimes:
                regions.append(region)
    return regions


def categorize_index_futures_direction_regions(
    symbol: str,
    bars: Sequence[Bar],
    *,
    futures_symbol: str,
    futures_bars: Sequence[Bar],
    periods: Sequence[Period] = DEFAULT_PERIODS,
    alignments: Sequence[FuturesAlignment] = DEFAULT_FUTURES_ALIGNMENTS,
    min_bars: int = 2,
    flat_threshold_pct: float = DEFAULT_FLAT_THRESHOLD_PCT,
    slope_severity_thresholds: (
        Mapping[str, Sequence[float] | Mapping[str, float]]
        | Sequence[float]
        | Mapping[str, float]
        | None
    ) = None,
    futures_slope_severity_thresholds: (
        Mapping[str, Sequence[float] | Mapping[str, float]]
        | Sequence[float]
        | Mapping[str, float]
        | None
    ) = None,
    bucket_timezone: str = DEFAULT_BUCKET_TIMEZONE,
) -> list[IndexFuturesDirectionRegion]:
    bucket_zone = _bucket_zone(bucket_timezone)
    allowed_alignments = set(alignments)
    sorted_bars = sorted(bars, key=lambda bar: _parse_utc(bar.timestamp_utc))
    sorted_futures_bars = sorted(
        futures_bars,
        key=lambda bar: _parse_utc(bar.timestamp_utc),
    )
    regions: list[IndexFuturesDirectionRegion] = []
    for period in periods:
        buckets, bucket_bounds = _bucket_bars(sorted_bars, period, bucket_zone)
        futures_buckets, _ = _bucket_bars(sorted_futures_bars, period, bucket_zone)
        for bucket, bucket_bars in sorted(buckets.items()):
            if len(bucket_bars) < min_bars:
                continue
            futures_bucket_bars = futures_buckets.get(bucket)
            if futures_bucket_bars is None or len(futures_bucket_bars) < min_bars:
                continue
            (
                bucket_start_local,
                bucket_end_local,
                bucket_start_utc,
                bucket_end_utc,
            ) = bucket_bounds[bucket]
            region = _index_futures_direction_region(
                symbol,
                futures_symbol,
                period,
                bucket,
                bucket_bars,
                futures_bucket_bars,
                bucket_timezone=bucket_timezone,
                bucket_start_utc=bucket_start_utc,
                bucket_end_utc=bucket_end_utc,
                bucket_start_local=bucket_start_local,
                bucket_end_local=bucket_end_local,
                flat_threshold_pct=flat_threshold_pct,
                slope_severity_thresholds=thresholds_for_period(
                    slope_severity_thresholds,
                    period,
                ),
                futures_slope_severity_thresholds=thresholds_for_period(
                    futures_slope_severity_thresholds,
                    period,
                ),
            )
            if region.futures_alignment in allowed_alignments:
                regions.append(region)
    return regions


def categorize_options_probability_regions(
    symbol: str,
    bars: Sequence[Bar],
    *,
    option_trades: Sequence[OptionTrade],
    periods: Sequence[Period] = DEFAULT_PERIODS,
    momentum_regimes: Sequence[MomentumRegime] = DEFAULT_MOMENTUM_REGIMES,
    min_bars: int = 2,
    min_option_trades: int = 1,
    low_momentum_threshold: float = DEFAULT_LOW_MOMENTUM_THRESHOLD,
    high_momentum_threshold: float = DEFAULT_HIGH_MOMENTUM_THRESHOLD,
    bucket_timezone: str = DEFAULT_BUCKET_TIMEZONE,
) -> list[OptionsProbabilityRegion]:
    bucket_zone = _bucket_zone(bucket_timezone)
    allowed_regimes = set(momentum_regimes)
    normalized_symbol = symbol.upper()
    sorted_bars = sorted(bars, key=lambda bar: _parse_utc(bar.timestamp_utc))
    sorted_option_trades = [
        trade
        for trade in sorted(
            option_trades,
            key=lambda trade: _parse_utc(trade.trade_time_utc),
        )
        if trade.underlying.upper() == normalized_symbol
    ]
    regions: list[OptionsProbabilityRegion] = []
    for period in periods:
        buckets, bucket_bounds = _bucket_bars(sorted_bars, period, bucket_zone)
        for bucket, bucket_bars in sorted(buckets.items()):
            if len(bucket_bars) < min_bars:
                continue
            (
                bucket_start_local,
                bucket_end_local,
                bucket_start_utc,
                bucket_end_utc,
            ) = bucket_bounds[bucket]
            bucket_trades = [
                trade
                for trade in sorted_option_trades
                if bucket_start_utc <= _parse_utc(trade.trade_time_utc) < bucket_end_utc
            ]
            if len(bucket_trades) < min_option_trades:
                continue
            region = _options_probability_region(
                symbol,
                period,
                bucket,
                bucket_bars,
                bucket_trades,
                bucket_timezone=bucket_timezone,
                bucket_start_utc=bucket_start_utc,
                bucket_end_utc=bucket_end_utc,
                bucket_start_local=bucket_start_local,
                bucket_end_local=bucket_end_local,
                low_momentum_threshold=low_momentum_threshold,
                high_momentum_threshold=high_momentum_threshold,
            )
            if region.options_momentum_regime in allowed_regimes:
                regions.append(region)
    return regions


def categorize_volume_orderbook_regions(
    symbol: str,
    bars: Sequence[Bar],
    *,
    periods: Sequence[Period] = DEFAULT_PERIODS,
    volume_regimes: Sequence[VolumeRegime] = DEFAULT_VOLUME_REGIMES,
    volume_directions: Sequence[VolumeDirection] = DEFAULT_VOLUME_DIRECTIONS,
    min_bars: int = 2,
    low_relative_volume_threshold: float = DEFAULT_LOW_RELATIVE_VOLUME_THRESHOLD,
    high_relative_volume_threshold: float = DEFAULT_HIGH_RELATIVE_VOLUME_THRESHOLD,
    volume_direction_threshold: float = DEFAULT_VOLUME_DIRECTION_THRESHOLD,
    flat_threshold_pct: float = DEFAULT_FLAT_THRESHOLD_PCT,
    slope_severity_thresholds: (
        Mapping[str, Sequence[float] | Mapping[str, float]]
        | Sequence[float]
        | Mapping[str, float]
        | None
    ) = None,
    bucket_timezone: str = DEFAULT_BUCKET_TIMEZONE,
) -> list[VolumeOrderbookRegion]:
    bucket_zone = _bucket_zone(bucket_timezone)
    allowed_regimes = set(volume_regimes)
    allowed_directions = set(volume_directions)
    sorted_bars = sorted(bars, key=lambda bar: _parse_utc(bar.timestamp_utc))
    regions: list[VolumeOrderbookRegion] = []
    for period in periods:
        buckets, bucket_bounds = _bucket_bars(sorted_bars, period, bucket_zone)
        bucket_volume_totals: dict[str, float] = {}
        for bucket, bucket_bars in buckets.items():
            if len(bucket_bars) < min_bars:
                continue
            volume_total = _volume_total(bucket_bars)
            if volume_total is None:
                continue
            bucket_volume_totals[bucket] = volume_total
        if not bucket_volume_totals:
            continue
        period_average_volume = mean(bucket_volume_totals.values())
        if period_average_volume <= 0:
            continue

        for bucket, bucket_bars in sorted(buckets.items()):
            volume_total = bucket_volume_totals.get(bucket)
            if volume_total is None:
                continue
            (
                bucket_start_local,
                bucket_end_local,
                bucket_start_utc,
                bucket_end_utc,
            ) = bucket_bounds[bucket]
            region = _volume_orderbook_region(
                symbol,
                period,
                bucket,
                bucket_bars,
                period_average_volume=period_average_volume,
                bucket_timezone=bucket_timezone,
                bucket_start_utc=bucket_start_utc,
                bucket_end_utc=bucket_end_utc,
                bucket_start_local=bucket_start_local,
                bucket_end_local=bucket_end_local,
                low_relative_volume_threshold=low_relative_volume_threshold,
                high_relative_volume_threshold=high_relative_volume_threshold,
                volume_direction_threshold=volume_direction_threshold,
                flat_threshold_pct=flat_threshold_pct,
                slope_severity_thresholds=thresholds_for_period(
                    slope_severity_thresholds,
                    period,
                ),
            )
            if (
                region.volume_regime in allowed_regimes
                and region.volume_direction in allowed_directions
            ):
                regions.append(region)
    return regions


def direction_region_summary(
    regions: Sequence[DirectionRegion],
) -> dict[str, object]:
    return region_summary(regions)


def region_summary(
    regions: Sequence[TuningRegion],
    *,
    tuning_subcategory: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    region_rows = [region.to_dict() for region in regions]
    by_region_id = Counter(
        str(row["regionId"]) for row in region_rows if row.get("regionId")
    )
    duplicate_region_ids = {
        region_id: count
        for region_id, count in sorted(by_region_id.items())
        if count > 1
    }
    by_period = Counter(region.period for region in regions)
    by_category = Counter(region.category for region in regions)
    by_period_category = Counter(
        f"{region.period}:{region.category}" for region in regions
    )
    by_subcategory = Counter(region.tuning_subcategory for region in regions)
    by_bar_size = Counter(
        str(row["barSize"]) for row in region_rows if row.get("barSize")
    )
    by_what_to_show = Counter(
        str(row["whatToShow"]) for row in region_rows if row.get("whatToShow")
    )
    by_use_rth = Counter(
        str(row["useRth"]) for row in region_rows if row.get("useRth") is not None
    )
    by_direction = Counter(
        region.direction
        for region in regions
        if isinstance(
            region,
            (DirectionRegion, IndexFuturesDirectionRegion, VolumeOrderbookRegion),
        )
    )
    by_futures_direction = Counter(
        region.futures_direction
        for region in regions
        if isinstance(region, IndexFuturesDirectionRegion)
    )
    by_futures_alignment = Counter(
        region.futures_alignment
        for region in regions
        if isinstance(region, IndexFuturesDirectionRegion)
    )
    by_futures_symbol = Counter(
        region.futures_symbol
        for region in regions
        if isinstance(region, IndexFuturesDirectionRegion)
    )
    by_options_direction = Counter(
        region.options_direction
        for region in regions
        if isinstance(region, OptionsProbabilityRegion)
    )
    by_options_momentum_regime = Counter(
        region.options_momentum_regime
        for region in regions
        if isinstance(region, OptionsProbabilityRegion)
    )
    by_volume_regime = Counter(
        region.volume_regime
        for region in regions
        if isinstance(region, VolumeOrderbookRegion)
    )
    by_volume_direction = Counter(
        region.volume_direction
        for region in regions
        if isinstance(region, VolumeOrderbookRegion)
    )
    by_fusion_direction = Counter(
        region.fusion_direction
        for region in regions
        if isinstance(region, VolumeOrderbookRegion)
    )
    by_orderbook_status = Counter(
        region.orderbook_status
        for region in regions
        if isinstance(region, VolumeOrderbookRegion)
    )
    by_volatility_regime = Counter(
        region.individual_volatility_regime
        for region in regions
        if isinstance(region, VolatilityRegion)
    )
    by_market_volatility_regime = Counter(
        region.market_volatility_regime
        for region in regions
        if isinstance(region, VolatilityRegion) and region.market_volatility_regime
    )
    by_timezone = Counter(region.bucket_timezone for region in regions)
    symbols = sorted({region.symbol for region in regions})
    if not by_subcategory and tuning_subcategory:
        by_subcategory[tuning_subcategory] = 0
    schema = "tuning_regions.v1"
    if set(by_subcategory) == {"direction"}:
        schema = "tuning_direction_regions.v1"
    elif set(by_subcategory) == {"volatility"}:
        schema = "tuning_volatility_regions.v1"
    elif set(by_subcategory) == {"index_futures_direction"}:
        schema = "tuning_index_futures_direction_regions.v1"
    elif set(by_subcategory) == {"options_probability_map_3d"}:
        schema = "tuning_options_probability_regions.v1"
    elif set(by_subcategory) == {"trade_volume_orderbook"}:
        schema = "tuning_trade_volume_orderbook_regions.v1"
    summary = {
        "schema": schema,
        "tuningSubcategory": next(iter(by_subcategory)) if len(by_subcategory) == 1 else None,
        "tuningSubcategories": dict(sorted(by_subcategory.items())),
        "symbols": symbols,
        "regions": len(regions),
        "uniqueRegionIds": len(by_region_id),
        "duplicateRegionCount": sum(count - 1 for count in duplicate_region_ids.values()),
        "duplicateRegionIds": duplicate_region_ids,
        "barSizes": dict(sorted(by_bar_size.items())),
        "whatToShow": dict(sorted(by_what_to_show.items())),
        "useRth": dict(sorted(by_use_rth.items())),
        "periods": dict(sorted(by_period.items())),
        "directions": dict(sorted(by_direction.items())),
        "futuresDirections": dict(sorted(by_futures_direction.items())),
        "futuresAlignments": dict(sorted(by_futures_alignment.items())),
        "futuresSymbols": dict(sorted(by_futures_symbol.items())),
        "optionsDirections": dict(sorted(by_options_direction.items())),
        "optionsMomentumRegimes": dict(sorted(by_options_momentum_regime.items())),
        "volumeRegimes": dict(sorted(by_volume_regime.items())),
        "volumeDirections": dict(sorted(by_volume_direction.items())),
        "fusionDirections": dict(sorted(by_fusion_direction.items())),
        "orderbookStatuses": dict(sorted(by_orderbook_status.items())),
        "individualVolatilityRegimes": dict(sorted(by_volatility_regime.items())),
        "marketVolatilityRegimes": dict(sorted(by_market_volatility_regime.items())),
        "categories": dict(sorted(by_category.items())),
        "periodCategories": dict(sorted(by_period_category.items())),
        "bucketTimezones": dict(sorted(by_timezone.items())),
    }
    if metadata:
        summary.update(metadata)
    return summary


def write_regions_jsonl(path: Path, regions: Sequence[TuningRegion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for region in regions:
            handle.write(json.dumps(region.to_dict(), sort_keys=True))
            handle.write("\n")


def write_regions_csv(path: Path, regions: Sequence[TuningRegion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    region_rows = [region.to_dict() for region in regions]
    extra_fields = sorted(
        {
            key
            for row in region_rows
            for key in row
            if key not in CSV_FIELD_ORDER and key != "backtestRegion"
        }
    )
    fields = [
        field
        for field in (*CSV_FIELD_ORDER, *extra_fields)
        if any(field in row for row in region_rows)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in region_rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_region_summary(
    path: Path,
    regions: Sequence[TuningRegion],
    *,
    tuning_subcategory: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    write_json(
        path,
        region_summary(
            regions,
            tuning_subcategory=tuning_subcategory,
            metadata=metadata,
        ),
    )


def _with_region_id(row: dict[str, object]) -> dict[str, object]:
    output = dict(row)
    output["regionId"] = region_id_for_row(output)
    return output


def _normalize_region_id_value(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def _load_jsonl_region_ids(path: Path) -> set[str]:
    region_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse region JSONL {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                continue
            region_id = row.get("regionId")
            if region_id:
                region_ids.add(str(region_id))
            else:
                region_ids.add(region_id_for_row(row))
    return region_ids


def _load_csv_region_ids(path: Path) -> set[str]:
    region_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            region_id = row.get("regionId")
            if region_id:
                region_ids.add(region_id)
            else:
                region_ids.add(region_id_for_row(row))
    return region_ids


def normalize_periods(raw_periods: Iterable[str] | None) -> tuple[Period, ...]:
    if not raw_periods:
        return DEFAULT_PERIODS
    periods: list[Period] = []
    seen: set[str] = set()
    for raw_period in raw_periods:
        period = raw_period.lower()
        if period not in {"day", "week", "month"}:
            raise ValueError(f"Unsupported period {raw_period!r}")
        if period not in seen:
            seen.add(period)
            periods.append(period)  # type: ignore[arg-type]
    return tuple(periods)


def normalize_directions(raw_directions: Iterable[str] | None) -> tuple[Direction, ...]:
    return cast(
        tuple[Direction, ...],
        _normalize_literal_values(
            raw_directions,
            default=DEFAULT_DIRECTIONS,
            supported={"up", "down", "flat"},
            label="direction",
        ),
    )


def normalize_volatility_regimes(
    raw_regimes: Iterable[str] | None,
) -> tuple[VolatilityRegime, ...]:
    return cast(
        tuple[VolatilityRegime, ...],
        _normalize_literal_values(
            raw_regimes,
            default=DEFAULT_VOLATILITY_REGIMES,
            supported={"low", "medium", "high"},
            label="volatility regime",
        ),
    )


def normalize_futures_alignments(
    raw_alignments: Iterable[str] | None,
) -> tuple[FuturesAlignment, ...]:
    return cast(
        tuple[FuturesAlignment, ...],
        _normalize_literal_values(
            raw_alignments,
            default=DEFAULT_FUTURES_ALIGNMENTS,
            supported={"aligned", "conflicting", "neutral_or_unknown"},
            label="futures alignment",
        ),
    )


def normalize_momentum_regimes(
    raw_regimes: Iterable[str] | None,
) -> tuple[MomentumRegime, ...]:
    return cast(
        tuple[MomentumRegime, ...],
        _normalize_literal_values(
            raw_regimes,
            default=DEFAULT_MOMENTUM_REGIMES,
            supported={"low", "medium", "high"},
            label="momentum regime",
        ),
    )


def normalize_volume_regimes(
    raw_regimes: Iterable[str] | None,
) -> tuple[VolumeRegime, ...]:
    return cast(
        tuple[VolumeRegime, ...],
        _normalize_literal_values(
            raw_regimes,
            default=DEFAULT_VOLUME_REGIMES,
            supported={"low", "normal", "high"},
            label="volume regime",
        ),
    )


def normalize_volume_directions(
    raw_directions: Iterable[str] | None,
) -> tuple[VolumeDirection, ...]:
    return cast(
        tuple[VolumeDirection, ...],
        _normalize_literal_values(
            raw_directions,
            default=DEFAULT_VOLUME_DIRECTIONS,
            supported={"up", "down", "neutral"},
            label="volume direction",
        ),
    )


def _normalize_literal_values(
    raw_values: Iterable[str] | None,
    *,
    default: tuple[str, ...],
    supported: set[str],
    label: str,
) -> tuple[str, ...]:
    if not raw_values:
        return default
    values: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        value = raw_value.lower()
        if value not in supported:
            raise ValueError(f"Unsupported {label} {raw_value!r}")
        if value not in seen:
            seen.add(value)
            values.append(value)
    return tuple(values)


def _direction_region(
    symbol: str,
    period: Period,
    bucket: str,
    bars: Sequence[Bar],
    bucket_timezone: str,
    bucket_start_utc: datetime,
    bucket_end_utc: datetime,
    bucket_start_local: datetime,
    bucket_end_local: datetime,
    *,
    flat_threshold_pct: float,
    slope_severity_thresholds: tuple[float, float, float, float] | None,
) -> DirectionRegion:
    open_price = float(bars[0].open)
    close_price = float(bars[-1].close)
    high = max(float(bar.high) for bar in bars)
    low = min(float(bar.low) for bar in bars)
    volume_values = [float(bar.volume) for bar in bars if bar.volume is not None]
    volume = sum(volume_values) if len(volume_values) == len(bars) else None
    linear_slope_pct = _full_window_slope_pct([float(bar.close) for bar in bars])
    return_pct = close_price / open_price - 1.0 if open_price else 0.0
    direction = _direction_from_slope(linear_slope_pct, flat_threshold_pct)
    severity = _slope_severity(linear_slope_pct, slope_severity_thresholds)
    return DirectionRegion(
        symbol=symbol,
        period=period,
        bucket=bucket,
        category=f"{direction}_slope_{severity}",
        direction=direction,
        curve_slope_severity=severity,
        curve_slope_severity_baseline=DEFAULT_CURVE_SLOPE_SEVERITY,
        curve_slope_severity_thresholds=slope_severity_thresholds,
        start_utc=bars[0].timestamp_utc,
        end_utc=bars[-1].timestamp_utc,
        bucket_timezone=bucket_timezone,
        bucket_start_utc=_format_utc(bucket_start_utc),
        bucket_end_utc=_format_utc(bucket_end_utc),
        bucket_start_local=_format_local(bucket_start_local),
        bucket_end_local=_format_local(bucket_end_local),
        bars=len(bars),
        open=open_price,
        high=high,
        low=low,
        close=close_price,
        volume=volume,
        return_pct=return_pct,
        linear_slope_pct=linear_slope_pct,
    )


def _volatility_region(
    symbol: str,
    period: Period,
    bucket: str,
    bars: Sequence[Bar],
    bucket_timezone: str,
    bucket_start_utc: datetime,
    bucket_end_utc: datetime,
    bucket_start_local: datetime,
    bucket_end_local: datetime,
    *,
    market_symbol: str | None,
    market_bars: Sequence[Bar] | None,
    low_threshold_pct: float,
    high_threshold_pct: float,
    volatility_regime_thresholds: tuple[float, float] | None,
    market_volatility_regime_thresholds: tuple[float, float] | None,
) -> VolatilityRegion:
    open_price = float(bars[0].open)
    close_price = float(bars[-1].close)
    returns = _returns(bars)
    return_stddev = pstdev(returns) if returns else 0.0
    realized_volatility = return_stddev * sqrt(len(returns)) if returns else 0.0
    effective_volatility_thresholds = volatility_regime_thresholds or (
        low_threshold_pct,
        high_threshold_pct,
    )
    individual_regime = _volatility_regime(
        realized_volatility,
        effective_volatility_thresholds,
    )
    market_return_pct = None
    market_return_stddev = None
    market_realized_volatility = None
    market_regime = None
    volatility_spread = None
    effective_market_thresholds = market_volatility_regime_thresholds or (
        low_threshold_pct,
        high_threshold_pct,
    )
    if market_bars:
        market_returns = _returns(market_bars)
        market_return_stddev = pstdev(market_returns) if market_returns else 0.0
        market_realized_volatility = (
            market_return_stddev * sqrt(len(market_returns)) if market_returns else 0.0
        )
        market_regime = _volatility_regime(
            market_realized_volatility,
            effective_market_thresholds,
        )
        market_return_pct = (
            float(market_bars[-1].close) / float(market_bars[0].open) - 1.0
            if float(market_bars[0].open)
            else 0.0
        )
        volatility_spread = realized_volatility - market_realized_volatility
    category = (
        f"individual_{individual_regime}_market_{market_regime}_vol"
        if market_regime
        else f"individual_{individual_regime}_vol"
    )
    volume_values = [float(bar.volume) for bar in bars if bar.volume is not None]
    volume = sum(volume_values) if len(volume_values) == len(bars) else None
    return VolatilityRegion(
        symbol=symbol,
        period=period,
        bucket=bucket,
        category=category,
        individual_volatility_regime=individual_regime,
        individual_volatility_regime_thresholds=volatility_regime_thresholds,
        market_symbol=market_symbol if market_regime else None,
        market_volatility_regime=market_regime,
        market_volatility_regime_thresholds=(
            market_volatility_regime_thresholds if market_regime else None
        ),
        volatility_spread_pct=volatility_spread,
        start_utc=bars[0].timestamp_utc,
        end_utc=bars[-1].timestamp_utc,
        bucket_timezone=bucket_timezone,
        bucket_start_utc=_format_utc(bucket_start_utc),
        bucket_end_utc=_format_utc(bucket_end_utc),
        bucket_start_local=_format_local(bucket_start_local),
        bucket_end_local=_format_local(bucket_end_local),
        bars=len(bars),
        open=open_price,
        high=max(float(bar.high) for bar in bars),
        low=min(float(bar.low) for bar in bars),
        close=close_price,
        volume=volume,
        return_pct=close_price / open_price - 1.0 if open_price else 0.0,
        return_stddev=return_stddev,
        realized_volatility_pct=realized_volatility,
        market_return_pct=market_return_pct,
        market_return_stddev=market_return_stddev,
        market_realized_volatility_pct=market_realized_volatility,
    )


def _index_futures_direction_region(
    symbol: str,
    futures_symbol: str,
    period: Period,
    bucket: str,
    bars: Sequence[Bar],
    futures_bars: Sequence[Bar],
    bucket_timezone: str,
    bucket_start_utc: datetime,
    bucket_end_utc: datetime,
    bucket_start_local: datetime,
    bucket_end_local: datetime,
    *,
    flat_threshold_pct: float,
    slope_severity_thresholds: tuple[float, float, float, float] | None,
    futures_slope_severity_thresholds: tuple[float, float, float, float] | None,
) -> IndexFuturesDirectionRegion:
    symbol_stats = _direction_stats(
        bars,
        flat_threshold_pct,
        slope_severity_thresholds=slope_severity_thresholds,
    )
    futures_stats = _direction_stats(
        futures_bars,
        flat_threshold_pct,
        slope_severity_thresholds=futures_slope_severity_thresholds,
    )
    alignment, category = _futures_alignment_category(
        str(symbol_stats["direction"]),
        str(futures_stats["direction"]),
    )
    return IndexFuturesDirectionRegion(
        symbol=symbol,
        futures_symbol=futures_symbol,
        period=period,
        bucket=bucket,
        category=category,
        direction=str(symbol_stats["direction"]),
        futures_direction=str(futures_stats["direction"]),
        futures_alignment=alignment,
        curve_slope_severity=int(symbol_stats["curveSlopeSeverity"]),
        futures_curve_slope_severity=int(futures_stats["curveSlopeSeverity"]),
        curve_slope_severity_thresholds=slope_severity_thresholds,
        futures_curve_slope_severity_thresholds=futures_slope_severity_thresholds,
        start_utc=bars[0].timestamp_utc,
        end_utc=bars[-1].timestamp_utc,
        bucket_timezone=bucket_timezone,
        bucket_start_utc=_format_utc(bucket_start_utc),
        bucket_end_utc=_format_utc(bucket_end_utc),
        bucket_start_local=_format_local(bucket_start_local),
        bucket_end_local=_format_local(bucket_end_local),
        bars=len(bars),
        open=float(symbol_stats["open"]),
        high=float(symbol_stats["high"]),
        low=float(symbol_stats["low"]),
        close=float(symbol_stats["close"]),
        volume=symbol_stats["volume"],  # type: ignore[arg-type]
        return_pct=float(symbol_stats["returnPct"]),
        linear_slope_pct=float(symbol_stats["linearSlopePct"]),
        futures_return_pct=float(futures_stats["returnPct"]),
        futures_linear_slope_pct=float(futures_stats["linearSlopePct"]),
    )


def _options_probability_region(
    symbol: str,
    period: Period,
    bucket: str,
    bars: Sequence[Bar],
    option_trades: Sequence[OptionTrade],
    bucket_timezone: str,
    bucket_start_utc: datetime,
    bucket_end_utc: datetime,
    bucket_start_local: datetime,
    bucket_end_local: datetime,
    *,
    low_momentum_threshold: float,
    high_momentum_threshold: float,
) -> OptionsProbabilityRegion:
    probability_map = build_options_probability_map(option_trades)
    aggregate = probability_map.get("aggregateByUnderlying", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    symbol_aggregate = aggregate.get(symbol.upper(), {})
    if not isinstance(symbol_aggregate, dict):
        symbol_aggregate = {}
    up_probability = float(symbol_aggregate.get("upProbability", 0.5))
    down_probability = float(symbol_aggregate.get("downProbability", 0.5))
    momentum_probability = float(symbol_aggregate.get("momentumProbability", 0.0))
    options_direction = str(symbol_aggregate.get("momentumDirection", "neutral"))
    momentum_regime = _momentum_regime(
        momentum_probability,
        low_momentum_threshold,
        high_momentum_threshold,
    )
    cells = probability_map.get("cells", [])
    if not isinstance(cells, list):
        cells = []
    call_premium = sum(float(cell.get("callPremium", 0.0)) for cell in cells if isinstance(cell, dict))
    put_premium = sum(float(cell.get("putPremium", 0.0)) for cell in cells if isinstance(cell, dict))
    call_volume = sum(float(cell.get("callVolume", 0.0)) for cell in cells if isinstance(cell, dict))
    put_volume = sum(float(cell.get("putVolume", 0.0)) for cell in cells if isinstance(cell, dict))
    open_price = float(bars[0].open)
    close_price = float(bars[-1].close)
    volume_values = [float(bar.volume) for bar in bars if bar.volume is not None]
    return OptionsProbabilityRegion(
        symbol=symbol,
        period=period,
        bucket=bucket,
        category=f"options_{options_direction}_momentum_{momentum_regime}",
        options_direction=options_direction,
        options_momentum_regime=momentum_regime,
        options_up_probability=up_probability,
        options_down_probability=down_probability,
        options_momentum_probability=momentum_probability,
        options_cell_count=int(symbol_aggregate.get("cells", len(cells))),
        option_trade_count=len(option_trades),
        option_call_premium=call_premium,
        option_put_premium=put_premium,
        option_call_volume=call_volume,
        option_put_volume=put_volume,
        options_probability_map_3d=probability_map,
        start_utc=bars[0].timestamp_utc,
        end_utc=bars[-1].timestamp_utc,
        bucket_timezone=bucket_timezone,
        bucket_start_utc=_format_utc(bucket_start_utc),
        bucket_end_utc=_format_utc(bucket_end_utc),
        bucket_start_local=_format_local(bucket_start_local),
        bucket_end_local=_format_local(bucket_end_local),
        bars=len(bars),
        open=open_price,
        high=max(float(bar.high) for bar in bars),
        low=min(float(bar.low) for bar in bars),
        close=close_price,
        volume=sum(volume_values) if len(volume_values) == len(bars) else None,
        return_pct=close_price / open_price - 1.0 if open_price else 0.0,
    )


def _volume_orderbook_region(
    symbol: str,
    period: Period,
    bucket: str,
    bars: Sequence[Bar],
    bucket_timezone: str,
    bucket_start_utc: datetime,
    bucket_end_utc: datetime,
    bucket_start_local: datetime,
    bucket_end_local: datetime,
    *,
    period_average_volume: float,
    low_relative_volume_threshold: float,
    high_relative_volume_threshold: float,
    volume_direction_threshold: float,
    flat_threshold_pct: float,
    slope_severity_thresholds: tuple[float, float, float, float] | None,
) -> VolumeOrderbookRegion:
    stats = _direction_stats(
        bars,
        flat_threshold_pct,
        slope_severity_thresholds=slope_severity_thresholds,
    )
    volumes = [float(bar.volume) for bar in bars if bar.volume is not None]
    volume_total = sum(volumes)
    relative_volume = (
        volume_total / period_average_volume if period_average_volume > 0 else 0.0
    )
    volume_regime = _volume_regime(
        relative_volume,
        low_relative_volume_threshold,
        high_relative_volume_threshold,
    )
    price_volume_correlation = _correlation(_returns(bars), _deltas(volumes))
    price_direction = str(stats["direction"])
    volume_direction = "neutral"
    if relative_volume >= volume_direction_threshold and price_direction in {"up", "down"}:
        volume_direction = price_direction
    fusion_confidence = min(
        1.0,
        abs(price_volume_correlation) * min(relative_volume, 2.0),
    )
    orderbook_status = "awaiting_l2_orderbook_ingestion"
    return VolumeOrderbookRegion(
        symbol=symbol,
        period=period,
        bucket=bucket,
        category=f"volume_{volume_direction}_{volume_regime}_orderbook_pending",
        direction=price_direction,
        curve_slope_severity=int(stats["curveSlopeSeverity"]),
        curve_slope_severity_thresholds=slope_severity_thresholds,
        volume_regime=volume_regime,
        volume_direction=volume_direction,
        relative_volume=relative_volume,
        period_average_volume=period_average_volume,
        average_bar_volume=mean(volumes),
        latest_bar_volume=volumes[-1],
        price_volume_correlation=price_volume_correlation,
        fusion_direction=volume_direction,
        fusion_confidence=fusion_confidence,
        orderbook_status=orderbook_status,
        orderbook_integration_branch=ORDERBOOK_INTEGRATION_BRANCH,
        orderbook_bid_ask_imbalance=None,
        orderbook_book_pressure=None,
        orderbook_depth_slope=None,
        start_utc=bars[0].timestamp_utc,
        end_utc=bars[-1].timestamp_utc,
        bucket_timezone=bucket_timezone,
        bucket_start_utc=_format_utc(bucket_start_utc),
        bucket_end_utc=_format_utc(bucket_end_utc),
        bucket_start_local=_format_local(bucket_start_local),
        bucket_end_local=_format_local(bucket_end_local),
        bars=len(bars),
        open=float(stats["open"]),
        high=float(stats["high"]),
        low=float(stats["low"]),
        close=float(stats["close"]),
        volume=volume_total,
        return_pct=float(stats["returnPct"]),
        linear_slope_pct=float(stats["linearSlopePct"]),
    )


def _direction_stats(
    bars: Sequence[Bar],
    flat_threshold_pct: float,
    *,
    slope_severity_thresholds: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    open_price = float(bars[0].open)
    close_price = float(bars[-1].close)
    linear_slope_pct = _full_window_slope_pct([float(bar.close) for bar in bars])
    volume_values = [float(bar.volume) for bar in bars if bar.volume is not None]
    return {
        "open": open_price,
        "high": max(float(bar.high) for bar in bars),
        "low": min(float(bar.low) for bar in bars),
        "close": close_price,
        "volume": sum(volume_values) if len(volume_values) == len(bars) else None,
        "returnPct": close_price / open_price - 1.0 if open_price else 0.0,
        "linearSlopePct": linear_slope_pct,
        "direction": _direction_from_slope(linear_slope_pct, flat_threshold_pct),
        "curveSlopeSeverity": _slope_severity(linear_slope_pct, slope_severity_thresholds),
    }


def _futures_alignment_category(
    symbol_direction: str,
    futures_direction: str,
) -> tuple[FuturesAlignment, str]:
    directional_values = {"up", "down"}
    if symbol_direction in directional_values and futures_direction in directional_values:
        if symbol_direction == futures_direction:
            return "aligned", f"aligned_{symbol_direction}"
        return (
            "conflicting",
            f"conflicting_symbol_{symbol_direction}_future_{futures_direction}",
        )
    return "neutral_or_unknown", f"symbol_{symbol_direction}_future_{futures_direction}"


def _bucket_bars(
    bars: Sequence[Bar],
    period: Period,
    bucket_zone: ZoneInfo,
) -> tuple[
    dict[str, list[Bar]],
    dict[str, tuple[datetime, datetime, datetime, datetime]],
]:
    buckets: dict[str, list[Bar]] = {}
    bucket_bounds: dict[str, tuple[datetime, datetime, datetime, datetime]] = {}
    for bar in bars:
        timestamp = _parse_utc(bar.timestamp_utc)
        bucket, bucket_start_local, bucket_end_local = _period_bucket(
            timestamp,
            period,
            bucket_zone,
        )
        buckets.setdefault(bucket, []).append(bar)
        bucket_bounds[bucket] = (
            bucket_start_local,
            bucket_end_local,
            bucket_start_local.astimezone(UTC),
            bucket_end_local.astimezone(UTC),
        )
    return buckets, bucket_bounds


def _volume_total(bars: Sequence[Bar]) -> float | None:
    volumes = [float(bar.volume) for bar in bars if bar.volume is not None]
    if len(volumes) != len(bars) or not volumes:
        return None
    return sum(volumes)


def _period_bucket(
    timestamp: datetime,
    period: Period,
    bucket_zone: ZoneInfo,
) -> tuple[str, datetime, datetime]:
    local_timestamp = timestamp.astimezone(bucket_zone)
    if period == "day":
        start = datetime(
            local_timestamp.year,
            local_timestamp.month,
            local_timestamp.day,
            tzinfo=bucket_zone,
        )
        return (start.date().isoformat(), start, start + timedelta(days=1))
    if period == "week":
        start_date = local_timestamp.date() - timedelta(days=local_timestamp.weekday())
        start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=bucket_zone)
        iso_year, iso_week, _ = local_timestamp.isocalendar()
        return (f"{iso_year}-W{iso_week:02d}", start, start + timedelta(days=7))
    start = datetime(local_timestamp.year, local_timestamp.month, 1, tzinfo=bucket_zone)
    if local_timestamp.month == 12:
        end = datetime(local_timestamp.year + 1, 1, 1, tzinfo=bucket_zone)
    else:
        end = datetime(local_timestamp.year, local_timestamp.month + 1, 1, tzinfo=bucket_zone)
    return (f"{local_timestamp.year}-{local_timestamp.month:02d}", start, end)


def _parse_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _format_local(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _bucket_zone(bucket_timezone: str) -> ZoneInfo:
    if bucket_timezone.upper() == "UTC":
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(bucket_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unsupported bucket timezone {bucket_timezone!r}") from exc


def _direction_from_slope(slope_pct: float, flat_threshold_pct: float) -> str:
    if slope_pct > flat_threshold_pct:
        return "up"
    if slope_pct < -flat_threshold_pct:
        return "down"
    return "flat"


def _slope_severity(
    slope_pct: float,
    thresholds: tuple[float, float, float, float] | None = None,
) -> int:
    return slope_severity_from_slope(slope_pct, thresholds)


def _volatility_regime(
    realized_volatility_pct: float,
    thresholds: tuple[float, float],
) -> VolatilityRegime:
    return cast(
        VolatilityRegime,
        volatility_regime_from_realized(realized_volatility_pct, thresholds),
    )


def _momentum_regime(
    momentum_probability: float,
    low_momentum_threshold: float,
    high_momentum_threshold: float,
) -> MomentumRegime:
    if momentum_probability < low_momentum_threshold:
        return "low"
    if momentum_probability < high_momentum_threshold:
        return "medium"
    return "high"


def _volume_regime(
    relative_volume: float,
    low_relative_volume_threshold: float,
    high_relative_volume_threshold: float,
) -> VolumeRegime:
    if relative_volume <= low_relative_volume_threshold:
        return "low"
    if relative_volume >= high_relative_volume_threshold:
        return "high"
    return "normal"
