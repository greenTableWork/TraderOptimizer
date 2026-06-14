from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trader_optimizer.config import write_json
from trader_optimizer.data import Bar
from trader_optimizer.series_math import full_window_slope_pct as _full_window_slope_pct


SLOPE_SEVERITY_CONFIG_SCHEMA = "instrument_slope_severity_config.v1"
DEFAULT_SLOPE_SEVERITY_THRESHOLDS: tuple[float, float, float, float] = (
    0.0025,
    0.01,
    0.03,
    0.07,
)
DEFAULT_SLOPE_SEVERITY_QUANTILES: tuple[float, float, float, float] = (
    0.20,
    0.40,
    0.60,
    0.80,
)
SLOPE_SEVERITY_THRESHOLD_KEYS = (
    "severity1Max",
    "severity2Max",
    "severity3Max",
    "severity4Max",
)


@dataclass(frozen=True)
class SlopeSeverityEntry:
    symbol: str
    period: str
    thresholds: tuple[float, float, float, float]
    bucket_timezone: str | None = None
    bar_size: str | None = None
    what_to_show: str | None = None
    use_rth: int | None = None
    sample_count: int = 0
    min_bars: int | None = None
    start_utc: str | None = None
    end_utc: str | None = None
    method: str = "absolute_linear_slope_pct_quantiles"

    def to_dict(self) -> dict[str, object]:
        output: dict[str, object] = {
            "symbol": self.symbol,
            "period": self.period,
            "thresholds": slope_severity_thresholds_to_dict(self.thresholds),
            "sampleCount": self.sample_count,
            "method": self.method,
        }
        if self.bucket_timezone is not None:
            output["bucketTimezone"] = self.bucket_timezone
        if self.bar_size is not None:
            output["barSize"] = self.bar_size
        if self.what_to_show is not None:
            output["whatToShow"] = self.what_to_show
        if self.use_rth is not None:
            output["useRth"] = self.use_rth
        if self.min_bars is not None:
            output["minBars"] = self.min_bars
        if self.start_utc is not None:
            output["startUtc"] = self.start_utc
        if self.end_utc is not None:
            output["endUtc"] = self.end_utc
        return output


@dataclass(frozen=True)
class SlopeSeverityConfig:
    entries: tuple[SlopeSeverityEntry, ...]
    fallback_thresholds: tuple[float, float, float, float] = DEFAULT_SLOPE_SEVERITY_THRESHOLDS
    schema: str = SLOPE_SEVERITY_CONFIG_SCHEMA

    @classmethod
    def from_path(cls, path: Path) -> SlopeSeverityConfig:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Slope severity config {path} must be a JSON object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SlopeSeverityConfig:
        schema = str(payload.get("schema") or SLOPE_SEVERITY_CONFIG_SCHEMA)
        if schema != SLOPE_SEVERITY_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported slope severity config schema {schema!r}; "
                f"expected {SLOPE_SEVERITY_CONFIG_SCHEMA!r}"
            )
        fallback_thresholds = _thresholds_from_payload(
            payload.get("fallbackThresholds"),
            default=DEFAULT_SLOPE_SEVERITY_THRESHOLDS,
        )
        raw_entries = payload.get("entries") or ()
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
            raise ValueError("Slope severity config entries must be a list")
        entries = tuple(_entry_from_payload(entry) for entry in raw_entries)
        return cls(entries=entries, fallback_thresholds=fallback_thresholds, schema=schema)

    def thresholds_for(
        self,
        symbol: str,
        period: str,
        *,
        bucket_timezone: str | None = None,
        bar_size: str | None = None,
        what_to_show: str | None = None,
        use_rth: int | None = None,
    ) -> tuple[float, float, float, float]:
        normalized_symbol = symbol.upper()
        normalized_period = period.lower()
        scored: list[tuple[int, SlopeSeverityEntry]] = []
        for entry in self.entries:
            if entry.symbol.upper() != normalized_symbol or entry.period.lower() != normalized_period:
                continue
            score = _optional_match_score(entry.bucket_timezone, bucket_timezone)
            if score is None:
                continue
            total_score = score
            for left, right in (
                (entry.bar_size, bar_size),
                (entry.what_to_show, what_to_show),
                (
                    str(entry.use_rth) if entry.use_rth is not None else None,
                    str(use_rth) if use_rth is not None else None,
                ),
            ):
                score = _optional_match_score(left, right)
                if score is None:
                    break
                total_score += score
            else:
                scored.append((total_score, entry))
        if not scored:
            return self.fallback_thresholds
        return max(scored, key=lambda item: item[0])[1].thresholds

    def thresholds_by_period(
        self,
        symbol: str,
        periods: Iterable[str],
        *,
        bucket_timezone: str | None = None,
        bar_size: str | None = None,
        what_to_show: str | None = None,
        use_rth: int | None = None,
    ) -> dict[str, tuple[float, float, float, float]]:
        return {
            period: self.thresholds_for(
                symbol,
                period,
                bucket_timezone=bucket_timezone,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
            )
            for period in periods
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fallbackThresholds": slope_severity_thresholds_to_dict(
                self.fallback_thresholds
            ),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def load_slope_severity_config(path: Path | None) -> SlopeSeverityConfig | None:
    if path is None:
        return None
    return SlopeSeverityConfig.from_path(path)


def slope_severity_from_slope(
    slope_pct: float,
    thresholds: Sequence[float] | Mapping[str, float] | None = None,
) -> int:
    normalized_thresholds = _thresholds_from_payload(
        thresholds,
        default=DEFAULT_SLOPE_SEVERITY_THRESHOLDS,
    )
    magnitude = abs(float(slope_pct))
    for severity, threshold in enumerate(normalized_thresholds, start=1):
        if magnitude < threshold or (threshold == 0.0 and magnitude == 0.0):
            return severity
    return 5


def slope_severity_thresholds_to_dict(
    thresholds: Sequence[float] | Mapping[str, float],
) -> dict[str, float]:
    normalized = _thresholds_from_payload(
        thresholds,
        default=DEFAULT_SLOPE_SEVERITY_THRESHOLDS,
    )
    return {
        key: value
        for key, value in zip(SLOPE_SEVERITY_THRESHOLD_KEYS, normalized, strict=True)
    }


def thresholds_for_period(
    thresholds_by_period: Mapping[str, Sequence[float] | Mapping[str, float]]
    | Sequence[float]
    | Mapping[str, float]
    | None,
    period: str,
) -> tuple[float, float, float, float] | None:
    if thresholds_by_period is None:
        return None
    if _looks_like_thresholds(thresholds_by_period):
        return _thresholds_from_payload(
            thresholds_by_period,
            default=DEFAULT_SLOPE_SEVERITY_THRESHOLDS,
        )
    if not isinstance(thresholds_by_period, Mapping):
        return _thresholds_from_payload(
            thresholds_by_period,
            default=DEFAULT_SLOPE_SEVERITY_THRESHOLDS,
        )
    raw = thresholds_by_period.get(period) or thresholds_by_period.get(period.lower())
    if raw is None:
        return None
    return _thresholds_from_payload(raw, default=DEFAULT_SLOPE_SEVERITY_THRESHOLDS)


def build_slope_severity_config(
    symbol_bars: Mapping[str, Sequence[Bar]],
    *,
    periods: Sequence[str],
    bucket_timezone: str,
    min_bars: int,
    quantiles: Sequence[float] = DEFAULT_SLOPE_SEVERITY_QUANTILES,
    profiles: Mapping[str, Mapping[str, object]] | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> dict[str, object]:
    normalized_quantiles = _normalize_quantiles(quantiles)
    entries: list[SlopeSeverityEntry] = []
    for symbol, bars in sorted(symbol_bars.items()):
        profile = (profiles or {}).get(symbol, {})
        slope_samples = slope_samples_by_period(
            bars,
            periods=periods,
            bucket_timezone=bucket_timezone,
            min_bars=min_bars,
        )
        for period in periods:
            samples = slope_samples.get(period, [])
            thresholds = _thresholds_from_samples(samples, normalized_quantiles)
            entries.append(
                SlopeSeverityEntry(
                    symbol=symbol,
                    period=period,
                    thresholds=thresholds,
                    bucket_timezone=bucket_timezone,
                    bar_size=_optional_str(profile.get("barSize") or profile.get("bar_size")),
                    what_to_show=_optional_str(
                        profile.get("whatToShow") or profile.get("what_to_show")
                    ),
                    use_rth=_optional_int(profile.get("useRth") or profile.get("use_rth")),
                    sample_count=len(samples),
                    min_bars=min_bars,
                    start_utc=start_utc,
                    end_utc=end_utc,
                )
            )
    config = SlopeSeverityConfig(entries=tuple(entries))
    output = config.to_dict()
    output.update(
        {
            "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "method": "absolute_linear_slope_pct_quantiles",
            "quantiles": {
                key: quantile
                for key, quantile in zip(
                    SLOPE_SEVERITY_THRESHOLD_KEYS,
                    normalized_quantiles,
                    strict=True,
                )
            },
            "severityRange": [1, 5],
        }
    )
    return output


def write_slope_severity_config(path: Path, config: Mapping[str, object]) -> None:
    write_json(path, config)


def slope_samples_by_period(
    bars: Sequence[Bar],
    *,
    periods: Sequence[str],
    bucket_timezone: str,
    min_bars: int,
) -> dict[str, list[float]]:
    bucket_zone = _bucket_zone(bucket_timezone)
    sorted_bars = sorted(bars, key=lambda bar: _parse_utc(bar.timestamp_utc))
    output: dict[str, list[float]] = {period: [] for period in periods}
    for period in periods:
        buckets: dict[str, list[Bar]] = {}
        for bar in sorted_bars:
            bucket = _period_bucket(_parse_utc(bar.timestamp_utc), period, bucket_zone)
            buckets.setdefault(bucket, []).append(bar)
        for bucket_bars in buckets.values():
            if len(bucket_bars) < min_bars:
                continue
            slope_pct = _full_window_slope_pct([float(bar.close) for bar in bucket_bars])
            output[period].append(abs(slope_pct))
    return output


def _entry_from_payload(payload: object) -> SlopeSeverityEntry:
    if not isinstance(payload, Mapping):
        raise ValueError("Slope severity config entry must be an object")
    symbol = str(payload.get("symbol") or "").upper()
    period = str(payload.get("period") or "").lower()
    if not symbol:
        raise ValueError("Slope severity config entry missing symbol")
    if period not in {"day", "week", "month"}:
        raise ValueError(f"Slope severity config entry has unsupported period {period!r}")
    return SlopeSeverityEntry(
        symbol=symbol,
        period=period,
        thresholds=_thresholds_from_payload(payload.get("thresholds"), default=DEFAULT_SLOPE_SEVERITY_THRESHOLDS),
        bucket_timezone=_optional_str(payload.get("bucketTimezone")),
        bar_size=_optional_str(payload.get("barSize")),
        what_to_show=_optional_str(payload.get("whatToShow")),
        use_rth=_optional_int(payload.get("useRth")),
        sample_count=int(payload.get("sampleCount") or 0),
        min_bars=_optional_int(payload.get("minBars")),
        start_utc=_optional_str(payload.get("startUtc")),
        end_utc=_optional_str(payload.get("endUtc")),
        method=str(payload.get("method") or "absolute_linear_slope_pct_quantiles"),
    )


def _thresholds_from_payload(
    payload: object,
    *,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if payload is None:
        return default
    if isinstance(payload, Mapping):
        values = [payload.get(key) for key in SLOPE_SEVERITY_THRESHOLD_KEYS]
        if any(value is None for value in values):
            raise ValueError(
                "Slope severity thresholds must define severity1Max through severity4Max"
            )
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        values = list(payload)
    else:
        raise ValueError("Slope severity thresholds must be a list or object")
    if len(values) != 4:
        raise ValueError("Slope severity thresholds must contain exactly four values")
    thresholds = tuple(float(value) for value in values)
    if any(value < 0 for value in thresholds):
        raise ValueError("Slope severity thresholds must be non-negative")
    if list(thresholds) != sorted(thresholds):
        raise ValueError("Slope severity thresholds must be non-decreasing")
    return thresholds  # type: ignore[return-value]


def _looks_like_thresholds(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(key in value for key in SLOPE_SEVERITY_THRESHOLD_KEYS)
    return False


def _thresholds_from_samples(
    samples: Sequence[float],
    quantiles: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not samples:
        return DEFAULT_SLOPE_SEVERITY_THRESHOLDS
    sorted_samples = sorted(float(sample) for sample in samples)
    return tuple(_quantile(sorted_samples, quantile) for quantile in quantiles)  # type: ignore[return-value]


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile for empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def _normalize_quantiles(
    raw_quantiles: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(raw_quantiles) != 4:
        raise ValueError("Exactly four slope severity quantiles are required")
    quantiles = tuple(float(value) for value in raw_quantiles)
    if any(value <= 0.0 or value >= 1.0 for value in quantiles):
        raise ValueError("Slope severity quantiles must be between 0 and 1")
    if list(quantiles) != sorted(quantiles):
        raise ValueError("Slope severity quantiles must be non-decreasing")
    return quantiles  # type: ignore[return-value]


def _optional_match_score(left: str | None, right: str | None) -> int | None:
    if right is None or right == "":
        return 0
    if left is None or left == "":
        return 0
    if str(left).upper() != str(right).upper():
        return None
    return 1


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _bucket_zone(bucket_timezone: str) -> ZoneInfo:
    if bucket_timezone.upper() == "UTC":
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(bucket_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unsupported bucket timezone {bucket_timezone!r}") from exc


def _period_bucket(timestamp: datetime, period: str, bucket_zone: ZoneInfo) -> str:
    local_timestamp = timestamp.astimezone(bucket_zone)
    if period == "day":
        return local_timestamp.date().isoformat()
    if period == "week":
        iso_year, iso_week, _ = local_timestamp.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if period == "month":
        return f"{local_timestamp.year}-{local_timestamp.month:02d}"
    raise ValueError(f"Unsupported period {period!r}")


def _parse_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
