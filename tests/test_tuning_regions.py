import json
from dataclasses import replace
from pathlib import Path

import pytest

from trader_optimizer.data import Bar
from trader_optimizer.market_features import ORDERBOOK_INTEGRATION_BRANCH, OptionTrade
from trader_optimizer.slope_severity import (
    SlopeSeverityConfig,
    build_slope_severity_config,
    slope_severity_from_slope,
)
from trader_optimizer.volatility_regime import (
    VolatilityRegimeConfig,
    build_volatility_regime_config,
    volatility_regime_from_realized,
)
from trader_optimizer.tuning_regions import (
    categorize_direction_regions,
    categorize_index_futures_direction_regions,
    categorize_options_probability_regions,
    categorize_volume_orderbook_regions,
    categorize_volatility_regions,
    direction_region_summary,
    filter_duplicate_regions,
    load_region_ids,
    normalize_directions,
    normalize_futures_alignments,
    normalize_momentum_regimes,
    normalize_periods,
    normalize_volume_directions,
    normalize_volume_regimes,
    normalize_volatility_regimes,
    region_id_for_region,
    write_region_summary,
    write_regions_csv,
    write_regions_jsonl,
)


def _bar(timestamp: str, close: float, volume: float = 100.0) -> Bar:
    return Bar(
        timestamp_utc=timestamp,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
    )


def test_categorize_direction_regions_builds_day_week_month_regions() -> None:
    bars = [
        _bar("2026-05-04T13:30:00+00:00", 100.0),
        _bar("2026-05-04T20:00:00+00:00", 104.0),
        _bar("2026-05-05T13:30:00+00:00", 103.0),
        _bar("2026-05-05T20:00:00+00:00", 101.0),
        _bar("2026-06-01T13:30:00+00:00", 110.0),
        _bar("2026-06-01T20:00:00+00:00", 116.0),
    ]

    regions = categorize_direction_regions("AAPL", bars)

    assert [region.period for region in regions] == [
        "day",
        "day",
        "day",
        "week",
        "week",
        "month",
        "month",
    ]
    assert regions[0].bucket == "2026-05-04"
    assert regions[0].category == "up_slope_4"
    assert regions[0].bucket_timezone == "UTC"
    assert regions[0].bucket_start_local == "2026-05-04T00:00:00+00:00"
    assert regions[0].curve_slope_severity_baseline == 3
    assert regions[0].to_dict()["backtestRegion"] == {
        "startUtc": "2026-05-04T13:30:00+00:00",
        "endUtc": "2026-05-04T20:00:00+00:00",
        "category": "up_slope_4",
        "period": "day",
    }
    assert regions[1].category.startswith("down_slope_")
    assert regions[-1].bucket == "2026-06"


def test_direction_slope_severity_can_use_instrument_normalized_thresholds() -> None:
    regions = categorize_direction_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T20:00:00+00:00", 104.0),
        ],
        periods=("day",),
        slope_severity_thresholds={"day": (0.05, 0.10, 0.20, 0.40)},
    )

    assert len(regions) == 1
    assert regions[0].linear_slope_pct == pytest.approx(0.04)
    assert regions[0].category == "up_slope_1"
    assert regions[0].curve_slope_severity == 1
    assert regions[0].to_dict()["curveSlopeSeverityThresholds"] == {
        "severity1Max": 0.05,
        "severity2Max": 0.10,
        "severity3Max": 0.20,
        "severity4Max": 0.40,
    }


def test_categorize_direction_regions_supports_exchange_timezone_buckets() -> None:
    bars = [
        _bar("2026-05-04T23:30:00+00:00", 100.0),
        _bar("2026-05-05T00:30:00+00:00", 104.0),
    ]

    utc_regions = categorize_direction_regions("AAPL", bars, periods=("day",))
    new_york_regions = categorize_direction_regions(
        "AAPL",
        bars,
        periods=("day",),
        bucket_timezone="America/New_York",
    )

    assert utc_regions == []
    assert len(new_york_regions) == 1
    assert new_york_regions[0].bucket == "2026-05-04"
    assert new_york_regions[0].bucket_timezone == "America/New_York"
    assert new_york_regions[0].bucket_start_utc == "2026-05-04T04:00:00+00:00"
    assert new_york_regions[0].bucket_end_utc == "2026-05-05T04:00:00+00:00"
    assert new_york_regions[0].bucket_start_local == "2026-05-04T00:00:00-04:00"


def test_categorize_direction_regions_can_filter_flat_regions() -> None:
    regions = categorize_direction_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T20:00:00+00:00", 100.1),
        ],
        periods=("day",),
        directions=("up", "down"),
    )

    assert regions == []


def test_categorize_direction_regions_skips_sparse_buckets() -> None:
    regions = categorize_direction_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-05T13:30:00+00:00", 101.0),
        ],
        periods=("day",),
    )

    assert regions == []


def test_region_summary_and_writers(tmp_path: Path) -> None:
    regions = categorize_direction_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0, 50.0),
            _bar("2026-05-04T20:00:00+00:00", 104.0, 70.0),
        ],
        periods=("day",),
    )

    summary = direction_region_summary(regions)
    assert summary["schema"] == "tuning_direction_regions.v1"
    assert summary["tuningSubcategory"] == "direction"
    assert summary["uniqueRegionIds"] == 1
    assert summary["duplicateRegionCount"] == 0
    assert summary["categories"] == {"up_slope_4": 1}
    assert summary["bucketTimezones"] == {"UTC": 1}

    jsonl_path = tmp_path / "regions.jsonl"
    csv_path = tmp_path / "regions.csv"
    summary_path = tmp_path / "summary.json"
    write_regions_jsonl(jsonl_path, regions)
    write_regions_csv(csv_path, regions)
    write_region_summary(summary_path, regions)

    jsonl_row = json.loads(jsonl_path.read_text().splitlines()[0])
    assert jsonl_row["category"] == "up_slope_4"
    assert jsonl_row["regionId"] == region_id_for_region(regions[0])
    csv_header = csv_path.read_text().splitlines()[0]
    assert "regionId" in csv_header
    assert "backtestRegion" not in csv_header
    assert json.loads(summary_path.read_text())["regions"] == 1


def test_region_ids_can_filter_duplicate_reruns(tmp_path: Path) -> None:
    regions = categorize_direction_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T20:00:00+00:00", 104.0),
        ],
        periods=("day",),
    )
    assert len(regions) == 1
    first_id = region_id_for_region(regions[0])

    unique_regions, skipped_ids = filter_duplicate_regions([regions[0], regions[0]])
    assert unique_regions == regions
    assert skipped_ids == [first_id]

    jsonl_path = tmp_path / "existing_regions.jsonl"
    csv_path = tmp_path / "existing_regions.csv"
    write_regions_jsonl(jsonl_path, regions)
    write_regions_csv(csv_path, regions)
    existing_ids = load_region_ids([jsonl_path, csv_path])

    filtered_regions, skipped_existing = filter_duplicate_regions(
        regions,
        existing_region_ids=existing_ids,
    )

    assert existing_ids == {first_id}
    assert filtered_regions == []
    assert skipped_existing == [first_id]


def test_region_id_includes_bar_profile_when_attached() -> None:
    regions = categorize_direction_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T20:00:00+00:00", 104.0),
        ],
        periods=("day",),
    )
    ten_second_region = replace(
        regions[0],
        bar_size="10 secs",
        what_to_show="TRADES",
        use_rth=1,
    )
    one_minute_region = replace(
        regions[0],
        bar_size="1 min",
        what_to_show="TRADES",
        use_rth=1,
    )

    assert region_id_for_region(ten_second_region) != region_id_for_region(
        one_minute_region
    )
    row = ten_second_region.to_dict()
    assert row["barSize"] == "10 secs"
    assert row["whatToShow"] == "TRADES"
    assert row["useRth"] == 1


def test_normalize_periods_rejects_unknown_period() -> None:
    with pytest.raises(ValueError, match="Unsupported period"):
        normalize_periods(["quarter"])


def test_normalize_directions_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="Unsupported direction"):
        normalize_directions(["sideways"])


def test_categorize_volatility_regions_combines_individual_and_market_regimes() -> None:
    bars = [
        _bar("2026-05-04T13:30:00+00:00", 100.0),
        _bar("2026-05-04T14:30:00+00:00", 112.0),
        _bar("2026-05-04T15:30:00+00:00", 90.0),
        _bar("2026-05-04T20:00:00+00:00", 118.0),
    ]
    market_bars = [
        _bar("2026-05-04T13:30:00+00:00", 1000.0),
        _bar("2026-05-04T14:30:00+00:00", 1001.0),
        _bar("2026-05-04T15:30:00+00:00", 1002.0),
        _bar("2026-05-04T20:00:00+00:00", 1003.0),
    ]

    regions = categorize_volatility_regions(
        "AAPL",
        bars,
        market_symbol="SPX",
        market_bars=market_bars,
        periods=("day",),
    )

    assert len(regions) == 1
    region = regions[0]
    assert region.tuning_subcategory == "volatility"
    assert region.individual_volatility_regime == "high"
    assert region.market_symbol == "SPX"
    assert region.market_volatility_regime == "low"
    assert region.category == "individual_high_market_low_vol"
    assert region.realized_volatility_pct > region.market_realized_volatility_pct
    assert region.to_dict()["backtestRegion"] == {
        "startUtc": "2026-05-04T13:30:00+00:00",
        "endUtc": "2026-05-04T20:00:00+00:00",
        "category": "individual_high_market_low_vol",
        "period": "day",
    }


def test_categorize_volatility_regions_can_use_instrument_normalized_thresholds() -> None:
    bars = [
        _bar("2026-05-04T13:30:00+00:00", 100.0),
        _bar("2026-05-04T14:30:00+00:00", 112.0),
        _bar("2026-05-04T15:30:00+00:00", 90.0),
        _bar("2026-05-04T20:00:00+00:00", 118.0),
    ]
    market_bars = [
        _bar("2026-05-04T13:30:00+00:00", 1000.0),
        _bar("2026-05-04T14:30:00+00:00", 1001.0),
        _bar("2026-05-04T15:30:00+00:00", 1002.0),
        _bar("2026-05-04T20:00:00+00:00", 1003.0),
    ]

    regions = categorize_volatility_regions(
        "AAPL",
        bars,
        market_symbol="SPX",
        market_bars=market_bars,
        periods=("day",),
        volatility_regime_thresholds={"day": (0.50, 0.80)},
        market_volatility_regime_thresholds={"day": (0.0000001, 0.0000002)},
    )

    assert len(regions) == 1
    row = regions[0].to_dict()
    assert regions[0].individual_volatility_regime == "low"
    assert regions[0].market_volatility_regime == "high"
    assert regions[0].category == "individual_low_market_high_vol"
    assert row["individualVolatilityRegimeThresholds"] == {
        "lowMax": 0.50,
        "mediumMax": 0.80,
    }
    assert row["marketVolatilityRegimeThresholds"] == {
        "lowMax": 0.0000001,
        "mediumMax": 0.0000002,
    }


def test_categorize_volatility_regions_can_filter_regimes() -> None:
    regions = categorize_volatility_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T14:30:00+00:00", 100.1),
            _bar("2026-05-04T20:00:00+00:00", 100.2),
        ],
        periods=("day",),
        regimes=("high",),
    )

    assert regions == []


def test_volatility_region_summary_and_csv_fields(tmp_path: Path) -> None:
    regions = categorize_volatility_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T14:30:00+00:00", 112.0),
            _bar("2026-05-04T20:00:00+00:00", 90.0),
        ],
        periods=("day",),
    )

    summary_path = tmp_path / "vol_summary.json"
    csv_path = tmp_path / "vol_regions.csv"
    write_region_summary(summary_path, regions)
    write_regions_csv(csv_path, regions)

    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "tuning_volatility_regions.v1"
    assert summary["tuningSubcategory"] == "volatility"
    assert summary["individualVolatilityRegimes"] == {"high": 1}
    header = csv_path.read_text().splitlines()[0]
    assert "realizedVolatilityPct" in header
    assert "individualVolatilityRegime" in header


def test_normalize_volatility_regimes_rejects_unknown_regime() -> None:
    with pytest.raises(ValueError, match="Unsupported volatility regime"):
        normalize_volatility_regimes(["extreme"])


def test_index_futures_direction_regions_label_alignment() -> None:
    bars = [
        _bar("2026-05-04T13:30:00+00:00", 100.0),
        _bar("2026-05-04T14:30:00+00:00", 104.0),
        _bar("2026-05-04T20:00:00+00:00", 108.0),
    ]
    futures_bars = [
        _bar("2026-05-04T13:30:00+00:00", 5000.0),
        _bar("2026-05-04T14:30:00+00:00", 5050.0),
        _bar("2026-05-04T20:00:00+00:00", 5100.0),
    ]

    regions = categorize_index_futures_direction_regions(
        "AAPL",
        bars,
        futures_symbol="ES",
        futures_bars=futures_bars,
        periods=("day",),
    )

    assert len(regions) == 1
    region = regions[0]
    assert region.tuning_subcategory == "index_futures_direction"
    assert region.category == "aligned_up"
    assert region.direction == "up"
    assert region.futures_direction == "up"
    assert region.futures_alignment == "aligned"
    assert region.futures_symbol == "ES"
    assert region.to_dict()["backtestRegion"] == {
        "startUtc": "2026-05-04T13:30:00+00:00",
        "endUtc": "2026-05-04T20:00:00+00:00",
        "category": "aligned_up",
        "period": "day",
    }


def test_index_futures_direction_regions_can_use_separate_slope_thresholds() -> None:
    regions = categorize_index_futures_direction_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T20:00:00+00:00", 104.0),
        ],
        futures_symbol="ES",
        futures_bars=[
            _bar("2026-05-04T13:30:00+00:00", 5000.0),
            _bar("2026-05-04T20:00:00+00:00", 5100.0),
        ],
        periods=("day",),
        slope_severity_thresholds={"day": (0.05, 0.10, 0.20, 0.40)},
        futures_slope_severity_thresholds={"day": (0.005, 0.01, 0.015, 0.03)},
    )

    assert len(regions) == 1
    assert regions[0].curve_slope_severity == 1
    assert regions[0].futures_curve_slope_severity == 4
    row = regions[0].to_dict()
    assert row["curveSlopeSeverityThresholds"]["severity1Max"] == 0.05
    assert row["futuresCurveSlopeSeverityThresholds"]["severity4Max"] == 0.03


def test_index_futures_direction_regions_label_conflict_and_filter_alignment() -> None:
    bars = [
        _bar("2026-05-04T13:30:00+00:00", 100.0),
        _bar("2026-05-04T14:30:00+00:00", 104.0),
        _bar("2026-05-04T20:00:00+00:00", 108.0),
    ]
    futures_bars = [
        _bar("2026-05-04T13:30:00+00:00", 5000.0),
        _bar("2026-05-04T14:30:00+00:00", 4950.0),
        _bar("2026-05-04T20:00:00+00:00", 4900.0),
    ]

    regions = categorize_index_futures_direction_regions(
        "AAPL",
        bars,
        futures_symbol="ES",
        futures_bars=futures_bars,
        periods=("day",),
        alignments=("conflicting",),
    )

    assert len(regions) == 1
    assert regions[0].category == "conflicting_symbol_up_future_down"
    assert regions[0].futures_alignment == "conflicting"

    filtered = categorize_index_futures_direction_regions(
        "AAPL",
        bars,
        futures_symbol="ES",
        futures_bars=futures_bars,
        periods=("day",),
        alignments=("aligned",),
    )
    assert filtered == []


def test_index_futures_region_summary_counts_futures_fields() -> None:
    regions = categorize_index_futures_direction_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T20:00:00+00:00", 100.1),
        ],
        futures_symbol="ES",
        futures_bars=[
            _bar("2026-05-04T13:30:00+00:00", 5000.0),
            _bar("2026-05-04T20:00:00+00:00", 5050.0),
        ],
        periods=("day",),
    )

    summary = direction_region_summary(regions)
    assert summary["schema"] == "tuning_index_futures_direction_regions.v1"
    assert summary["tuningSubcategory"] == "index_futures_direction"
    assert summary["futuresDirections"] == {"up": 1}
    assert summary["futuresAlignments"] == {"neutral_or_unknown": 1}
    assert summary["futuresSymbols"] == {"ES": 1}
    assert summary["categories"] == {"symbol_flat_future_up": 1}


def test_normalize_futures_alignments_rejects_unknown_alignment() -> None:
    with pytest.raises(ValueError, match="Unsupported futures alignment"):
        normalize_futures_alignments(["mixed"])


def test_options_probability_regions_build_3d_probability_category() -> None:
    bars = [
        _bar("2026-05-04T13:30:00+00:00", 100.0),
        _bar("2026-05-04T14:30:00+00:00", 102.0),
        _bar("2026-05-04T20:00:00+00:00", 104.0),
    ]
    option_trades = [
        OptionTrade(
            underlying="AAPL",
            trade_time_utc="2026-05-04T14:00:00+00:00",
            expiration_days=14,
            strike_moneyness=1.02,
            side="CALL",
            premium=10_000,
            volume=100,
            open_interest=500,
            implied_volatility=0.35,
        ),
        OptionTrade(
            underlying="AAPL",
            trade_time_utc="2026-05-04T15:00:00+00:00",
            expiration_days=14,
            strike_moneyness=1.02,
            side="PUT",
            premium=2_000,
            volume=20,
            open_interest=300,
            implied_volatility=0.30,
        ),
    ]

    regions = categorize_options_probability_regions(
        "AAPL",
        bars,
        option_trades=option_trades,
        periods=("day",),
    )

    assert len(regions) == 1
    region = regions[0]
    assert region.tuning_subcategory == "options_probability_map_3d"
    assert region.category == "options_up_momentum_high"
    assert region.options_direction == "up"
    assert region.options_momentum_regime == "high"
    assert region.options_up_probability > region.options_down_probability
    assert region.options_cell_count == 2
    assert region.option_trade_count == 2
    assert region.options_probability_map_3d["axes"] == [
        "expiration_days",
        "strike_moneyness",
        "trade_time_bucket",
    ]
    assert region.to_dict()["backtestRegion"] == {
        "startUtc": "2026-05-04T13:30:00+00:00",
        "endUtc": "2026-05-04T20:00:00+00:00",
        "category": "options_up_momentum_high",
        "period": "day",
    }


def test_options_probability_regions_filter_momentum_and_ignore_other_underlyings() -> None:
    regions = categorize_options_probability_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T20:00:00+00:00", 100.2),
        ],
        option_trades=[
            OptionTrade(
                underlying="MSFT",
                trade_time_utc="2026-05-04T14:00:00+00:00",
                expiration_days=14,
                strike_moneyness=1.0,
                side="CALL",
                premium=10_000,
                volume=100,
            ),
            OptionTrade(
                underlying="AAPL",
                trade_time_utc="2026-05-04T14:00:00+00:00",
                expiration_days=14,
                strike_moneyness=1.0,
                side="CALL",
                premium=1_000,
                volume=10,
            ),
            OptionTrade(
                underlying="AAPL",
                trade_time_utc="2026-05-04T15:00:00+00:00",
                expiration_days=14,
                strike_moneyness=1.0,
                side="PUT",
                premium=1_000,
                volume=10,
            ),
        ],
        periods=("day",),
        momentum_regimes=("high",),
    )

    assert regions == []


def test_options_probability_region_summary_counts_options_fields(tmp_path: Path) -> None:
    regions = categorize_options_probability_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0),
            _bar("2026-05-04T20:00:00+00:00", 104.0),
        ],
        option_trades=[
            OptionTrade(
                underlying="AAPL",
                trade_time_utc="2026-05-04T14:00:00+00:00",
                expiration_days=14,
                strike_moneyness=1.0,
                side="PUT",
                premium=8_000,
                volume=80,
            )
        ],
        periods=("day",),
    )

    summary_path = tmp_path / "options_summary.json"
    csv_path = tmp_path / "options_regions.csv"
    write_region_summary(summary_path, regions)
    write_regions_csv(csv_path, regions)

    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "tuning_options_probability_regions.v1"
    assert summary["tuningSubcategory"] == "options_probability_map_3d"
    assert summary["optionsDirections"] == {"down": 1}
    assert summary["optionsMomentumRegimes"] == {"high": 1}
    assert summary["categories"] == {"options_down_momentum_high": 1}
    header = csv_path.read_text().splitlines()[0]
    assert "optionsUpProbability" in header
    assert "optionsProbabilityMap3d" in header


def test_empty_region_summary_can_preserve_requested_subcategory(tmp_path: Path) -> None:
    summary_path = tmp_path / "empty_options_summary.json"
    write_region_summary(
        summary_path,
        [],
        tuning_subcategory="options_probability_map_3d",
    )
    summary = json.loads(summary_path.read_text())

    assert summary["schema"] == "tuning_options_probability_regions.v1"
    assert summary["tuningSubcategory"] == "options_probability_map_3d"
    assert summary["tuningSubcategories"] == {"options_probability_map_3d": 0}


def test_normalize_momentum_regimes_rejects_unknown_regime() -> None:
    with pytest.raises(ValueError, match="Unsupported momentum regime"):
        normalize_momentum_regimes(["explosive"])


def test_volume_orderbook_regions_label_relative_volume_and_pending_l2() -> None:
    bars = [
        _bar("2026-05-04T13:30:00+00:00", 100.0, 100.0),
        _bar("2026-05-04T14:30:00+00:00", 103.0, 130.0),
        _bar("2026-05-04T20:00:00+00:00", 108.0, 190.0),
        _bar("2026-05-05T13:30:00+00:00", 108.0, 60.0),
        _bar("2026-05-05T14:30:00+00:00", 106.0, 70.0),
        _bar("2026-05-05T20:00:00+00:00", 104.0, 80.0),
    ]

    regions = categorize_volume_orderbook_regions("AAPL", bars, periods=("day",))

    assert len(regions) == 2
    high_region = regions[0]
    assert high_region.tuning_subcategory == "trade_volume_orderbook"
    assert high_region.category == "volume_up_high_orderbook_pending"
    assert high_region.direction == "up"
    assert high_region.volume_regime == "high"
    assert high_region.volume_direction == "up"
    assert high_region.fusion_direction == "up"
    assert high_region.orderbook_status == "awaiting_l2_orderbook_ingestion"
    assert high_region.orderbook_integration_branch == ORDERBOOK_INTEGRATION_BRANCH
    assert high_region.relative_volume > 1.25
    assert high_region.price_volume_correlation > 0.0
    assert high_region.to_dict()["tradeVolumeOrderbook"] == {
        "status": "volume_only",
        "volumeDirection": "up",
        "fusionDirection": "up",
        "confidence": high_region.fusion_confidence,
        "orderbookStatus": "awaiting_l2_orderbook_ingestion",
        "integrationBranch": ORDERBOOK_INTEGRATION_BRANCH,
        "requiredOrderbookFeatures": [
            "bid_ask_imbalance",
            "book_pressure",
            "depth_slope",
        ],
    }
    assert high_region.to_dict()["backtestRegion"] == {
        "startUtc": "2026-05-04T13:30:00+00:00",
        "endUtc": "2026-05-04T20:00:00+00:00",
        "category": "volume_up_high_orderbook_pending",
        "period": "day",
    }
    assert regions[1].category == "volume_neutral_low_orderbook_pending"


def test_volume_orderbook_regions_use_normalized_slope_severity() -> None:
    regions = categorize_volume_orderbook_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0, 100.0),
            _bar("2026-05-04T20:00:00+00:00", 104.0, 140.0),
            _bar("2026-05-05T13:30:00+00:00", 104.0, 100.0),
            _bar("2026-05-05T20:00:00+00:00", 104.2, 100.0),
        ],
        periods=("day",),
        slope_severity_thresholds={"day": (0.05, 0.10, 0.20, 0.40)},
    )

    assert len(regions) == 2
    assert regions[0].curve_slope_severity == 1
    assert regions[0].to_dict()["curveSlopeSeverityThresholds"]["severity1Max"] == 0.05


def test_volume_orderbook_regions_filter_regime_and_direction() -> None:
    regions = categorize_volume_orderbook_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0, 100.0),
            _bar("2026-05-04T14:30:00+00:00", 103.0, 130.0),
            _bar("2026-05-04T20:00:00+00:00", 108.0, 190.0),
            _bar("2026-05-05T13:30:00+00:00", 108.0, 60.0),
            _bar("2026-05-05T14:30:00+00:00", 106.0, 70.0),
            _bar("2026-05-05T20:00:00+00:00", 104.0, 80.0),
        ],
        periods=("day",),
        volume_regimes=("high",),
        volume_directions=("up",),
    )

    assert len(regions) == 1
    assert regions[0].category == "volume_up_high_orderbook_pending"


def test_volume_orderbook_region_summary_and_csv_fields(tmp_path: Path) -> None:
    regions = categorize_volume_orderbook_regions(
        "AAPL",
        [
            _bar("2026-05-04T13:30:00+00:00", 100.0, 100.0),
            _bar("2026-05-04T14:30:00+00:00", 103.0, 130.0),
            _bar("2026-05-04T20:00:00+00:00", 108.0, 190.0),
            _bar("2026-05-05T13:30:00+00:00", 108.0, 60.0),
            _bar("2026-05-05T14:30:00+00:00", 106.0, 70.0),
            _bar("2026-05-05T20:00:00+00:00", 104.0, 80.0),
        ],
        periods=("day",),
    )

    summary_path = tmp_path / "volume_summary.json"
    csv_path = tmp_path / "volume_regions.csv"
    write_region_summary(summary_path, regions)
    write_regions_csv(csv_path, regions)

    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "tuning_trade_volume_orderbook_regions.v1"
    assert summary["tuningSubcategory"] == "trade_volume_orderbook"
    assert summary["directions"] == {"down": 1, "up": 1}
    assert summary["volumeRegimes"] == {"high": 1, "low": 1}
    assert summary["volumeDirections"] == {"neutral": 1, "up": 1}
    assert summary["fusionDirections"] == {"neutral": 1, "up": 1}
    assert summary["orderbookStatuses"] == {"awaiting_l2_orderbook_ingestion": 2}
    header = csv_path.read_text().splitlines()[0]
    assert "relativeVolume" in header
    assert "orderbookIntegrationBranch" in header


def test_empty_region_summary_can_preserve_volume_orderbook_subcategory(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "empty_volume_summary.json"
    write_region_summary(
        summary_path,
        [],
        tuning_subcategory="trade_volume_orderbook",
    )
    summary = json.loads(summary_path.read_text())

    assert summary["schema"] == "tuning_trade_volume_orderbook_regions.v1"
    assert summary["tuningSubcategory"] == "trade_volume_orderbook"
    assert summary["tuningSubcategories"] == {"trade_volume_orderbook": 0}


def test_normalize_volume_filters_reject_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported volume regime"):
        normalize_volume_regimes(["spike"])
    with pytest.raises(ValueError, match="Unsupported volume direction"):
        normalize_volume_directions(["sideways"])


def test_build_slope_severity_config_and_lookup_by_profile() -> None:
    config_payload = build_slope_severity_config(
        {
            "AAPL": [
                _bar("2026-05-04T13:30:00+00:00", 100.0),
                _bar("2026-05-04T20:00:00+00:00", 101.0),
                _bar("2026-05-05T13:30:00+00:00", 100.0),
                _bar("2026-05-05T20:00:00+00:00", 104.0),
                _bar("2026-05-06T13:30:00+00:00", 100.0),
                _bar("2026-05-06T20:00:00+00:00", 109.0),
                _bar("2026-05-07T13:30:00+00:00", 100.0),
                _bar("2026-05-07T20:00:00+00:00", 116.0),
                _bar("2026-05-08T13:30:00+00:00", 100.0),
                _bar("2026-05-08T20:00:00+00:00", 125.0),
            ]
        },
        periods=("day",),
        bucket_timezone="America/New_York",
        min_bars=2,
        profiles={
            "AAPL": {
                "barSize": "10 secs",
                "whatToShow": "TRADES",
                "useRth": 1,
            }
        },
    )

    config = SlopeSeverityConfig.from_dict(config_payload)
    thresholds = config.thresholds_for(
        "AAPL",
        "day",
        bucket_timezone="America/New_York",
        bar_size="10 secs",
        what_to_show="TRADES",
        use_rth=1,
    )

    assert config_payload["schema"] == "instrument_slope_severity_config.v1"
    assert len(config_payload["entries"]) == 1
    assert thresholds == pytest.approx((0.034, 0.07, 0.118, 0.178))
    assert slope_severity_from_slope(0.04, thresholds) == 2


def test_build_volatility_regime_config_and_lookup_by_profile() -> None:
    config_payload = build_volatility_regime_config(
        {
            "AAPL": [
                _bar("2026-05-04T13:30:00+00:00", 100.0),
                _bar("2026-05-04T16:00:00+00:00", 101.0),
                _bar("2026-05-04T20:00:00+00:00", 102.0),
                _bar("2026-05-05T13:30:00+00:00", 100.0),
                _bar("2026-05-05T16:00:00+00:00", 103.0),
                _bar("2026-05-05T20:00:00+00:00", 101.0),
                _bar("2026-05-06T13:30:00+00:00", 100.0),
                _bar("2026-05-06T16:00:00+00:00", 108.0),
                _bar("2026-05-06T20:00:00+00:00", 96.0),
            ]
        },
        periods=("day",),
        bucket_timezone="America/New_York",
        min_bars=3,
        profiles={
            "AAPL": {
                "barSize": "10 secs",
                "whatToShow": "TRADES",
                "useRth": 1,
            }
        },
    )

    config = VolatilityRegimeConfig.from_dict(config_payload)
    thresholds = config.thresholds_for(
        "AAPL",
        "day",
        bucket_timezone="America/New_York",
        bar_size="10 secs",
        what_to_show="TRADES",
        use_rth=1,
    )

    assert config_payload["schema"] == "instrument_volatility_regime_config.v1"
    assert len(config_payload["entries"]) == 1
    assert config_payload["entries"][0]["sampleCount"] == 3
    assert thresholds[0] < thresholds[1]
    assert volatility_regime_from_realized(thresholds[0] / 2.0, thresholds) == "low"
    assert volatility_regime_from_realized(
        (thresholds[0] + thresholds[1]) / 2.0,
        thresholds,
    ) == "medium"
    assert volatility_regime_from_realized(thresholds[1] + 0.01, thresholds) == "high"
