from pathlib import Path

from trader_optimizer.strategy_configs import StrategyCandidate
from trader_optimizer.tuning_profile import (
    DEFAULT_CURVE_SLOPE_SEVERITY,
    ORDERBOOK_INTEGRATION_BRANCH,
    build_candidate_tuning_profile,
    direction_plan_label,
    tuning_category_labels,
)


def _candidate(
    *,
    name: str = "ts005_tsla",
    strategy_type: str = "TechnicalSignal",
    variant: str = "TS-005",
    symbol: str = "TSLA",
    sec_type: str = "STOCK",
) -> StrategyCandidate:
    return StrategyCandidate(
        name=name,
        path=Path(f"{name}.json"),
        strategy_type=strategy_type,
        variant=variant,
        symbols=(symbol,),
        config={
            "strategy_type": strategy_type,
            "signal_type": variant,
            "contract": {
                "symbol": symbol,
                "secType": sec_type,
                "currency": "USD",
                "exchange": "IBKR",
            },
        },
    )


def test_tuning_profile_covers_requested_signal_categories() -> None:
    profile = build_candidate_tuning_profile(
        _candidate(),
        tuned_fields=["rsiPeriod", "divergenceLookback", "orderQuantity"],
        data_profiles={
            "TSLA": {
                "bar_size": "10 secs",
                "what_to_show": "TRADES",
                "use_rth": 1,
            }
        },
        hyperparameters={"rsiPeriod": 14, "divergenceLookback": 20},
        strategy_budget=30_000,
    )

    assert profile["schema"] == "strategy_tuning_profile.v1"
    assert tuning_category_labels(profile) == [
        "direction",
        "volatility",
        "indexFuturesDirection",
        "optionsProbabilityMap3d",
        "tradeVolumeOrderbook",
    ]
    direction = profile["tunedFor"]["direction"]
    assert direction["expectedDirection"] == "up_or_down"
    assert direction["curveSlopeSeverity"] == DEFAULT_CURVE_SLOPE_SEVERITY
    assert "slope 3" in direction_plan_label(profile)
    assert profile["tunedFor"]["indexFuturesDirection"]["instruments"][0][
        "candidateIndexFutures"
    ] == ["NQ", "ES"]
    assert profile["tunedFor"]["optionsProbabilityMap3d"]["probabilityMap3d"][
        "predictionTargets"
    ] == ["direction", "momentum"]
    assert (
        profile["tunedFor"]["tradeVolumeOrderbook"]["orderbook"]["branch"]
        == ORDERBOOK_INTEGRATION_BRANCH
    )


def test_tuning_profile_marks_crypto_futures_and_options_not_applicable() -> None:
    profile = build_candidate_tuning_profile(
        _candidate(
            name="btc_mac",
            strategy_type="MovingAverageCross",
            variant="MovingAverageCross",
            symbol="BTC",
            sec_type="CRYPTO",
        ),
        tuned_fields=["fastWindow", "slowWindow", "orderQuantity"],
    )

    assert profile["tunedFor"]["indexFuturesDirection"]["status"] == "not_applicable"
    assert profile["tunedFor"]["optionsProbabilityMap3d"]["status"] == "not_applicable"
    assert "indexFuturesDirection" not in tuning_category_labels(profile)
    assert "optionsProbabilityMap3d" not in tuning_category_labels(profile)


def test_tuning_profile_uses_configured_slope_severity() -> None:
    candidate = _candidate()
    candidate.config["curveSlopeSeverity"] = 5

    profile = build_candidate_tuning_profile(
        candidate,
        tuned_fields=["fastWindow", "slowWindow"],
    )

    assert profile["tunedFor"]["direction"]["curveSlopeSeverity"] == 5


def test_tuning_profile_records_concrete_tuning_regions() -> None:
    profile = build_candidate_tuning_profile(
        _candidate(symbol="AAPL"),
        tuned_fields=["fastWindow", "slowWindow"],
        tuning_regions=[
            {
                "regionId": "tuning-region-v1:abc123",
                "role": "train",
                "tuningSubcategory": "direction",
                "symbol": "AAPL",
                "period": "week",
                "category": "up_slope_4",
                "barSize": "10 secs",
                "whatToShow": "TRADES",
                "useRth": 1,
                "startUtc": "2026-05-04T13:30:00+00:00",
                "endUtc": "2026-05-08T19:59:50+00:00",
                "backtestRegion": {"ignored": True},
            },
            {
                "regionId": "tuning-region-v1:abc123",
                "role": "train",
                "tuningSubcategory": "direction",
                "symbol": "AAPL",
                "period": "week",
                "category": "up_slope_4",
            },
        ],
    )

    assert profile["tuningRegions"] == [
        {
            "regionId": "tuning-region-v1:abc123",
            "role": "train",
            "tuningSubcategory": "direction",
            "symbol": "AAPL",
            "period": "week",
            "category": "up_slope_4",
            "barSize": "10 secs",
            "whatToShow": "TRADES",
            "useRth": 1,
            "startUtc": "2026-05-04T13:30:00+00:00",
            "endUtc": "2026-05-08T19:59:50+00:00",
        }
    ]
    assert profile["optimizedFor"]["backtesterGates"] == [
        "positive_strategy_return",
        "beat_spx_same_window",
        "beat_same_stock_buy_and_hold",
    ]
