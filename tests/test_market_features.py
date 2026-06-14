from pathlib import Path

from trader_optimizer.data import Bar
from trader_optimizer.market_features import (
    OptionTrade,
    build_market_feature_summary,
    build_options_probability_map,
)
from trader_optimizer.strategy_configs import StrategyCandidate
from trader_optimizer.tuning_profile import build_candidate_tuning_profile


def _bar(index: int, close: float, volume: float) -> Bar:
    return Bar(
        timestamp_utc=f"2026-05-26T13:{30 + index:02d}:00+00:00",
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
    )


def test_market_feature_summary_computes_direction_volatility_and_volume() -> None:
    bars = [
        _bar(0, 100.0, 100.0),
        _bar(1, 102.0, 125.0),
        _bar(2, 105.0, 160.0),
        _bar(3, 109.0, 240.0),
    ]

    summary = build_market_feature_summary({"AAPL": bars})

    direction = summary["direction"]["bySymbol"]["AAPL"]
    assert direction["predictedDirection"] == "up"
    assert direction["curveSlopeSeverity"] >= 3
    assert summary["volatility"]["bySymbol"]["AAPL"]["regime"] in {"low", "medium", "high"}
    volume = summary["volume"]["bySymbol"]["AAPL"]
    assert volume["relativeVolume"] > 1.0
    assert volume["predictedDirection"] == "up"
    assert volume["confidence"] > 0.0
    assert summary["tradeVolumeOrderbook"]["fusionDirection"] == "up"
    assert summary["tradeVolumeOrderbook"]["orderbookStatus"] == "awaiting_l2_orderbook_ingestion"


def test_market_feature_summary_compares_index_futures_direction() -> None:
    bars = [
        _bar(0, 100.0, 100.0),
        _bar(1, 103.0, 120.0),
        _bar(2, 106.0, 140.0),
    ]
    futures_bars = [
        _bar(0, 18_000.0, 1000.0),
        _bar(1, 18_040.0, 1200.0),
        _bar(2, 18_090.0, 1500.0),
    ]

    summary = build_market_feature_summary(
        {"AAPL": bars},
        index_futures_bars={"NQ": futures_bars},
    )

    futures = summary["indexFuturesDirection"]
    assert futures["status"] == "active"
    assert futures["aggregate"]["predictedDirection"] == "up"
    assert futures["instrumentAlignment"]["alignment"] == "aligned"
    assert summary["volatility"]["marketByFuture"]["NQ"]["predictedDirection"] == "up"


def test_options_probability_map_builds_3d_direction_cells() -> None:
    cells = build_options_probability_map(
        [
            OptionTrade(
                underlying="AAPL",
                trade_time_utc="2026-05-26T14:05:00+00:00",
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
                trade_time_utc="2026-05-26T14:09:00+00:00",
                expiration_days=14,
                strike_moneyness=1.02,
                side="PUT",
                premium=2_000,
                volume=20,
                open_interest=300,
                implied_volatility=0.30,
            ),
        ]
    )

    assert cells["status"] == "active"
    assert cells["axes"] == ["expiration_days", "strike_moneyness", "trade_time_bucket"]
    assert len(cells["cells"]) == 1
    cell = cells["cells"][0]
    assert cell["upProbability"] > cell["downProbability"]
    assert cell["momentumDirection"] == "up"
    assert cell["momentumProbability"] > 0.0
    assert cells["aggregateByUnderlying"]["AAPL"]["momentumDirection"] == "up"


def test_tuning_profile_embeds_market_feature_evidence() -> None:
    bars = [_bar(0, 100.0, 100.0), _bar(1, 104.0, 160.0), _bar(2, 108.0, 220.0)]
    candidate = StrategyCandidate(
        name="tsla_ema",
        path=Path("tsla_ema.json"),
        strategy_type="TechnicalSignal",
        config={
            "strategy_type": "TechnicalSignal",
            "signal_type": "TS-002",
            "contract": {
                "symbol": "TSLA",
                "secType": "STOCK",
                "currency": "USD",
                "exchange": "IBKR",
            },
        },
        symbols=("TSLA",),
        variant="TS-002",
    )

    profile = build_candidate_tuning_profile(
        candidate,
        tuned_fields=["fastWindow", "slowWindow", "orderQuantity"],
        data_profiles={"TSLA": {"what_to_show": "TRADES"}},
        market_features=build_market_feature_summary({"TSLA": bars}),
    )

    direction = profile["tunedFor"]["direction"]
    assert direction["observedDirection"] == "up"
    assert direction["observedCurveSlopeSeverity"] >= 3
    trade_volume = profile["tunedFor"]["tradeVolumeOrderbook"]["tradeVolume"]
    assert trade_volume["status"] == "active"
    assert trade_volume["evidence"]["aggregate"]["predictedDirection"] == "up"
    assert profile["tunedFor"]["tradeVolumeOrderbook"]["evidence"]["fusionDirection"] == "up"
    assert (
        profile["tunedFor"]["tradeVolumeOrderbook"]["orderbook"]["evidenceStatus"]
        == "awaiting_l2_orderbook_ingestion"
    )
