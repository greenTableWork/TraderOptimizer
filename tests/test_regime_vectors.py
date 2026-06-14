import json
from pathlib import Path

from trader_optimizer.data import Bar, DataProfile
from trader_optimizer.regime_vectors import (
    ADVANCED_REGIME_PARAMETERS,
    REGIME_VECTOR_SCHEMA,
    RegimeVectorContext,
    build_regime_vector,
    write_regime_vector_summary,
    write_regime_vectors_jsonl,
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


def _profile(symbol: str) -> DataProfile:
    return DataProfile(
        symbol=symbol,
        bar_size="10 secs",
        what_to_show="TRADES",
        use_rth=1,
        count=4,
        first_timestamp="2026-05-04T13:30:00+00:00",
        last_timestamp="2026-05-04T20:00:00+00:00",
    )


def test_build_regime_vector_includes_core_and_advanced_parameters() -> None:
    bars = [
        _bar("2026-05-04T13:30:00+00:00", 100.0, 100.0),
        _bar("2026-05-04T14:30:00+00:00", 102.0, 110.0),
        _bar("2026-05-04T15:30:00+00:00", 104.0, 150.0),
        _bar("2026-05-04T20:00:00+00:00", 108.0, 200.0),
    ]
    market_bars = [
        _bar("2026-05-04T13:30:00+00:00", 1000.0, 1000.0),
        _bar("2026-05-04T14:30:00+00:00", 1002.0, 1000.0),
        _bar("2026-05-04T15:30:00+00:00", 1004.0, 1000.0),
        _bar("2026-05-04T20:00:00+00:00", 1008.0, 1000.0),
    ]

    vector = build_regime_vector(
        RegimeVectorContext(
            symbol="AAPL",
            bars=bars,
            profile=_profile("AAPL"),
            market_symbol="SPX",
            market_bars=market_bars,
            futures_symbol="ES",
            futures_bars=market_bars,
            slope_severity_thresholds=(0.05, 0.10, 0.20, 0.40),
            volatility_regime_thresholds=(0.05, 0.20),
            market_volatility_regime_thresholds=(0.001, 0.01),
        ),
        generated_utc="2026-06-01T00:00:00+00:00",
    )

    assert vector["schema"] == REGIME_VECTOR_SCHEMA
    assert vector["core"]["directionSign"] == "up"
    assert vector["core"]["slopeSeverity"] == 2
    assert vector["core"]["instrumentVolatilityRegime"] == "low"
    assert vector["core"]["marketVolatilityRegime"] == "medium"
    assert vector["core"]["indexFuturesAlignment"] == "aligned"
    assert vector["advancedParameters"] == list(ADVANCED_REGIME_PARAMETERS)
    assert vector["advanced"]["optionsSurfaceRegime"]["status"] == "not_loaded"
    assert vector["advanced"]["liquidityOrderFlowRegime"]["status"] == "volume_only"


def test_write_regime_vector_artifacts(tmp_path: Path) -> None:
    vector = {
        "schema": REGIME_VECTOR_SCHEMA,
        "symbol": "AAPL",
        "core": {
            "directionSign": "up",
            "instrumentVolatilityRegime": "low",
            "marketVolatilityRegime": "not_loaded",
            "indexFuturesAlignment": "not_loaded",
            "volumeRegime": "normal",
        },
    }
    output_path = tmp_path / "vectors.jsonl"
    summary_path = tmp_path / "summary.json"

    write_regime_vectors_jsonl(output_path, [vector])
    write_regime_vector_summary(summary_path, [vector])

    assert json.loads(output_path.read_text())["symbol"] == "AAPL"
    summary = json.loads(summary_path.read_text())
    assert summary["vectorSchema"] == REGIME_VECTOR_SCHEMA
    assert summary["vectors"] == 1
    assert summary["counts"]["directionSign"] == {"up": 1}
