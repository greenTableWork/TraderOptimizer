import json
from pathlib import Path

from trader_optimizer.cli import main
from trader_optimizer.live_regime import (
    StrategyRegimeCandidate,
    build_live_regime_detections,
    build_live_regime_detection,
    regime_state_key,
    select_strategy_for_regime,
)
from trader_optimizer.regime_tuning_universe import regime_cell_id


def _vector(
    direction: str,
    *,
    symbol: str = "AAPL",
    instrument_vol: str = "medium",
    market_vol: str = "medium",
    volume: str = "high",
    change_point: float = 0.0,
) -> dict[str, object]:
    return {
        "schema": "instrument_regime_vector.v1",
        "symbol": symbol,
        "barSize": "10 secs",
        "whatToShow": "TRADES",
        "useRth": 1,
        "startUtc": "2026-05-29T13:30:00+00:00",
        "endUtc": "2026-05-29T19:59:50+00:00",
        "core": {
            "directionSign": direction,
            "slopeSeverity": 3,
            "instrumentVolatilityRegime": instrument_vol,
            "marketVolatilityRegime": market_vol,
            "indexFuturesAlignment": "aligned",
            "volumeRegime": volume,
            "volumeDirection": direction if direction in {"up", "down"} else "neutral",
        },
        "advanced": {
            "regimePersistence": {"status": "active", "score": 0.75},
            "covarianceStress": {"status": "active", "regime": "normal"},
            "momentumHorizonRegime": {"status": "active", "regime": "continuation_up"},
            "liquidityOrderFlowRegime": {"status": "volume_only", "regime": volume},
            "optionsSurfaceRegime": {"status": "not_loaded", "regime": "not_loaded"},
            "distributionClusterId": {"status": "active", "clusterId": "vol_medium_balanced"},
            "changePointConfidence": {
                "status": "active",
                "confidence": change_point,
                "direction": "stable",
            },
        },
    }


def test_live_regime_detection_includes_inter_world_contract() -> None:
    detection = build_live_regime_detection(
        _vector("up"),
        generated_utc="2026-06-01T00:00:00+00:00",
    )

    assert detection["schema"] == "live_regime_detection.v1"
    assert detection["transition"]["status"] == "initialized"
    assert detection["strategySelection"]["status"] == "no_validated_config"
    inter_worlds = detection["interWorlds"]
    assert inter_worlds["historicalOptimizerWorld"]["requiresBacktesterGatedConfig"]
    assert inter_worlds["liveRuntimeWorld"]["blockedActions"] == [
        "live_optimization",
        "ungated_config_promotion",
    ]
    detector_ids = {
        spec["detectorId"]
        for spec in inter_worlds["corpusWorld"]["detectors"]
    }
    assert "direction_slope_persistence" in detector_ids
    assert "futures_alignment" in detector_ids
    assert "liquidity_orderflow" in detector_ids


def test_live_regime_hysteresis_holds_then_switches() -> None:
    detections, states = build_live_regime_detections(
        [_vector("up")],
        generated_utc="2026-06-01T00:00:00+00:00",
    )
    state = states[regime_state_key(_vector("up"))]

    detections, states = build_live_regime_detections(
        [_vector("down"), _vector("down"), _vector("down")],
        previous_states={regime_state_key(_vector("up")): state},
        min_persistence=3,
        generated_utc="2026-06-01T00:01:00+00:00",
    )

    assert [item["transition"]["status"] for item in detections] == [
        "held",
        "held",
        "switched",
    ]
    assert states[regime_state_key(_vector("up"))]["activeRegimeCell"]["directionSign"] == "down"


def test_live_regime_change_point_can_override_hysteresis() -> None:
    first = build_live_regime_detection(
        _vector("up"),
        generated_utc="2026-06-01T00:00:00+00:00",
    )
    second = build_live_regime_detection(
        _vector("down", change_point=0.95),
        previous_state=first["state"],
        min_persistence=5,
        change_point_threshold=0.80,
        generated_utc="2026-06-01T00:01:00+00:00",
    )

    assert second["transition"]["status"] == "switched"
    assert second["transition"]["reason"] == "change_point_override"


def test_select_strategy_for_regime_prefers_exact_validated_match() -> None:
    cell = {
        "directionSign": "up",
        "instrumentVolatilityRegime": "medium",
        "marketVolatilityRegime": "medium",
        "volumeRegime": "high",
    }
    selection = select_strategy_for_regime(
        symbol="AAPL",
        active_regime_cell_id=regime_cell_id(cell),
        active_regime_cell=cell,
        candidates=[
            StrategyRegimeCandidate(
                symbol="AAPL",
                strategy_name="mac_aapl",
                config_path="mac.json",
                regime_cell_id=regime_cell_id(cell),
                regime_cell=cell,
                validation_status="ok",
                same_stock_excess_return_pct=0.02,
            )
        ],
    )

    assert selection["status"] == "exact_match"
    assert selection["selected"]["strategyName"] == "mac_aapl"


def test_detect_live_regimes_cli_writes_shadow_outputs(tmp_path: Path) -> None:
    vectors = tmp_path / "vectors.jsonl"
    output = tmp_path / "detections.jsonl"
    summary = tmp_path / "summary.json"
    state = tmp_path / "state.json"
    vectors.write_text(json.dumps(_vector("up")) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "detect-live-regimes",
            "--regime-vectors",
            str(vectors),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--state-output",
            str(state),
            "--quiet",
        ]
    )

    assert exit_code == 0
    detection = json.loads(output.read_text(encoding="utf-8"))
    assert detection["mode"] == "shadow"
    assert json.loads(summary.read_text(encoding="utf-8"))["detections"] == 1
    assert regime_state_key(_vector("up")) in json.loads(
        state.read_text(encoding="utf-8")
    )["states"]
