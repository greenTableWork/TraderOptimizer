import json
from pathlib import Path

from trader_optimizer.regime_tuning_universe import (
    REGIME_TUNING_TASK_SCHEMA,
    build_regime_tuning_tasks,
    load_regime_vectors_jsonl,
    regime_cell_for_vector,
    summarize_regime_tuning_universe,
)
from trader_optimizer.strategy_configs import StrategyCandidate


def _candidate(name: str, symbol: str) -> StrategyCandidate:
    return StrategyCandidate(
        name=name,
        path=Path(f"/tmp/trader/{name}.json"),
        strategy_type="MovingAverageCross",
        config={"strategy_type": "MovingAverageCross", "contract": {"symbol": symbol}},
        symbols=(symbol,),
        variant="MovingAverageCross",
    )


def _vector(symbol: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "startUtc": "2026-05-29T13:30:00+00:00",
        "endUtc": "2026-05-29T20:00:00+00:00",
        "bars": 10,
        "barSize": "10 secs",
        "whatToShow": "TRADES",
        "useRth": 1,
        "core": {
            "directionSign": "up",
            "instrumentVolatilityRegime": "medium",
            "marketVolatilityRegime": "medium",
            "volumeRegime": "high",
        },
        "advanced": {},
    }


def test_build_regime_tuning_tasks_defaults_to_matching_symbol() -> None:
    tasks = build_regime_tuning_tasks(
        [_vector("AAPL")],
        [_candidate("mac_aapl", "AAPL"), _candidate("mac_msft", "MSFT")],
        trader_root=Path("/tmp/trader"),
        output_root=Path("/tmp/out"),
        export_root=Path("/tmp/export"),
    )

    assert len(tasks) == 1
    task = tasks[0]
    assert task["schema"] == REGIME_TUNING_TASK_SCHEMA
    assert task["symbol"] == "AAPL"
    assert task["strategyName"] == "mac_aapl"
    assert task["runnableWithExistingConfig"] is True
    assert task["requiresRetargeting"] is False
    assert "--start-utc" in task["optimizerCommand"]
    assert "mac_aapl.json" in task["optimizerCommand"]


def test_build_regime_tuning_tasks_can_emit_full_retargeting_universe() -> None:
    tasks = build_regime_tuning_tasks(
        [_vector("AAPL")],
        [_candidate("mac_aapl", "AAPL"), _candidate("mac_msft", "MSFT")],
        trader_root=Path("/tmp/trader"),
        output_root=Path("/tmp/out"),
        export_root=None,
        strategy_scope="all",
    )

    assert len(tasks) == 2
    assert [task["requiresRetargeting"] for task in tasks] == [False, True]
    summary = summarize_regime_tuning_universe(
        tasks,
        vector_count=1,
        strategy_count=2,
        strategy_scope="all",
    )
    assert summary["tasks"] == 2
    assert summary["runnableTasks"] == 1
    assert summary["requiresRetargetingTasks"] == 1


def test_regime_cell_and_jsonl_loader(tmp_path: Path) -> None:
    path = tmp_path / "vectors.jsonl"
    path.write_text(json.dumps(_vector("AAPL")) + "\n", encoding="utf-8")

    vectors = load_regime_vectors_jsonl(path)
    cell = regime_cell_for_vector(vectors[0])

    assert len(vectors) == 1
    assert cell == {
        "directionSign": "up",
        "instrumentVolatilityRegime": "medium",
        "marketVolatilityRegime": "medium",
        "volumeRegime": "high",
    }
