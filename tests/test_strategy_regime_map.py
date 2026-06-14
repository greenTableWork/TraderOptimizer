import json
from pathlib import Path

from trader_optimizer.live_regime import (
    load_strategy_regime_candidates,
    select_strategy_for_regime,
)
from trader_optimizer.regime_tuning_universe import regime_cell_id
from trader_optimizer.strategy_regime_map import (
    build_strategy_regime_map_from_run_summary,
    summarize_strategy_regime_map,
    write_strategy_regime_map_jsonl,
)


def test_build_strategy_regime_map_from_universe_run_summary(tmp_path: Path) -> None:
    cell = {
        "directionSign": "up",
        "instrumentVolatilityRegime": "medium",
        "marketVolatilityRegime": "low",
        "volumeRegime": "high",
    }
    task_id = "aapl_mac_up_medium_low_high"
    universe = tmp_path / "universe.jsonl"
    universe.write_text(
        json.dumps(
            {
                "taskId": task_id,
                "symbol": "AAPL",
                "strategyName": "mac_aapl",
                "strategyType": "MovingAverageCross",
                "variant": "MovingAverageCross",
                "strategySymbols": ["AAPL"],
                "sourceConfig": "/configs/mac_aapl.json",
                "startUtc": "2026-05-29T13:30:00+00:00",
                "endUtc": "2026-05-29T19:59:50+00:00",
                "durationSeconds": 23400,
                "durationBars": 2340,
                "barSize": "10 secs",
                "whatToShow": "TRADES",
                "useRth": 1,
                "regimeCellId": regime_cell_id(cell),
                "regimeCell": cell,
                "regimeDimensions": list(cell),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_dir = tmp_path / "task"
    strategy_dir = task_dir / "mac_aapl"
    strategy_dir.mkdir(parents=True)
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    exported_config = export_dir / "mac_aapl.optimized.json"
    exported_config.write_text("{}", encoding="utf-8")
    (export_dir / "index.json").write_text(
        json.dumps(
            {
                "configs": [
                    {
                        "name": "mac_aapl",
                        "exported_config": str(exported_config),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    best_summary = strategy_dir / "best_summary.json"
    best_summary.write_text(
        json.dumps(
            {
                "backtester": {
                    "benchmarks": {
                        "positive_return": {"passed": True},
                        "same_stock_buy_and_hold": {
                            "passed": True,
                            "excess_return_pct": 0.02,
                        },
                        "spx_buy_and_hold": {
                            "passed": True,
                            "excess_return_pct": 0.01,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    batch_summary = task_dir / "batch_summary.json"
    batch_summary.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "name": "mac_aapl",
                        "strategy_type": "MovingAverageCross",
                        "variant": "MovingAverageCross",
                        "symbols": ["AAPL"],
                        "source_config": "/configs/mac_aapl.json",
                        "status": "ok",
                        "best_config": str(strategy_dir / "best_config.json"),
                        "summary": str(best_summary),
                        "best_value": 0.03,
                        "strategy_return_pct": 0.04,
                        "benchmark_return_pct": 0.02,
                        "excess_return_pct": 0.02,
                        "tuning_profile": {
                            "categoryLabels": ["direction"],
                            "optimizedFor": {"primaryObjective": "test"},
                        },
                    },
                    {
                        "name": "mac_failed",
                        "status": "benchmark_failed",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    run_summary = tmp_path / "run_summary.json"
    run_summary.write_text(
        json.dumps(
            {
                "schema": "regime_tuning_universe_run.v1",
                "universe": str(universe),
                "results": [
                    {
                        "task_id": task_id,
                        "symbol": "AAPL",
                        "strategy_name": "mac_aapl",
                        "summary_path": str(batch_summary),
                        "command": [
                            "trader-optimizer",
                            "optimize-existing",
                            "--export-config-dir",
                            str(export_dir),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    entries = build_strategy_regime_map_from_run_summary(run_summary, cwd=tmp_path)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["schema"] == "strategy_regime_map.v1"
    assert entry["symbol"] == "AAPL"
    assert entry["configPath"] == str(exported_config)
    assert entry["validationStatus"] == "ok"
    assert entry["sameStockExcessReturnPct"] == 0.02
    assert entry["spxExcessReturnPct"] == 0.01
    assert entry["mapScore"] == 0.016666666666666666
    summary = summarize_strategy_regime_map(entries, run_summary_path=run_summary)
    assert summary["entries"] == 1
    assert summary["statusCounts"] == {"ok": 1}


def test_strategy_regime_map_jsonl_feeds_live_selector(tmp_path: Path) -> None:
    cell = {
        "directionSign": "up",
        "instrumentVolatilityRegime": "medium",
        "marketVolatilityRegime": "low",
        "volumeRegime": "high",
    }
    map_path = tmp_path / "strategy_map.jsonl"
    write_strategy_regime_map_jsonl(
        map_path,
        [
            {
                "symbol": "AAPL",
                "strategyName": "mac_aapl",
                "configPath": "/configs/mac_aapl.optimized.json",
                "regimeCellId": regime_cell_id(cell),
                "regimeCell": cell,
                "validationStatus": "ok",
                "sameStockExcessReturnPct": 0.02,
                "spxExcessReturnPct": 0.01,
            }
        ],
    )

    selection = select_strategy_for_regime(
        symbol="AAPL",
        active_regime_cell_id=regime_cell_id(cell),
        active_regime_cell=cell,
        candidates=load_strategy_regime_candidates(map_path),
    )

    assert selection["status"] == "exact_match"
    assert selection["selected"]["configPath"] == "/configs/mac_aapl.optimized.json"
