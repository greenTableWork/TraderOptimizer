import threading
import time
from pathlib import Path

from trader_optimizer.batch import BatchItemResult, BatchSettings, optimize_candidates
from trader_optimizer.postgres import PostgresSettings
from trader_optimizer.strategy_configs import StrategyCandidate


def _candidate(name: str) -> StrategyCandidate:
    return StrategyCandidate(
        name=name,
        path=Path(f"{name}.json"),
        strategy_type="MovingAverageCross",
        config={
            "strategy_type": "MovingAverageCross",
            "contract": {"symbol": name.upper()},
        },
        symbols=(name.upper(),),
        variant="MovingAverageCross",
    )


def test_optimize_candidates_runs_candidates_concurrently_and_preserves_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_optimize_candidate(
        candidate: StrategyCandidate,
        settings: BatchSettings,
    ) -> BatchItemResult:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return BatchItemResult(
            name=candidate.name,
            strategy_type=candidate.strategy_type,
            variant=candidate.variant,
            symbols=candidate.symbols,
            source_config=str(candidate.path),
            status="ok",
            excess_return_pct=0.01,
        )

    monkeypatch.setattr(
        "trader_optimizer.batch._optimize_candidate",
        fake_optimize_candidate,
    )
    monkeypatch.setattr(
        "trader_optimizer.batch._write_batch_summary",
        lambda settings, results: None,
    )

    settings = BatchSettings(
        pg_settings=PostgresSettings(),
        optuna_storage_url="sqlite:///:memory:",
        output_dir=tmp_path,
        trials=1,
        max_bars=100,
        preferred_bar_size=None,
        train_fraction=0.7,
        verbose=False,
        workers=2,
    )

    results = optimize_candidates([_candidate("slow"), _candidate("fast")], settings)

    assert max_active == 2
    assert [result.name for result in results] == ["slow", "fast"]
