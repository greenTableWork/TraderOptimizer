import json
import threading
import time
from pathlib import Path

import pytest
from optuna.trial import FixedTrial

from trader_optimizer.batch import (
    BatchItemResult,
    BatchSettings,
    QuantitySearchSpace,
    optimize_candidates,
    _export_best_configs,
    _optimize_candidate,
    _quantity_search_space,
    _single_signal_config_and_result,
    _quantity_upper_bound,
)
from trader_optimizer.data import Bar
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
    precreated_studies: list[str] = []
    start_snapshots: list[list[str]] = []

    def fake_optimize_candidate(
        candidate: StrategyCandidate,
        settings: BatchSettings,
    ) -> BatchItemResult:
        nonlocal active, max_active
        start_snapshots.append(list(precreated_studies))
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
        "trader_optimizer.batch.create_or_load_study",
        lambda **kwargs: precreated_studies.append(str(kwargs["study_name"])),
    )
    monkeypatch.setattr(
        "trader_optimizer.batch._write_batch_summary",
        lambda settings, results: None,
    )

    settings = BatchSettings(
        pg_settings=PostgresSettings(),
        optuna_storage_url="sqlite:///:memory:",
        output_dir=tmp_path / "parallel_test",
        trials=1,
        max_bars=100,
        preferred_bar_size=None,
        train_fraction=0.7,
        verbose=False,
        workers=2,
    )

    results = optimize_candidates([_candidate("slow"), _candidate("fast")], settings)

    assert precreated_studies == [
        "parallel_test_slow_simple",
        "parallel_test_fast_simple",
    ]
    assert start_snapshots
    assert all(snapshot == precreated_studies for snapshot in start_snapshots)
    assert max_active == 2
    assert [result.name for result in results] == ["slow", "fast"]


def test_optimize_candidate_retries_transient_storage_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    attempts = 0

    def fake_run_candidate_optimization(
        candidate: StrategyCandidate,
        settings: BatchSettings,
    ) -> BatchItemResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("deadlock detected")
        return BatchItemResult(
            name=candidate.name,
            strategy_type=candidate.strategy_type,
            variant=candidate.variant,
            symbols=candidate.symbols,
            source_config=str(candidate.path),
            status="ok",
        )

    monkeypatch.setattr(
        "trader_optimizer.batch._run_candidate_optimization",
        fake_run_candidate_optimization,
    )
    monkeypatch.setattr("trader_optimizer.batch.time.sleep", lambda _: None)
    settings = BatchSettings(
        pg_settings=PostgresSettings(),
        optuna_storage_url="sqlite:///:memory:",
        output_dir=tmp_path,
        trials=1,
        max_bars=100,
        preferred_bar_size=None,
        train_fraction=0.7,
        verbose=False,
        workers=1,
    )

    result = _optimize_candidate(_candidate("retry"), settings)

    assert attempts == 2
    assert result.status == "ok"


def test_quantity_upper_bound_uses_strategy_budget() -> None:
    bars = [
        Bar(
            timestamp_utc="2026-05-26T13:30:00+00:00",
            open=100,
            high=101,
            low=99,
            close=125,
        )
    ]

    assert _quantity_upper_bound(bars, 30_000) == 240
    assert _quantity_upper_bound(bars, None) == 20


def test_quantity_search_space_allows_fractional_crypto_under_budget() -> None:
    bars = [
        Bar(
            timestamp_utc="2026-05-26T13:30:00+00:00",
            open=100_000,
            high=101_000,
            low=99_000,
            close=100_000,
        )
    ]
    candidate = StrategyCandidate(
        name="btc",
        path=Path("btc.json"),
        strategy_type="MovingAverageCross",
        config={
            "strategy_type": "MovingAverageCross",
            "contract": {
                "symbol": "BTC",
                "secType": "CRYPTO",
                "currency": "USD",
                "exchange": "IBKR",
            },
        },
        symbols=("BTC",),
        variant="MovingAverageCross",
    )

    search_space = _quantity_search_space(candidate, bars, 30_000)

    assert search_space.fractional
    assert search_space.upper_bound == pytest.approx(0.3)
    assert 0 < search_space.lower_bound <= search_space.upper_bound


def test_triple_moving_average_optimizer_keeps_middle_window_ordered() -> None:
    bars = [
        Bar(
            timestamp_utc=f"2026-05-26T13:{minute:02d}:00+00:00",
            open=100 + minute,
            high=101 + minute,
            low=99 + minute,
            close=100 + minute,
        )
        for minute in range(40)
    ]
    candidate = StrategyCandidate(
        name="matrend002_btc",
        path=Path("matrend002_btc.json"),
        strategy_type="MovingAverageCross",
        config={
            "strategy_type": "MovingAverageCross",
            "trendMode": "MATREND-002",
            "contract": {
                "symbol": "BTC",
                "secType": "CRYPTO",
                "currency": "USD",
                "exchange": "IBKR",
            },
        },
        symbols=("BTC",),
        variant="MovingAverageCross",
    )

    config, _ = _single_signal_config_and_result(
        candidate,
        bars,
        FixedTrial(
            {
                "orderQuantity": 0.1,
                "fastWindow": 5,
                "middleWindow": 12,
                "slowWindow": 30,
            }
        ),
        QuantitySearchSpace(
            lower_bound=0.01,
            upper_bound=0.5,
            fractional=True,
        ),
    )

    assert config["fastWindow"] == 5
    assert config["middleWindow"] == 12
    assert config["slowWindow"] == 30


def test_export_best_configs_includes_tuning_profile(tmp_path: Path) -> None:
    best_config = tmp_path / "best_config.json"
    best_config.write_text('{"strategy_type":"MovingAverageCross"}', encoding="utf-8")
    tuning_profile = {
        "schema": "strategy_tuning_profile.v1",
        "tunedFor": {
            "direction": {
                "expectedDirection": "up",
                "curveSlopeSeverity": 3,
            }
        },
    }

    _export_best_configs(
        tmp_path / "exports",
        [
            BatchItemResult(
                name="mac_aapl",
                strategy_type="MovingAverageCross",
                variant="MovingAverageCross",
                symbols=("AAPL",),
                source_config="source.json",
                status="ok",
                best_config=str(best_config),
                tuning_profile=tuning_profile,
            )
        ],
    )

    index = json.loads((tmp_path / "exports" / "index.json").read_text())
    assert index["configs"][0]["tuning_profile"] == tuning_profile
