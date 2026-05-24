from trader_optimizer.sweeps import (
    SweepCandidate,
    choose_best,
    run_sweep_tasks,
    write_sweep_report,
)


def _candidate(strategy_id: str, total_return: float, trades: int = 1) -> SweepCandidate:
    return SweepCandidate(
        strategy_id=strategy_id,
        total_return=total_return,
        max_drawdown=-0.01,
        sharpe=total_return * 10,
        trade_count=trades,
        bars=10,
        config={"return": total_return},
    )


def test_choose_best_prefers_profitable_return() -> None:
    best = choose_best(
        [
            _candidate("A", -0.01),
            _candidate("A", 0.02),
            _candidate("A", 0.01),
        ]
    )

    assert best.total_return == 0.02


def test_run_sweep_tasks_supports_parallel_workers() -> None:
    tasks = [
        ("one", lambda: ([_candidate("A", 0.01)], [_candidate("A", 0.01)])),
        ("two", lambda: ([_candidate("B", 0.02)], [_candidate("B", 0.02)])),
    ]

    selected, candidates = run_sweep_tasks(tasks, workers=2, verbose=False)

    assert {candidate.strategy_id for candidate in selected} == {"A", "B"}
    assert len(candidates) == 2


def test_write_sweep_report_persists_to_postgres(monkeypatch) -> None:
    class FakePostgresContext:
        def __enter__(self):
            return "connection"

        def __exit__(self, exc_type, exc, traceback):
            return False

    calls = []

    def fake_insert(conn, report_name, candidates, selected):
        calls.append((conn, report_name, candidates, selected))

    monkeypatch.setattr(
        "trader_optimizer.sweeps.postgres_connection",
        lambda settings: FakePostgresContext(),
    )
    monkeypatch.setattr(
        "trader_optimizer.sweeps.insert_optimizer_sweep_report",
        fake_insert,
    )

    selected = [_candidate("A", 0.02)]
    candidates = [_candidate("A", 0.01), selected[0]]

    write_sweep_report(object(), "report", candidates, selected)

    assert calls == [("connection", "report", candidates, selected)]
