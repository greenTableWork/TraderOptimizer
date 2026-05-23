from pathlib import Path

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


def test_write_sweep_report_marks_selected(tmp_path: Path) -> None:
    selected = [_candidate("A", 0.02)]
    candidates = [_candidate("A", 0.01), selected[0]]
    report_path = tmp_path / "report.csv"

    write_sweep_report(report_path, candidates, selected)

    text = report_path.read_text()
    assert "strategy_id,selected,total_return" in text
    assert "A,True,0.02" in text
    assert "A,False,0.01" in text
