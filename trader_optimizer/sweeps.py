from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from trader_optimizer.postgres import (
    PostgresSettings,
    insert_optimizer_sweep_report,
    postgres_connection,
)


@dataclass(frozen=True)
class SweepCandidate:
    strategy_id: str
    total_return: float
    max_drawdown: float
    sharpe: float
    trade_count: int
    bars: int
    config: dict[str, object]


SweepCallable = Callable[[], tuple[list[SweepCandidate], list[SweepCandidate]]]


def choose_best(candidates: list[SweepCandidate]) -> SweepCandidate:
    profitable = [candidate for candidate in candidates if candidate.total_return > 0.0]
    pool = profitable or candidates
    return max(pool, key=lambda candidate: (candidate.total_return, candidate.sharpe, -candidate.trade_count))


def run_sweep_tasks(
    tasks: list[tuple[str, SweepCallable]],
    workers: int,
    verbose: bool = True,
) -> tuple[list[SweepCandidate], list[SweepCandidate]]:
    selected: list[SweepCandidate] = []
    all_candidates: list[SweepCandidate] = []
    workers = max(1, workers)
    if workers == 1:
        for name, task in tasks:
            task_selected, task_candidates = task()
            selected.extend(task_selected)
            all_candidates.extend(task_candidates)
            if verbose:
                print(f"Finished {name} sweep: selected {len(task_selected)}, candidates {len(task_candidates)}")
        return selected, all_candidates

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task): name for name, task in tasks}
        for future in as_completed(futures):
            name = futures[future]
            task_selected, task_candidates = future.result()
            selected.extend(task_selected)
            all_candidates.extend(task_candidates)
            if verbose:
                print(f"Finished {name} sweep: selected {len(task_selected)}, candidates {len(task_candidates)}")
    return selected, all_candidates


def write_sweep_report(
    pg_settings: PostgresSettings,
    report_name: str,
    candidates: list[SweepCandidate],
    selected: list[SweepCandidate],
) -> None:
    with postgres_connection(pg_settings) as conn:
        insert_optimizer_sweep_report(conn, report_name, candidates, selected)
