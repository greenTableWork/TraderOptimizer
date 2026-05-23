from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
    report_path: Path,
    candidates: list[SweepCandidate],
    selected: list[SweepCandidate],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selected_keys = {
        candidate.strategy_id: json.dumps(candidate.config, sort_keys=True)
        for candidate in selected
    }
    rows = []
    for candidate in candidates:
        config_json = json.dumps(candidate.config, sort_keys=True)
        rows.append(
            {
                "strategy_id": candidate.strategy_id,
                "selected": config_json == selected_keys[candidate.strategy_id],
                "total_return": candidate.total_return,
                "max_drawdown": candidate.max_drawdown,
                "sharpe": candidate.sharpe,
                "trade_count": candidate.trade_count,
                "bars": candidate.bars,
                "config": config_json,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["strategy_id"]),
            not bool(row["selected"]),
            -float(row["total_return"]),
        )
    )
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy_id",
                "selected",
                "total_return",
                "max_drawdown",
                "sharpe",
                "trade_count",
                "bars",
                "config",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
