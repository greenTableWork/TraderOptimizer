from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trader_optimizer.config import write_json
from trader_optimizer.regime_tuning_universe import load_regime_vectors_jsonl


@dataclass(frozen=True)
class UniverseRunResult:
    task_id: str
    symbol: str
    strategy_name: str
    status: str
    return_code: int | None
    log_path: str | None
    summary_path: str | None
    command: list[str]

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = args.cwd.resolve()
    tasks = load_universe_tasks(args.universe)
    if not args.include_retargeting:
        tasks = [
            task
            for task in tasks
            if bool(task.get("runnableWithExistingConfig"))
        ]
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    jobs = resolve_jobs(args.jobs, len(tasks))
    log_dir = (
        args.log_dir
        or args.universe.parent / f"{args.universe.stem}_run_logs"
    ).resolve()
    status_output = (
        args.status_output
        or args.universe.with_name(f"{args.universe.stem}_run_summary.json")
    ).resolve()

    if not tasks:
        write_run_summary(status_output, [], args.universe, jobs)
        print(f"No tasks selected from {args.universe}")
        print(f"summary: {status_output}")
        return 0

    print(f"tasks: {len(tasks)}")
    print(f"jobs: {jobs}")
    print(f"inner_workers: {args.inner_workers}")
    print(f"log_dir: {log_dir}")
    print(f"status_output: {status_output}")
    if not args.skip_backtester_prebuild and not args.dry_run:
        built_backtesters = prebuild_backtesters_for_tasks(
            tasks,
            cwd=cwd,
            preset=args.backtester_preset,
        )
        if built_backtesters:
            print("prebuilt_backtesters:")
            for backtester in built_backtesters:
                print(f"  {backtester}")

    results = run_universe_tasks(
        tasks,
        cwd=cwd,
        jobs=jobs,
        log_dir=log_dir,
        inner_workers=args.inner_workers,
        dry_run=args.dry_run,
        strict_exit_codes=args.strict_exit_codes,
    )
    write_run_summary(status_output, results, args.universe, jobs)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print("status_counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"summary: {status_output}")
    failed = counts.get("failed", 0)
    if args.strict_exit_codes:
        failed += counts.get("completed_nonzero", 0)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run optimizer commands from a regime tuning universe JSONL.",
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel task processes. Default 0 uses local CPU count.",
    )
    parser.add_argument(
        "--inner-workers",
        type=int,
        default=1,
        help="Override each task's optimize-existing --workers value.",
    )
    parser.add_argument(
        "--include-retargeting",
        action="store_true",
        help=(
            "Also run tasks marked requiresRetargeting. Default runs only valid "
            "existing configs."
        ),
    )
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-backtester-prebuild",
        action="store_true",
        help=(
            "Do not build TraderCore BackTester once before launching optimizer "
            "tasks. By default the runner prebuilds once because generated task "
            "commands usually pass --skip-backtester-build to preserve concurrency."
        ),
    )
    parser.add_argument(
        "--backtester-preset",
        default="debug",
        help="TraderCore CMake preset/build directory used for the prebuild. Default: debug.",
    )
    parser.add_argument(
        "--strict-exit-codes",
        action="store_true",
        help=(
            "Treat optimize-existing exit code 1 as failure even when it produced "
            "batch_summary.json. By default that means completed_nonzero because "
            "benchmark-gated runs often exit 1 when nothing is promoted."
        ),
    )
    return parser


def load_universe_tasks(path: Path) -> list[dict[str, Any]]:
    return load_regime_vectors_jsonl(path)


def resolve_jobs(requested_jobs: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested_jobs > 0:
        return min(requested_jobs, task_count)
    return min(task_count, max(1, os.cpu_count() or 1))


def prebuild_backtesters_for_tasks(
    tasks: list[dict[str, Any]],
    *,
    cwd: Path,
    preset: str,
) -> list[Path]:
    built_backtesters: list[Path] = []
    for trader_root in trader_roots_for_tasks(tasks, cwd=cwd):
        trader_core_root = trader_root / "TraderCore"
        subprocess.run(
            ["cmake", "--build", "--preset", preset, "--target", "BackTester"],
            cwd=trader_core_root,
            check=True,
        )
        backtester = trader_core_root / "build" / preset / "BackTesting" / "BackTester"
        if not backtester.exists():
            raise FileNotFoundError(f"BackTester build did not create {backtester}")
        built_backtesters.append(backtester.resolve())
    return built_backtesters


def trader_roots_for_tasks(tasks: list[dict[str, Any]], *, cwd: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for task in tasks:
        command = task.get("optimizerCommand")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            continue
        root_value = command_option_value(command, "--trader-root")
        if root_value is None:
            continue
        root = Path(root_value)
        if not root.is_absolute():
            root = cwd / root
        root = root.resolve()
        if root not in seen:
            roots.append(root)
            seen.add(root)
    return roots


def command_option_value(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def run_universe_tasks(
    tasks: list[dict[str, Any]],
    *,
    cwd: Path,
    jobs: int,
    log_dir: Path,
    inner_workers: int,
    dry_run: bool,
    strict_exit_codes: bool,
) -> list[UniverseRunResult]:
    log_dir.mkdir(parents=True, exist_ok=True)
    results_by_index: list[UniverseRunResult | None] = [None] * len(tasks)
    if jobs == 1:
        for index, task in enumerate(tasks):
            result = run_one_task(
                task,
                index=index,
                total=len(tasks),
                cwd=cwd,
                log_dir=log_dir,
                inner_workers=inner_workers,
                dry_run=dry_run,
                strict_exit_codes=strict_exit_codes,
            )
            results_by_index[index] = result
            print_result(index, len(tasks), result)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    run_one_task,
                    task,
                    index=index,
                    total=len(tasks),
                    cwd=cwd,
                    log_dir=log_dir,
                    inner_workers=inner_workers,
                    dry_run=dry_run,
                    strict_exit_codes=strict_exit_codes,
                ): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results_by_index[index] = result
                print_result(index, len(tasks), result)
    return [result for result in results_by_index if result is not None]


def run_one_task(
    task: dict[str, Any],
    *,
    index: int,
    total: int,
    cwd: Path,
    log_dir: Path,
    inner_workers: int,
    dry_run: bool,
    strict_exit_codes: bool,
) -> UniverseRunResult:
    del total
    command = command_for_task(task, inner_workers=inner_workers)
    task_id = str(task.get("taskId") or f"task_{index + 1}")
    safe_task_id = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in task_id
    )
    log_path = log_dir / f"{index + 1:05d}_{safe_task_id}.log"
    summary_path = batch_summary_path(command, cwd)
    if dry_run:
        log_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        return _result(task, "dry_run", None, log_path, summary_path, command)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode == 0:
        status = "completed"
    elif summary_path is not None and summary_path.exists() and not strict_exit_codes:
        status = "completed_nonzero"
    else:
        status = "failed"
    return _result(task, status, completed.returncode, log_path, summary_path, command)


def command_for_task(task: dict[str, Any], *, inner_workers: int) -> list[str]:
    command = task.get("optimizerCommand")
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ValueError(f"Task {task.get('taskId')} has no optimizerCommand list")
    return command_with_option(list(command), "--workers", str(inner_workers))


def command_with_option(command: list[str], option: str, value: str) -> list[str]:
    try:
        index = command.index(option)
    except ValueError:
        return [*command, option, value]
    if index + 1 >= len(command):
        return [*command, value]
    updated = list(command)
    updated[index + 1] = value
    return updated


def batch_summary_path(command: list[str], cwd: Path) -> Path | None:
    try:
        index = command.index("--output-dir")
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    output_dir = Path(command[index + 1])
    if not output_dir.is_absolute():
        output_dir = cwd / output_dir
    return output_dir / "batch_summary.json"


def _result(
    task: dict[str, Any],
    status: str,
    return_code: int | None,
    log_path: Path | None,
    summary_path: Path | None,
    command: list[str],
) -> UniverseRunResult:
    return UniverseRunResult(
        task_id=str(task.get("taskId") or ""),
        symbol=str(task.get("symbol") or ""),
        strategy_name=str(task.get("strategyName") or ""),
        status=status,
        return_code=return_code,
        log_path=str(log_path) if log_path is not None else None,
        summary_path=str(summary_path) if summary_path is not None else None,
        command=command,
    )


def print_result(index: int, total: int, result: UniverseRunResult) -> None:
    rc = "" if result.return_code is None else f" rc={result.return_code}"
    print(
        f"[{index + 1}/{total}] {result.status}{rc} "
        f"{result.symbol} {result.strategy_name} log={result.log_path}",
        flush=True,
    )


def write_run_summary(
    path: Path,
    results: list[UniverseRunResult],
    universe_path: Path,
    jobs: int,
) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    write_json(
        path,
        {
            "schema": "regime_tuning_universe_run.v1",
            "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "universe": str(universe_path),
            "jobs": jobs,
            "tasks": len(results),
            "statusCounts": dict(sorted(counts.items())),
            "results": [result.to_dict() for result in results],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
