from pathlib import Path

from trader_optimizer.regime_tuning_runner import (
    batch_summary_path,
    command_option_value,
    command_for_task,
    command_with_option,
    resolve_jobs,
    trader_roots_for_tasks,
)


def test_command_for_task_overrides_inner_workers() -> None:
    task = {
        "optimizerCommand": [
            ".venv/bin/trader-optimizer",
            "optimize-existing",
            "--workers",
            "10",
            "--output-dir",
            "runs/task",
        ],
    }

    command = command_for_task(task, inner_workers=1)

    assert command == [
        ".venv/bin/trader-optimizer",
        "optimize-existing",
        "--workers",
        "1",
        "--output-dir",
        "runs/task",
    ]


def test_command_with_option_appends_missing_option() -> None:
    assert command_with_option(["cmd"], "--workers", "2") == ["cmd", "--workers", "2"]


def test_command_option_value_returns_existing_value() -> None:
    assert command_option_value(["cmd", "--trader-root", "/tmp/trader"], "--trader-root") == (
        "/tmp/trader"
    )
    assert command_option_value(["cmd"], "--trader-root") is None


def test_batch_summary_path_resolves_from_output_dir() -> None:
    command = [
        ".venv/bin/trader-optimizer",
        "optimize-existing",
        "--output-dir",
        "runs/task",
    ]

    assert batch_summary_path(command, Path("/tmp/repo")) == Path(
        "/tmp/repo/runs/task/batch_summary.json"
    )


def test_resolve_jobs_caps_to_task_count() -> None:
    assert resolve_jobs(8, 3) == 3
    assert resolve_jobs(1, 3) == 1
    assert resolve_jobs(0, 1) == 1


def test_trader_roots_for_tasks_dedupes_and_resolves_relative_paths() -> None:
    cwd = Path("/tmp/work/TraderOptimizer")
    tasks = [
        {"optimizerCommand": ["cmd", "--trader-root", ".."]},
        {"optimizerCommand": ["cmd", "--trader-root", "/opt/trader"]},
        {"optimizerCommand": ["cmd", "--trader-root", ".."]},
        {"optimizerCommand": ["cmd"]},
    ]

    assert trader_roots_for_tasks(tasks, cwd=cwd) == [
        (cwd / "..").resolve(),
        Path("/opt/trader"),
    ]
