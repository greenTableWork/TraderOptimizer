#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from trader_optimizer.data import find_trader_root
from trader_optimizer.regime_tuning_universe import (
    DEFAULT_REGIME_DIMENSIONS,
    build_regime_tuning_tasks,
    load_regime_vectors_jsonl,
    write_regime_tuning_universe,
)
from trader_optimizer.strategy_configs import (
    discover_strategy_candidates,
    is_optimizer_supported_candidate,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    trader_root = (args.trader_root or find_trader_root()).resolve()
    vectors = load_regime_vectors_jsonl(args.regime_vectors)
    candidates = discover_strategy_candidates(trader_root, args.config_glob)
    candidates = filter_candidates(
        candidates,
        include_types=args.include_strategy_type,
        exclude_types=args.exclude_strategy_type,
        optimizer_supported_only=args.optimizer_supported_only,
    )
    dimensions = tuple(args.dimension or DEFAULT_REGIME_DIMENSIONS)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output
        or Path.cwd()
        / "runs"
        / "regime_tuning_universe"
        / f"{timestamp}_regime_tuning_universe.jsonl"
    ).resolve()
    summary_path = (
        args.summary_output
        or output_path.with_name(f"{output_path.stem}_summary.json")
    ).resolve()
    output_root = (args.task_output_root or output_path.parent / "tasks").resolve()
    export_root = args.export_config_root.resolve() if args.export_config_root else None

    tasks = build_regime_tuning_tasks(
        vectors,
        candidates,
        trader_root=trader_root,
        output_root=output_root,
        export_root=export_root,
        regime_dimensions=dimensions,
        strategy_scope=args.strategy_scope,
        trials=args.trials,
        max_bars=args.max_bars,
        workers=args.workers,
        skip_backtester_build=not args.build_backtester,
    )
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    write_regime_tuning_universe(
        output_path,
        summary_path,
        tasks,
        vector_count=len(vectors),
        strategy_count=len(candidates),
        strategy_scope=args.strategy_scope,
        regime_dimensions=dimensions,
    )
    if not args.quiet:
        runnable = sum(1 for task in tasks if task["runnableWithExistingConfig"])
        print(f"vectors: {len(vectors)}")
        print(f"strategies: {len(candidates)}")
        print(f"tasks: {len(tasks)}")
        print(f"runnable_with_existing_config: {runnable}")
        print(f"requires_retargeting: {len(tasks) - runnable}")
        print(f"output: {output_path}")
        print(f"summary: {summary_path}")
    return 0


def filter_candidates(
    candidates,
    *,
    include_types: list[str] | None,
    exclude_types: list[str] | None,
    optimizer_supported_only: bool = False,
):
    include = {value for value in include_types or ()}
    exclude = {value for value in exclude_types or ()}
    output = []
    for candidate in candidates:
        if include and candidate.strategy_type not in include:
            continue
        if exclude and candidate.strategy_type in exclude:
            continue
        if optimizer_supported_only and not is_optimizer_supported_candidate(candidate):
            continue
        output.append(candidate)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a regime-conditioned strategy tuning universe from regime vectors "
            "and existing Trader strategy candidates."
        )
    )
    parser.add_argument(
        "--trader-root",
        type=Path,
        default=None,
        help="Trader workspace root. Defaults to auto-discovery from cwd.",
    )
    parser.add_argument(
        "--regime-vectors",
        type=Path,
        required=True,
        help="Regime vector JSONL produced by scripts/build_regime_vectors.py.",
    )
    parser.add_argument(
        "--config-glob",
        action="append",
        default=None,
        help="Strategy config glob relative to trader root. May be repeated.",
    )
    parser.add_argument(
        "--include-strategy-type",
        action="append",
        default=None,
        help="Only include this strategy_type. May be repeated.",
    )
    parser.add_argument(
        "--exclude-strategy-type",
        action="append",
        default=None,
        help="Exclude this strategy_type. May be repeated.",
    )
    parser.add_argument(
        "--optimizer-supported-only",
        action="store_true",
        help=(
            "Exclude configs whose strategy_type variant is known to be skipped by "
            "the current optimizer search spaces."
        ),
    )
    parser.add_argument(
        "--strategy-scope",
        choices=("matching-symbol", "all"),
        default="matching-symbol",
        help=(
            "matching-symbol emits runnable tasks for configs already containing "
            "the vector ticker; all emits the full ticker x strategy universe and "
            "marks symbol mismatches as requiring retargeting."
        ),
    )
    parser.add_argument(
        "--dimension",
        action="append",
        default=None,
        help=(
            "Core regime dimension to include in the cell key. May be repeated. "
            "Defaults to directionSign, instrumentVolatilityRegime, "
            "marketVolatilityRegime, and volumeRegime."
        ),
    )
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--max-bars", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--build-backtester",
        action="store_true",
        help="Omit --skip-backtester-build from generated optimizer commands.",
    )
    parser.add_argument(
        "--task-output-root",
        type=Path,
        default=None,
        help="Directory root for per-task optimizer outputs.",
    )
    parser.add_argument(
        "--export-config-root",
        type=Path,
        default=None,
        help="Directory root for per-task exported configs.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit emitted tasks. Intended for smoke tests.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
