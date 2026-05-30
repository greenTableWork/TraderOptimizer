from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from trader_optimizer.backtester import BackTesterSettings
from trader_optimizer.data import find_trader_root, load_bars
from trader_optimizer.batch import (
    BatchSettings,
    optimize_candidates,
    write_optimization_plan,
)
from trader_optimizer.optimizer import OptimizationSettings, run_optimization
from trader_optimizer.postgres import PostgresSettings, optuna_storage_url, postgres_settings_from_env
from trader_optimizer.strategy_configs import discover_strategy_candidates


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        return optimize(args)
    if args.command == "optimize-existing":
        return optimize_existing(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trader-optimizer",
        description="Verbose Optuna optimizer for TraderCore config JSON.",
    )
    subparsers = parser.add_subparsers(dest="command")

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize ConstantStepOffset parameters and write a config.",
    )
    optimize_parser.add_argument(
        "--trader-root",
        type=Path,
        default=None,
        help="Trader workspace root. Defaults to auto-discovery from cwd.",
    )
    add_postgres_options(optimize_parser)
    add_backtester_options(optimize_parser)
    optimize_parser.add_argument("--symbol", default="AAPL")
    optimize_parser.add_argument("--bar-size", default="10 secs")
    optimize_parser.add_argument("--what-to-show", default="TRADES")
    optimize_parser.add_argument("--use-rth", type=int, default=1)
    optimize_parser.add_argument("--start-utc", default=None)
    optimize_parser.add_argument("--end-utc", default=None)
    optimize_parser.add_argument(
        "--max-bars",
        type=int,
        default=50000,
        help="Use the latest N bars. Set 0 for all matching bars.",
    )
    optimize_parser.add_argument("--trials", type=int, default=50)
    optimize_parser.add_argument("--train-fraction", type=float, default=0.70)
    optimize_parser.add_argument(
        "--min-trades",
        type=int,
        default=4,
        help="Penalty threshold for configs that barely trade.",
    )
    optimize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory. Defaults to runs/<timestamp>_<symbol>.",
    )
    optimize_parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name. Defaults to a timestamped name.",
    )
    optimize_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging.",
    )

    existing_parser = subparsers.add_parser(
        "optimize-existing",
        help="Discover existing Trader backtest configs and optimize each one.",
    )
    existing_parser.add_argument(
        "--trader-root",
        type=Path,
        default=None,
        help="Trader workspace root. Defaults to auto-discovery from cwd.",
    )
    add_postgres_options(existing_parser)
    add_backtester_options(existing_parser)
    existing_parser.add_argument(
        "--config-glob",
        action="append",
        default=None,
        help=(
            "Glob relative to trader root. May be repeated. Defaults to "
            "TraderCore backtesting configs and TraderLab stock-stress configs."
        ),
    )
    existing_parser.add_argument(
        "--bar-size",
        default=None,
        help="Preferred bar size. If omitted, the optimizer auto-selects local data.",
    )
    existing_parser.add_argument("--start-utc", default=None)
    existing_parser.add_argument("--end-utc", default=None)
    existing_parser.add_argument(
        "--include-strategy-type",
        action="append",
        default=None,
        help="Only optimize this strategy_type. May be repeated.",
    )
    existing_parser.add_argument(
        "--exclude-strategy-type",
        action="append",
        default=None,
        help="Skip this strategy_type. May be repeated.",
    )
    existing_parser.add_argument("--trials", type=int, default=25)
    existing_parser.add_argument("--max-bars", type=int, default=5000)
    existing_parser.add_argument("--train-fraction", type=float, default=0.70)
    existing_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel strategy optimizations. Default 0 uses up to 4 workers.",
    )
    existing_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Batch output directory. Defaults to runs/batch_<timestamp>.",
    )
    existing_parser.add_argument(
        "--plan-path",
        type=Path,
        default=None,
        help="Write a markdown optimization plan before running Optuna.",
    )
    existing_parser.add_argument(
        "--export-config-dir",
        type=Path,
        default=None,
        help="Copy every generated best_config.json into a stable directory.",
    )
    existing_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging.",
    )
    return parser


def add_postgres_options(parser: argparse.ArgumentParser) -> None:
    env = postgres_settings_from_env()
    parser.add_argument(
        "--pg-conninfo",
        default=env.conninfo,
        help="libpq PostgreSQL conninfo. Defaults to TRADER_PG_CONNINFO.",
    )
    parser.add_argument("--pg-host", default=env.host)
    parser.add_argument("--pg-port", type=int, default=env.port)
    parser.add_argument("--pg-database", default=env.database)
    parser.add_argument("--pg-user", default=env.user)
    parser.add_argument("--pg-password", default=env.password)
    parser.add_argument(
        "--optuna-storage-url",
        default=env.optuna_storage_url,
        help=(
            "SQLAlchemy PostgreSQL URL for Optuna. Defaults to "
            "TRADER_OPTIMIZER_OPTUNA_STORAGE or a URL built from PG settings."
        ),
    )


def add_backtester_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-backtester-validation",
        action="store_true",
        help="Disable real TraderCore BackTester validation for generated configs.",
    )
    parser.add_argument(
        "--backtester",
        type=Path,
        default=None,
        help="Override the TraderCore BackTester executable path.",
    )
    parser.add_argument(
        "--backtester-preset",
        default="debug",
        help="TraderCore CMake preset/build directory for BackTester. Default: debug.",
    )
    parser.add_argument(
        "--skip-backtester-build",
        action="store_true",
        help="Reuse the existing BackTester binary instead of building it first.",
    )
    parser.add_argument(
        "--backtester-timeout-seconds",
        type=int,
        default=300,
        help="Timeout for each BackTester validation run.",
    )
    parser.add_argument(
        "--benchmark-symbol",
        default="SPX",
        help="PostgreSQL symbol used for the market benchmark. Default: SPX.",
    )
    parser.add_argument(
        "--backtester-starting-cash",
        type=float,
        default=100000.0,
        help="Starting cash for generated BackTester validation configs.",
    )


def postgres_settings_from_args(args: argparse.Namespace) -> PostgresSettings:
    return PostgresSettings(
        conninfo=args.pg_conninfo or "",
        host=args.pg_host,
        port=args.pg_port,
        database=args.pg_database,
        user=args.pg_user,
        password=args.pg_password,
        optuna_storage_url=args.optuna_storage_url,
    )


def backtester_settings_from_args(
    args: argparse.Namespace,
    trader_root: Path,
    pg_settings: PostgresSettings,
) -> BackTesterSettings | None:
    if args.no_backtester_validation:
        return None
    return BackTesterSettings(
        trader_root=trader_root,
        pg_settings=pg_settings,
        preset=args.backtester_preset,
        backtester=args.backtester,
        skip_build=args.skip_backtester_build,
        timeout_seconds=args.backtester_timeout_seconds,
        benchmark_symbol=args.benchmark_symbol,
        starting_cash=args.backtester_starting_cash,
    )


def optimize(args: argparse.Namespace) -> int:
    trader_root = (args.trader_root or find_trader_root()).resolve()
    pg_settings = postgres_settings_from_args(args)
    backtester_settings = backtester_settings_from_args(args, trader_root, pg_settings)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    study_name = args.study_name or f"trader_optimizer_{args.symbol}_{timestamp}"
    output_dir = (
        args.output_dir
        or Path.cwd() / "runs" / f"{timestamp}_{args.symbol}_{args.bar_size.replace(' ', '')}"
    ).resolve()
    storage_url = optuna_storage_url(pg_settings)
    verbose = not args.quiet

    if verbose:
        print("TraderOptimizer starting")
        print(f"  trader_root: {trader_root}")
        print(f"  postgres: {pg_settings.display}")
        print(f"  optuna_storage: {storage_url}")
        print(f"  symbol: {args.symbol}")
        print(f"  bar_size: {args.bar_size}")
        print(f"  what_to_show: {args.what_to_show}")
        print(f"  use_rth: {args.use_rth}")
        print(f"  max_bars: {args.max_bars}")
        print(f"  trials: {args.trials}")

    window = load_bars(
        pg_settings=pg_settings,
        symbol=args.symbol,
        bar_size=args.bar_size,
        what_to_show=args.what_to_show,
        use_rth=args.use_rth,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        max_bars=args.max_bars,
    )

    if verbose:
        print("Loaded bars")
        print(f"  count: {len(window.bars)}")
        print(f"  first: {window.first_timestamp}")
        print(f"  last: {window.last_timestamp}")

    artifacts = run_optimization(
        window,
        OptimizationSettings(
            trials=args.trials,
            train_fraction=args.train_fraction,
            output_dir=output_dir,
            study_name=study_name,
            storage_url=storage_url,
            pg_settings=pg_settings,
            min_trades=args.min_trades,
            verbose=verbose,
            backtester_settings=backtester_settings,
        ),
    )

    if verbose:
        print("TraderOptimizer finished")
        print(f"  best_value: {artifacts.best_value:.8f}")
        print(f"  config_path: {artifacts.config_path}")
        print(f"  summary_path: {artifacts.summary_path}")
        print(f"  study_storage: {artifacts.study_storage}")
    if artifacts.backtester_status and artifacts.backtester_status != "ok":
        return 1
    return 0


def optimize_existing(args: argparse.Namespace) -> int:
    trader_root = (args.trader_root or find_trader_root()).resolve()
    pg_settings = postgres_settings_from_args(args)
    backtester_settings = backtester_settings_from_args(args, trader_root, pg_settings)
    storage_url = optuna_storage_url(pg_settings)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir or Path.cwd() / "runs" / f"batch_{timestamp}"
    ).resolve()
    verbose = not args.quiet
    candidates = discover_strategy_candidates(trader_root, args.config_glob)
    candidates = _filter_candidates(
        candidates,
        include_types=args.include_strategy_type,
        exclude_types=args.exclude_strategy_type,
    )
    if verbose:
        print("TraderOptimizer batch starting")
        print(f"  trader_root: {trader_root}")
        print(f"  postgres: {pg_settings.display}")
        print(f"  optuna_storage: {storage_url}")
        print(f"  candidates: {len(candidates)}")
        print(f"  trials_per_candidate: {args.trials}")
        print(f"  max_bars: {args.max_bars}")
        print(f"  workers: {args.workers or 'auto'}")
        print(f"  output_dir: {output_dir}")
        if args.plan_path:
            print(f"  plan_path: {args.plan_path.resolve()}")
        if args.export_config_dir:
            print(f"  export_config_dir: {args.export_config_dir.resolve()}")
    settings = BatchSettings(
        pg_settings=pg_settings,
        optuna_storage_url=storage_url,
        output_dir=output_dir,
        trials=args.trials,
        max_bars=args.max_bars,
        preferred_bar_size=args.bar_size,
        train_fraction=args.train_fraction,
        verbose=verbose,
        export_config_dir=args.export_config_dir.resolve()
        if args.export_config_dir
        else None,
        workers=args.workers,
        backtester_settings=backtester_settings,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
    )
    if args.plan_path:
        write_optimization_plan(candidates, settings, args.plan_path.resolve())
    results = optimize_candidates(candidates, settings)
    ok = sum(1 for result in results if result.status == "ok")
    not_exported = len(results) - ok
    if verbose:
        print("TraderOptimizer batch finished")
        print(f"  benchmark_passing: {ok}")
        print(f"  not_exported: {not_exported}")
        print(f"  summary_json: {output_dir / 'batch_summary.json'}")
        print("  pg_tables: optimizer_runs, optimizer_trials, optimizer_batch_results")
        if args.export_config_dir:
            print(f"  exported_configs: {args.export_config_dir.resolve()}")
    return 0 if ok else 1


def _filter_candidates(
    candidates,
    include_types: list[str] | None,
    exclude_types: list[str] | None,
):
    include = {item.lower() for item in include_types or []}
    exclude = {item.lower() for item in exclude_types or []}
    output = []
    for candidate in candidates:
        strategy_type = candidate.strategy_type.lower()
        if include and strategy_type not in include:
            continue
        if strategy_type in exclude:
            continue
        output.append(candidate)
    return output
