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
from trader_optimizer.benchmark_loop import (
    BenchmarkLoopSettings,
    run_benchmark_loop,
)
from trader_optimizer.optimizer import OptimizationSettings, run_optimization
from trader_optimizer.live_regime import (
    build_live_regime_detections,
    load_live_regime_state,
    load_strategy_regime_candidates,
    write_live_regime_detections_jsonl,
    write_live_regime_state,
    write_live_regime_summary,
)
from trader_optimizer.postgres import (
    PostgresSettings,
    insert_live_regime_detections,
    insert_strategy_regime_config_map,
    optuna_storage_url,
    postgres_connection,
    postgres_settings_from_env,
)
from trader_optimizer.regime_detector_specs import DETECTOR_SPEC_VERSION
from trader_optimizer.regime_tuning_universe import (
    DEFAULT_REGIME_DIMENSIONS,
    load_regime_vectors_jsonl,
)
from trader_optimizer.strategy_regime_map import (
    DEFAULT_VALIDATION_STATUSES,
    build_strategy_regime_map_from_run_summary,
    write_strategy_regime_map_jsonl,
    write_strategy_regime_map_summary,
)
from trader_optimizer.strategy_configs import discover_strategy_candidates


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        return optimize(args)
    if args.command == "optimize-existing":
        return optimize_existing(args)
    if args.command == "benchmark-loop":
        return benchmark_loop(args)
    if args.command == "detect-live-regimes":
        return detect_live_regimes(args)
    if args.command == "build-strategy-map":
        return build_strategy_map(args)
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
        "--strategy-budget",
        type=float,
        default=None,
        help=(
            "Budget used to size generated strategy configs. For single-symbol "
            "strategies this caps searched order quantity by first-bar notional; "
            "for portfolio strategies this overrides portfolioNotional."
        ),
    )
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

    benchmark_parser = subparsers.add_parser(
        "benchmark-loop",
        help="Continuously run rolling benchmarks for daily through yearly strategy windows.",
    )
    benchmark_parser.add_argument(
        "--trader-root",
        type=Path,
        default=None,
        help="Trader workspace root. Defaults to auto-discovery from cwd.",
    )
    add_postgres_options(benchmark_parser)
    add_backtester_options(benchmark_parser)
    benchmark_parser.add_argument(
        "--config-glob",
        action="append",
        default=None,
        help="Strategy config glob relative to trader root. May be repeated.",
    )
    benchmark_parser.add_argument("--bar-size", default=None)
    benchmark_parser.add_argument(
        "--include-strategy-type",
        action="append",
        default=None,
        help="Only benchmark this strategy_type. May be repeated.",
    )
    benchmark_parser.add_argument(
        "--exclude-strategy-type",
        action="append",
        default=None,
        help="Skip this strategy_type. May be repeated.",
    )
    benchmark_parser.add_argument("--trials", type=int, default=25)
    benchmark_parser.add_argument("--max-bars", type=int, default=0)
    benchmark_parser.add_argument("--train-fraction", type=float, default=0.70)
    benchmark_parser.add_argument(
        "--strategy-budget",
        type=float,
        default=None,
        help="Budget used to size generated strategy configs during benchmark runs.",
    )
    benchmark_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel strategy optimizations per benchmark period. Default 0 uses up to 4 workers.",
    )
    benchmark_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root for benchmark cycles. Defaults to runs/benchmarks.",
    )
    benchmark_parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="State JSON tracking promoted champions. Defaults to <output-root>/benchmark_state.json.",
    )
    benchmark_parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=24 * 60 * 60,
        help="Seconds between benchmark cycles when not using --once. Default: 86400.",
    )
    benchmark_parser.add_argument(
        "--once",
        action="store_true",
        help="Run one benchmark cycle and exit instead of looping forever.",
    )
    benchmark_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging.",
    )

    regime_parser = subparsers.add_parser(
        "detect-live-regimes",
        help="Convert current regime vectors into smoothed live regime decisions.",
    )
    add_postgres_options(regime_parser)
    regime_parser.add_argument(
        "--regime-vectors",
        type=Path,
        required=True,
        help="Current regime vector JSONL produced by scripts/build_regime_vectors.py.",
    )
    regime_parser.add_argument(
        "--strategy-map",
        type=Path,
        default=None,
        help="Optional JSONL of BackTester-gated strategy to regime-cell mappings.",
    )
    regime_parser.add_argument(
        "--state-input",
        type=Path,
        default=None,
        help="Previous live regime state JSON. Defaults to empty state.",
    )
    regime_parser.add_argument(
        "--state-output",
        type=Path,
        default=None,
        help="Write updated live regime state JSON.",
    )
    regime_parser.add_argument("--output", type=Path, default=None)
    regime_parser.add_argument("--summary-output", type=Path, default=None)
    regime_parser.add_argument(
        "--mode",
        choices=("shadow", "paper", "production"),
        default="shadow",
    )
    regime_parser.add_argument(
        "--min-persistence",
        type=int,
        default=3,
        help="Raw regime observations required before switching active cell.",
    )
    regime_parser.add_argument(
        "--change-point-threshold",
        type=float,
        default=0.80,
        help="Change-point confidence that can override hysteresis.",
    )
    regime_parser.add_argument(
        "--dimension",
        action="append",
        default=None,
        help=(
            "Core regime dimension for the active cell. May be repeated. "
            "Defaults to directionSign, instrumentVolatilityRegime, "
            "marketVolatilityRegime, and volumeRegime."
        ),
    )
    regime_parser.add_argument(
        "--write-postgres",
        action="store_true",
        help="Persist detections into live regime PostgreSQL tables.",
    )
    regime_parser.add_argument("--run-id", default=None)
    regime_parser.add_argument("--quiet", action="store_true")

    strategy_map_parser = subparsers.add_parser(
        "build-strategy-map",
        help="Build a live-compatible strategy-to-regime map from a universe run summary.",
    )
    add_postgres_options(strategy_map_parser)
    strategy_map_parser.add_argument(
        "--run-summary",
        type=Path,
        required=True,
        help="regime_tuning_universe_run.v1 summary produced by run_regime_tuning_universe.py.",
    )
    strategy_map_parser.add_argument(
        "--validation-status",
        action="append",
        default=None,
        help="Include this batch validation status. Defaults to ok. May be repeated.",
    )
    strategy_map_parser.add_argument("--output", type=Path, default=None)
    strategy_map_parser.add_argument("--summary-output", type=Path, default=None)
    strategy_map_parser.add_argument(
        "--write-postgres",
        action="store_true",
        help="Upsert the map into strategy_regime_config_map.",
    )
    strategy_map_parser.add_argument("--quiet", action="store_true")
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
        if args.strategy_budget is not None:
            print(f"  strategy_budget: {args.strategy_budget}")
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
        strategy_budget=args.strategy_budget,
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


def detect_live_regimes(args: argparse.Namespace) -> int:
    pg_settings = postgres_settings_from_args(args)
    vectors = load_regime_vectors_jsonl(args.regime_vectors)
    previous_states = load_live_regime_state(args.state_input)
    strategy_candidates = load_strategy_regime_candidates(args.strategy_map)
    dimensions = tuple(args.dimension or DEFAULT_REGIME_DIMENSIONS)
    generated_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output
        or Path.cwd()
        / "runs"
        / "live_regimes"
        / f"{timestamp}_live_regime_detections.jsonl"
    ).resolve()
    summary_path = (
        args.summary_output
        or output_path.with_name(f"{output_path.stem}_summary.json")
    ).resolve()
    state_output = (
        args.state_output
        or output_path.with_name(f"{output_path.stem}_state.json")
    ).resolve()
    run_id = args.run_id or f"live_regime_{timestamp}"

    detections, states = build_live_regime_detections(
        vectors,
        previous_states=previous_states,
        min_persistence=args.min_persistence,
        change_point_threshold=args.change_point_threshold,
        mode=args.mode,
        generated_utc=generated_utc,
        regime_dimensions=dimensions,
        strategy_candidates=strategy_candidates,
    )
    write_live_regime_detections_jsonl(output_path, detections)
    write_live_regime_state(state_output, states)
    write_live_regime_summary(summary_path, detections)
    if args.write_postgres:
        with postgres_connection(pg_settings) as conn:
            insert_live_regime_detections(
                conn,
                run_id=run_id,
                mode=args.mode,
                detector_spec_version=DETECTOR_SPEC_VERSION,
                detections=detections,
                metadata={
                    "regimeVectors": str(args.regime_vectors),
                    "stateInput": str(args.state_input)
                    if args.state_input
                    else None,
                    "strategyMap": str(args.strategy_map)
                    if args.strategy_map
                    else None,
                    "dimensions": list(dimensions),
                },
            )

    if not args.quiet:
        print("Live regime detection finished")
        print(f"  run_id: {run_id}")
        print(f"  mode: {args.mode}")
        print(f"  vectors: {len(vectors)}")
        print(f"  detections: {len(detections)}")
        print(f"  output: {output_path}")
        print(f"  state: {state_output}")
        print(f"  summary: {summary_path}")
        if args.write_postgres:
            print("  pg_tables: live_regime_vectors, regime_vector_history")
    return 0


def build_strategy_map(args: argparse.Namespace) -> int:
    run_summary_path = args.run_summary.resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output
        or Path.cwd()
        / "runs"
        / "strategy_regime_maps"
        / f"{timestamp}_strategy_regime_map.jsonl"
    ).resolve()
    summary_path = (
        args.summary_output
        or output_path.with_name(f"{output_path.stem}_summary.json")
    ).resolve()
    validation_statuses = tuple(args.validation_status or DEFAULT_VALIDATION_STATUSES)
    entries = build_strategy_regime_map_from_run_summary(
        run_summary_path,
        validation_statuses=validation_statuses,
        cwd=Path.cwd(),
    )
    write_strategy_regime_map_jsonl(output_path, entries)
    write_strategy_regime_map_summary(
        summary_path,
        entries,
        run_summary_path=run_summary_path,
    )
    if args.write_postgres:
        pg_settings = postgres_settings_from_args(args)
        with postgres_connection(pg_settings) as conn:
            insert_strategy_regime_config_map(conn, entries=entries)
    if not args.quiet:
        print("Strategy regime map finished")
        print(f"  run_summary: {run_summary_path}")
        print(f"  validation_statuses: {', '.join(validation_statuses)}")
        print(f"  entries: {len(entries)}")
        print(f"  output: {output_path}")
        print(f"  summary: {summary_path}")
        if args.write_postgres:
            print("  pg_table: strategy_regime_config_map")
    return 0


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
