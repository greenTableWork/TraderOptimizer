from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from trader_optimizer.data import default_market_db, find_trader_root, load_bars
from trader_optimizer.optimizer import OptimizationSettings, run_optimization


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        return optimize(args)
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
    optimize_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite historical bars DB. Defaults to TraderLab/Data/tws_historical.sqlite.",
    )
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
    return parser


def optimize(args: argparse.Namespace) -> int:
    trader_root = (args.trader_root or find_trader_root()).resolve()
    db_path = (args.db or default_market_db(trader_root)).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    study_name = args.study_name or f"trader_optimizer_{args.symbol}_{timestamp}"
    output_dir = (
        args.output_dir
        or Path.cwd() / "runs" / f"{timestamp}_{args.symbol}_{args.bar_size.replace(' ', '')}"
    ).resolve()
    storage_path = output_dir / "optuna-study.db"
    verbose = not args.quiet

    if verbose:
        print("TraderOptimizer starting")
        print(f"  trader_root: {trader_root}")
        print(f"  db: {db_path}")
        print(f"  symbol: {args.symbol}")
        print(f"  bar_size: {args.bar_size}")
        print(f"  what_to_show: {args.what_to_show}")
        print(f"  use_rth: {args.use_rth}")
        print(f"  max_bars: {args.max_bars}")
        print(f"  trials: {args.trials}")

    window = load_bars(
        db_path=db_path,
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
            storage_path=storage_path,
            min_trades=args.min_trades,
            verbose=verbose,
        ),
    )

    if verbose:
        print("TraderOptimizer finished")
        print(f"  best_value: {artifacts.best_value:.8f}")
        print(f"  config_path: {artifacts.config_path}")
        print(f"  summary_path: {artifacts.summary_path}")
        print(f"  study_db_path: {artifacts.study_db_path}")
    return 0
