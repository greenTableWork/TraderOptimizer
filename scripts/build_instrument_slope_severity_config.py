#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from trader_optimizer.data import available_profiles, choose_data_profile, load_bars
from trader_optimizer.postgres import PostgresSettings, postgres_settings_from_env
from trader_optimizer.slope_severity import (
    DEFAULT_SLOPE_SEVERITY_QUANTILES,
    build_slope_severity_config,
    write_slope_severity_config,
)
from trader_optimizer.tuning_regions import normalize_periods


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pg_settings = postgres_settings_from_args(args)
    periods = normalize_periods(args.period)
    symbols = tuple(args.symbol or ())
    if args.all_symbols:
        symbols = tuple(profile.symbol for profile in available_profiles(pg_settings))
    symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
    if not symbols:
        parser.error("pass at least one --symbol or use --all-symbols")

    symbol_bars = {}
    profiles = {}
    for symbol in symbols:
        profile = choose_data_profile(
            pg_settings,
            symbol,
            preferred_bar_size=args.bar_size,
        )
        window = load_bars(
            pg_settings=pg_settings,
            symbol=symbol,
            bar_size=profile.bar_size,
            what_to_show=profile.what_to_show,
            use_rth=profile.use_rth,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            max_bars=args.max_bars,
        )
        symbol_bars[symbol] = window.bars
        profiles[symbol] = {
            "barSize": profile.bar_size,
            "whatToShow": profile.what_to_show,
            "useRth": profile.use_rth,
        }
        if not args.quiet:
            print(
                f"{symbol}: {len(window.bars)} bars using "
                f"{profile.bar_size} {profile.what_to_show} rth={profile.use_rth}"
            )

    config = build_slope_severity_config(
        symbol_bars,
        periods=periods,
        bucket_timezone=args.bucket_timezone,
        min_bars=args.min_bars,
        quantiles=tuple(args.quantile or DEFAULT_SLOPE_SEVERITY_QUANTILES),
        profiles=profiles,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output
        or Path.cwd()
        / "runs"
        / "tuning_regions"
        / f"{timestamp}_slope_severity_config.json"
    ).resolve()
    write_slope_severity_config(output_path, config)
    if not args.quiet:
        print(f"entries: {len(config['entries'])}")
        print(f"output: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-instrument slope severity thresholds from PostgreSQL "
            "historical_bars for use by the tuning region creator."
        )
    )
    add_postgres_options(parser)
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="Symbol to include. May be repeated.",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Build entries for every symbol with PostgreSQL historical bars.",
    )
    parser.add_argument(
        "--period",
        action="append",
        choices=("day", "week", "month"),
        default=None,
        help="Region period. May be repeated. Defaults to day, week, and month.",
    )
    parser.add_argument(
        "--bucket-timezone",
        default="UTC",
        help=(
            "IANA timezone used for day/week/month bucket boundaries. "
            "Default: UTC."
        ),
    )
    parser.add_argument(
        "--bar-size",
        default=None,
        help="Preferred bar size. Defaults to the best available PostgreSQL profile.",
    )
    parser.add_argument("--start-utc", default=None)
    parser.add_argument("--end-utc", default=None)
    parser.add_argument(
        "--max-bars",
        type=int,
        default=0,
        help="Use latest N bars per symbol. Default 0 uses all matching bars.",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=2,
        help="Minimum bars required for a period bucket to become a slope sample.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        action="append",
        default=None,
        help=(
            "Severity boundary quantile. Pass exactly four values. Defaults to "
            "0.20, 0.40, 0.60, and 0.80."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def add_postgres_options(parser: argparse.ArgumentParser) -> None:
    env = postgres_settings_from_env()
    parser.add_argument("--pg-conninfo", default=env.conninfo)
    parser.add_argument("--pg-host", default=env.host)
    parser.add_argument("--pg-port", type=int, default=env.port)
    parser.add_argument("--pg-database", default=env.database)
    parser.add_argument("--pg-user", default=env.user)
    parser.add_argument("--pg-password", default=env.password)


def postgres_settings_from_args(args: argparse.Namespace) -> PostgresSettings:
    return PostgresSettings(
        conninfo=args.pg_conninfo or "",
        host=args.pg_host,
        port=args.pg_port,
        database=args.pg_database,
        user=args.pg_user,
        password=args.pg_password,
    )


if __name__ == "__main__":
    raise SystemExit(main())
