#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from trader_optimizer.data import Bar, DataProfile, choose_data_profile, load_bars
from trader_optimizer.market_feature_sources import (
    DEFAULT_OPTION_TRADE_TABLES,
    load_available_option_trades,
)
from trader_optimizer.market_features import index_futures_for_symbol
from trader_optimizer.slope_severity import load_slope_severity_config
from trader_optimizer.volatility_regime import load_volatility_regime_config
from trader_optimizer.postgres import PostgresSettings, postgres_settings_from_env
from trader_optimizer.tuning_regions import (
    categorize_direction_regions,
    categorize_index_futures_direction_regions,
    categorize_options_probability_regions,
    categorize_volume_orderbook_regions,
    categorize_volatility_regions,
    filter_duplicate_regions,
    load_region_ids,
    normalize_directions,
    normalize_futures_alignments,
    normalize_momentum_regimes,
    normalize_periods,
    normalize_volume_directions,
    normalize_volume_regimes,
    normalize_volatility_regimes,
    write_region_summary,
    write_regions_csv,
    write_regions_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pg_settings = postgres_settings_from_args(args)
    periods = normalize_periods(args.period)
    directions = normalize_directions(args.direction)
    volatility_regimes = normalize_volatility_regimes(args.volatility_regime)
    futures_alignments = normalize_futures_alignments(args.futures_alignment)
    momentum_regimes = normalize_momentum_regimes(args.options_momentum_regime)
    volume_regimes = normalize_volume_regimes(args.volume_regime)
    volume_directions = normalize_volume_directions(args.volume_direction)
    slope_severity_config = load_slope_severity_config(args.slope_severity_config)
    volatility_regime_config = load_volatility_regime_config(
        args.volatility_regime_config
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output
        or Path.cwd()
        / "runs"
        / "tuning_regions"
        / f"{timestamp}_{args.subcategory}_regions.{args.format}"
    ).resolve()
    summary_path = (
        args.summary_output
        or output_path.with_name(f"{output_path.stem}_summary.json")
    ).resolve()

    market_bars = None
    market_profile = None
    if args.subcategory == "volatility" and args.market_symbol:
        market_profile = choose_data_profile(
            pg_settings,
            args.market_symbol,
            preferred_bar_size=args.market_bar_size or args.bar_size,
        )
        market_bars = load_bars(
            pg_settings=pg_settings,
            symbol=args.market_symbol,
            bar_size=market_profile.bar_size,
            what_to_show=market_profile.what_to_show,
            use_rth=market_profile.use_rth,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            max_bars=args.max_bars,
        ).bars
        if not args.quiet:
            print(
                f"{args.market_symbol}: {len(market_bars)} market bars loaded "
                f"using {market_profile.bar_size} {market_profile.what_to_show} "
                f"rth={market_profile.use_rth}"
            )
    market_volatility_thresholds = (
        volatility_regime_config.thresholds_by_period(
            args.market_symbol,
            periods,
            bucket_timezone=args.bucket_timezone,
            bar_size=market_profile.bar_size if market_profile else None,
            what_to_show=market_profile.what_to_show if market_profile else None,
            use_rth=market_profile.use_rth if market_profile else None,
        )
        if volatility_regime_config is not None and args.market_symbol
        else None
    )

    option_trades = []
    if args.subcategory == "options-probability-map":
        option_trades = load_available_option_trades(
            pg_settings,
            args.symbol,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            limit=args.option_trade_limit,
            table_names=tuple(args.option_trade_table or DEFAULT_OPTION_TRADE_TABLES),
        )
        if not args.quiet:
            print(f"option_trades: {len(option_trades)} loaded")

    regions = []
    for symbol in args.symbol:
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
        slope_thresholds = (
            slope_severity_config.thresholds_by_period(
                symbol,
                periods,
                bucket_timezone=args.bucket_timezone,
                bar_size=profile.bar_size,
                what_to_show=profile.what_to_show,
                use_rth=profile.use_rth,
            )
            if slope_severity_config is not None
            else None
        )
        volatility_thresholds = (
            volatility_regime_config.thresholds_by_period(
                symbol,
                periods,
                bucket_timezone=args.bucket_timezone,
                bar_size=profile.bar_size,
                what_to_show=profile.what_to_show,
                use_rth=profile.use_rth,
            )
            if volatility_regime_config is not None
            else None
        )
        if args.subcategory == "direction":
            symbol_regions = categorize_direction_regions(
                symbol,
                window.bars,
                periods=periods,
                directions=directions,
                min_bars=args.min_bars,
                flat_threshold_pct=args.flat_threshold_pct,
                slope_severity_thresholds=slope_thresholds,
                bucket_timezone=args.bucket_timezone,
            )
        elif args.subcategory == "volatility":
            symbol_regions = categorize_volatility_regions(
                symbol,
                window.bars,
                market_symbol=args.market_symbol,
                market_bars=market_bars,
                periods=periods,
                regimes=volatility_regimes,
                min_bars=args.min_bars,
                low_threshold_pct=args.low_volatility_threshold_pct,
                high_threshold_pct=args.high_volatility_threshold_pct,
                volatility_regime_thresholds=volatility_thresholds,
                market_volatility_regime_thresholds=market_volatility_thresholds,
                bucket_timezone=args.bucket_timezone,
            )
        elif args.subcategory == "index-futures-direction":
            futures_symbol, futures_bars, futures_profile = load_futures_proxy_bars(
                pg_settings=pg_settings,
                symbol=symbol,
                explicit_futures_symbol=args.futures_symbol,
                preferred_bar_size=args.futures_bar_size or args.bar_size,
                start_utc=args.start_utc,
                end_utc=args.end_utc,
                max_bars=args.max_bars,
            )
            if futures_symbol is None or futures_bars is None:
                if not args.quiet:
                    candidates = ", ".join(index_futures_for_symbol(symbol))
                    print(
                        f"{symbol}: no index futures proxy bars loaded "
                        f"(candidates: {candidates})"
                    )
                symbol_regions = []
            else:
                symbol_regions = categorize_index_futures_direction_regions(
                    symbol,
                    window.bars,
                    futures_symbol=futures_symbol,
                    futures_bars=futures_bars,
                    periods=periods,
                    alignments=futures_alignments,
                    min_bars=args.min_bars,
                    flat_threshold_pct=args.flat_threshold_pct,
                    slope_severity_thresholds=slope_thresholds,
                    futures_slope_severity_thresholds=(
                        slope_severity_config.thresholds_by_period(
                            futures_symbol,
                            periods,
                            bucket_timezone=args.bucket_timezone,
                            bar_size=futures_profile.bar_size if futures_profile else None,
                            what_to_show=(
                                futures_profile.what_to_show if futures_profile else None
                            ),
                            use_rth=futures_profile.use_rth if futures_profile else None,
                        )
                        if slope_severity_config is not None
                        else None
                    ),
                    bucket_timezone=args.bucket_timezone,
                )
                if not args.quiet:
                    print(
                        f"{futures_symbol}: {len(futures_bars)} futures bars loaded "
                        f"for {symbol}"
                    )
        elif args.subcategory == "options-probability-map":
            symbol_regions = categorize_options_probability_regions(
                symbol,
                window.bars,
                option_trades=option_trades,
                periods=periods,
                momentum_regimes=momentum_regimes,
                min_bars=args.min_bars,
                min_option_trades=args.min_option_trades,
                low_momentum_threshold=args.low_options_momentum_threshold,
                high_momentum_threshold=args.high_options_momentum_threshold,
                bucket_timezone=args.bucket_timezone,
            )
        else:
            symbol_regions = categorize_volume_orderbook_regions(
                symbol,
                window.bars,
                periods=periods,
                volume_regimes=volume_regimes,
                volume_directions=volume_directions,
                min_bars=args.min_bars,
                low_relative_volume_threshold=args.low_relative_volume_threshold,
                high_relative_volume_threshold=args.high_relative_volume_threshold,
                volume_direction_threshold=args.volume_direction_threshold,
                flat_threshold_pct=args.flat_threshold_pct,
                slope_severity_thresholds=slope_thresholds,
                bucket_timezone=args.bucket_timezone,
            )
        symbol_regions = [
            replace(
                region,
                bar_size=profile.bar_size,
                what_to_show=profile.what_to_show,
                use_rth=profile.use_rth,
            )
            for region in symbol_regions
        ]
        regions.extend(symbol_regions)
        if not args.quiet:
            print(
                f"{symbol}: {len(window.bars)} bars -> {len(symbol_regions)} "
                f"{args.subcategory} regions using {profile.bar_size} {profile.what_to_show} "
                f"rth={profile.use_rth} bucket_timezone={args.bucket_timezone}"
            )

    generated_region_count = len(regions)
    dedupe_sources: list[Path] = []
    existing_region_ids: set[str] = set()
    if args.dedupe_existing:
        dedupe_sources = existing_region_output_paths(
            output_path,
            extra_paths=args.dedupe_path or (),
        )
        existing_region_ids = load_region_ids(dedupe_sources)
    regions, skipped_duplicate_region_ids = filter_duplicate_regions(
        regions,
        existing_region_ids=existing_region_ids,
    )
    dedupe_metadata = {
        "dedupe": {
            "enabled": bool(args.dedupe_existing),
            "sourcePaths": [str(path) for path in dedupe_sources],
            "existingRegionIds": len(existing_region_ids),
            "generatedRegions": generated_region_count,
            "skippedDuplicateRegions": len(skipped_duplicate_region_ids),
            "writtenRegions": len(regions),
            "skippedDuplicateRegionIds": sorted(set(skipped_duplicate_region_ids))[:50],
        }
    }

    if args.format == "csv":
        write_regions_csv(output_path, regions)
    else:
        write_regions_jsonl(output_path, regions)
    write_region_summary(
        summary_path,
        regions,
        tuning_subcategory=region_subcategory(args.subcategory),
        metadata=dedupe_metadata,
    )

    if not args.quiet:
        if skipped_duplicate_region_ids:
            print(f"dedupe skipped: {len(skipped_duplicate_region_ids)} regions")
        print(f"regions: {len(regions)}")
        print(f"output: {output_path}")
        print(f"summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Categorize PostgreSQL historical_bars into tuning regions "
            "for day/week/month backtest windows."
        )
    )
    add_postgres_options(parser)
    parser.add_argument(
        "--subcategory",
        choices=(
            "direction",
            "volatility",
            "index-futures-direction",
            "options-probability-map",
            "volume-orderbook",
            "trade-volume-orderbook",
        ),
        default="direction",
        help="Tuning subcategory to extract. Default: direction.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        required=True,
        help="Symbol to categorize. May be repeated.",
    )
    parser.add_argument(
        "--period",
        action="append",
        choices=("day", "week", "month"),
        default=None,
        help="Region period. May be repeated. Defaults to day, week, and month.",
    )
    parser.add_argument(
        "--direction",
        action="append",
        choices=("up", "down", "flat"),
        default=None,
        help="Direction mode only: emit this region direction. May be repeated. Defaults to all directions.",
    )
    parser.add_argument(
        "--volatility-regime",
        action="append",
        choices=("low", "medium", "high"),
        default=None,
        help="Volatility mode only: emit this individual volatility regime. May be repeated. Defaults to all regimes.",
    )
    parser.add_argument(
        "--low-volatility-threshold-pct",
        type=float,
        default=0.01,
        help="Volatility mode low/medium threshold. Default: 0.01.",
    )
    parser.add_argument(
        "--high-volatility-threshold-pct",
        type=float,
        default=0.03,
        help="Volatility mode medium/high threshold. Default: 0.03.",
    )
    parser.add_argument(
        "--market-symbol",
        default=None,
        help="Volatility mode optional market proxy symbol, for example SPX or ES.",
    )
    parser.add_argument(
        "--market-bar-size",
        default=None,
        help="Preferred market proxy bar size. Defaults to --bar-size.",
    )
    parser.add_argument(
        "--futures-symbol",
        default=None,
        help=(
            "Index futures mode optional explicit proxy symbol. If omitted, "
            "the script tries the configured proxy list for each symbol."
        ),
    )
    parser.add_argument(
        "--futures-bar-size",
        default=None,
        help="Preferred futures proxy bar size. Defaults to --bar-size.",
    )
    parser.add_argument(
        "--futures-alignment",
        action="append",
        choices=("aligned", "conflicting", "neutral_or_unknown"),
        default=None,
        help="Index futures mode only: emit this alignment. May be repeated.",
    )
    parser.add_argument(
        "--options-momentum-regime",
        action="append",
        choices=("low", "medium", "high"),
        default=None,
        help="Options mode only: emit this momentum regime. May be repeated.",
    )
    parser.add_argument(
        "--low-options-momentum-threshold",
        type=float,
        default=0.15,
        help="Options mode low/medium momentum threshold. Default: 0.15.",
    )
    parser.add_argument(
        "--high-options-momentum-threshold",
        type=float,
        default=0.45,
        help="Options mode medium/high momentum threshold. Default: 0.45.",
    )
    parser.add_argument(
        "--min-option-trades",
        type=int,
        default=1,
        help="Options mode minimum option trades required for a region. Default: 1.",
    )
    parser.add_argument(
        "--option-trade-limit",
        type=int,
        default=50000,
        help="Options mode maximum option trades to load from PostgreSQL.",
    )
    parser.add_argument(
        "--option-trade-table",
        action="append",
        default=None,
        help=(
            "Options mode PostgreSQL table name candidate. May be repeated. "
            "Defaults to option_trades/options_trades historical variants."
        ),
    )
    parser.add_argument(
        "--volume-regime",
        action="append",
        choices=("low", "normal", "high"),
        default=None,
        help=(
            "Volume/orderbook mode only: emit this relative-volume regime. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--volume-direction",
        action="append",
        choices=("up", "down", "neutral"),
        default=None,
        help=(
            "Volume/orderbook mode only: emit this volume-backed direction. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--low-relative-volume-threshold",
        type=float,
        default=0.75,
        help=(
            "Volume/orderbook mode low/normal relative-volume threshold. "
            "Default: 0.75."
        ),
    )
    parser.add_argument(
        "--high-relative-volume-threshold",
        type=float,
        default=1.25,
        help=(
            "Volume/orderbook mode normal/high relative-volume threshold. "
            "Default: 1.25."
        ),
    )
    parser.add_argument(
        "--volume-direction-threshold",
        type=float,
        default=1.1,
        help=(
            "Volume/orderbook mode relative-volume threshold for an up/down "
            "volume vote. Default: 1.1."
        ),
    )
    parser.add_argument(
        "--bucket-timezone",
        default="UTC",
        help=(
            "IANA timezone used for day/week/month bucket boundaries. "
            "Use America/New_York for NYSE-style equity calendar buckets. Default: UTC."
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
        help="Minimum bars required for a region to be emitted.",
    )
    parser.add_argument(
        "--flat-threshold-pct",
        type=float,
        default=0.0025,
        help="Absolute slope threshold for flat direction. Default: 0.0025.",
    )
    parser.add_argument(
        "--slope-severity-config",
        type=Path,
        default=None,
        help=(
            "Optional instrument-normalized slope severity JSON produced by "
            "scripts/build_instrument_slope_severity_config.py. When supplied, "
            "direction, index-futures-direction, and volume-orderbook categories "
            "use per-symbol/per-period severity thresholds instead of global defaults."
        ),
    )
    parser.add_argument(
        "--volatility-regime-config",
        type=Path,
        default=None,
        help=(
            "Optional instrument-normalized volatility regime JSON produced by "
            "scripts/build_instrument_volatility_regime_config.py. When supplied, "
            "volatility categories use per-symbol/per-period low/medium/high "
            "thresholds instead of global defaults."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Output format. Default: jsonl.",
    )
    parser.add_argument(
        "--dedupe-existing",
        dest="dedupe_existing",
        action="store_true",
        default=True,
        help=(
            "Skip generated regions whose stable regionId already appears in "
            "prior region JSONL/CSV files next to the output. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-dedupe-existing",
        dest="dedupe_existing",
        action="store_false",
        help="Do not scan prior output files for existing regionIds.",
    )
    parser.add_argument(
        "--dedupe-path",
        type=Path,
        action="append",
        default=None,
        help=(
            "Additional JSONL/CSV file or directory to scan for existing "
            "regionIds. May be repeated."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
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


def load_futures_proxy_bars(
    *,
    pg_settings: PostgresSettings,
    symbol: str,
    explicit_futures_symbol: str | None,
    preferred_bar_size: str | None,
    start_utc: str | None,
    end_utc: str | None,
    max_bars: int,
) -> tuple[str | None, list[Bar] | None, DataProfile | None]:
    candidates = [explicit_futures_symbol] if explicit_futures_symbol else index_futures_for_symbol(symbol)
    for futures_symbol in candidates:
        if not futures_symbol:
            continue
        try:
            profile = choose_data_profile(
                pg_settings,
                futures_symbol,
                preferred_bar_size=preferred_bar_size,
            )
            window = load_bars(
                pg_settings=pg_settings,
                symbol=futures_symbol,
                bar_size=profile.bar_size,
                what_to_show=profile.what_to_show,
                use_rth=profile.use_rth,
                start_utc=start_utc,
                end_utc=end_utc,
                max_bars=max_bars,
            )
            return futures_symbol, window.bars, profile
        except ValueError:
            if explicit_futures_symbol:
                raise
            continue
    return None, None, None


def existing_region_output_paths(
    output_path: Path,
    *,
    extra_paths: tuple[Path, ...] | list[Path] = (),
) -> list[Path]:
    candidates: list[Path] = []
    if output_path.parent.exists():
        candidates.extend(output_path.parent.glob("*regions.jsonl"))
        candidates.extend(output_path.parent.glob("*regions.csv"))
    for extra_path in extra_paths:
        if extra_path.is_dir():
            candidates.extend(extra_path.glob("*regions.jsonl"))
            candidates.extend(extra_path.glob("*regions.csv"))
        else:
            candidates.append(extra_path)

    output_resolved = output_path.resolve()
    output: list[Path] = []
    seen: set[Path] = set()
    for candidate in sorted(candidates):
        if candidate.suffix.lower() not in {".jsonl", ".csv"}:
            continue
        resolved = candidate.resolve()
        if resolved == output_resolved or resolved in seen:
            continue
        seen.add(resolved)
        output.append(candidate)
    return output


def region_subcategory(raw_subcategory: str) -> str:
    if raw_subcategory == "index-futures-direction":
        return "index_futures_direction"
    if raw_subcategory == "options-probability-map":
        return "options_probability_map_3d"
    if raw_subcategory in {"volume-orderbook", "trade-volume-orderbook"}:
        return "trade_volume_orderbook"
    return raw_subcategory


if __name__ == "__main__":
    raise SystemExit(main())
