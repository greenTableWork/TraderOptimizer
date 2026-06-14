#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trader_optimizer.data import Bar, DataProfile, available_profiles, choose_data_profile, load_bars
from trader_optimizer.market_features import index_futures_for_symbol
from trader_optimizer.postgres import PostgresSettings, postgres_settings_from_env
from trader_optimizer.regime_vectors import (
    RegimeVectorContext,
    build_regime_vector,
    write_regime_vector_summary,
    write_regime_vectors_jsonl,
)
from trader_optimizer.slope_severity import SlopeSeverityConfig, build_slope_severity_config
from trader_optimizer.volatility_regime import (
    VolatilityRegimeConfig,
    build_volatility_regime_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pg_settings = postgres_settings_from_args(args)
    symbols = selected_symbols(pg_settings, args)
    if not symbols:
        parser.error("pass at least one --symbol or use --all-symbols")

    symbol_bars: dict[str, list[Bar]] = {}
    profiles: dict[str, DataProfile] = {}
    skipped_symbols: list[dict[str, object]] = []
    for symbol in symbols:
        try:
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
        except Exception as exc:
            skipped_symbols.append({"symbol": symbol, "reason": str(exc)})
            continue
        symbol_bars[symbol] = window.bars
        profiles[symbol] = profile
        if not args.quiet:
            print(
                f"{symbol}: {len(window.bars)} bars using "
                f"{profile.bar_size} {profile.what_to_show} rth={profile.use_rth}"
            )

    if not symbol_bars:
        raise RuntimeError("No symbols could be loaded from PostgreSQL historical_bars")

    profile_payloads = {
        symbol: {
            "barSize": profile.bar_size,
            "whatToShow": profile.what_to_show,
            "useRth": profile.use_rth,
        }
        for symbol, profile in profiles.items()
    }
    slope_config = SlopeSeverityConfig.from_dict(
        build_slope_severity_config(
            symbol_bars,
            periods=(args.normalization_period,),
            bucket_timezone=args.bucket_timezone,
            min_bars=args.min_bars,
            profiles=profile_payloads,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
        )
    )
    volatility_config = VolatilityRegimeConfig.from_dict(
        build_volatility_regime_config(
            symbol_bars,
            periods=(args.normalization_period,),
            bucket_timezone=args.bucket_timezone,
            min_bars=args.min_bars,
            profiles=profile_payloads,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
        )
    )

    market_symbol, market_bars, market_profile = load_market_proxy(
        pg_settings,
        args,
        available_symbol_set(pg_settings),
    )
    futures_cache: dict[str, tuple[str | None, list[Bar] | None]] = {}
    generated_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    vectors = []
    for symbol, bars in sorted(symbol_bars.items()):
        profile = profiles[symbol]
        vector_bars = latest_period_bars(
            bars,
            period=args.normalization_period,
            bucket_timezone=args.bucket_timezone,
        )
        if len(vector_bars) < 2:
            skipped_symbols.append(
                {
                    "symbol": symbol,
                    "reason": (
                        f"latest {args.normalization_period} bucket has "
                        f"{len(vector_bars)} bars"
                    ),
                }
            )
            continue
        futures_symbol, futures_bars = load_futures_proxy(
            pg_settings,
            symbol=symbol,
            preferred_bar_size=args.futures_bar_size or profile.bar_size,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            max_bars=args.max_bars,
            cache=futures_cache,
        )
        context = RegimeVectorContext(
            symbol=symbol,
            bars=vector_bars,
            profile=profile,
            market_symbol=market_symbol if market_symbol != symbol else None,
            market_bars=(
                filter_bars_to_window(market_bars, vector_bars)
                if market_symbol != symbol
                else None
            ),
            futures_symbol=futures_symbol,
            futures_bars=filter_bars_to_window(futures_bars, vector_bars),
            slope_severity_thresholds=slope_config.thresholds_for(
                symbol,
                args.normalization_period,
                bucket_timezone=args.bucket_timezone,
                bar_size=profile.bar_size,
                what_to_show=profile.what_to_show,
                use_rth=profile.use_rth,
            ),
            volatility_regime_thresholds=volatility_config.thresholds_for(
                symbol,
                args.normalization_period,
                bucket_timezone=args.bucket_timezone,
                bar_size=profile.bar_size,
                what_to_show=profile.what_to_show,
                use_rth=profile.use_rth,
            ),
            market_volatility_regime_thresholds=(
                volatility_config.thresholds_for(
                    market_symbol,
                    args.normalization_period,
                    bucket_timezone=args.bucket_timezone,
                    bar_size=market_profile.bar_size if market_profile else None,
                    what_to_show=market_profile.what_to_show if market_profile else None,
                    use_rth=market_profile.use_rth if market_profile else None,
                )
                if market_symbol and market_profile
                else None
            ),
        )
        vectors.append(build_regime_vector(context, generated_utc=generated_utc))

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output
        or Path.cwd()
        / "runs"
        / "regime_vectors"
        / f"{timestamp}_regime_vectors.jsonl"
    ).resolve()
    summary_path = (
        args.summary_output
        or output_path.with_name(f"{output_path.stem}_summary.json")
    ).resolve()
    write_regime_vectors_jsonl(output_path, vectors)
    write_regime_vector_summary(
        summary_path,
        vectors,
        skipped_symbols=skipped_symbols,
    )
    if not args.quiet:
        print(f"vectors: {len(vectors)}")
        if skipped_symbols:
            print(f"skipped: {len(skipped_symbols)}")
        print(f"output: {output_path}")
        print(f"summary: {summary_path}")
    return 0


def selected_symbols(pg_settings: PostgresSettings, args: argparse.Namespace) -> tuple[str, ...]:
    symbols = tuple(args.symbol or ())
    if args.all_symbols:
        symbols = tuple(profile.symbol for profile in available_profiles(pg_settings))
    symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    return symbols


def available_symbol_set(pg_settings: PostgresSettings) -> set[str]:
    return {profile.symbol.upper() for profile in available_profiles(pg_settings)}


def load_market_proxy(
    pg_settings: PostgresSettings,
    args: argparse.Namespace,
    available_symbols: set[str],
) -> tuple[str | None, list[Bar] | None, DataProfile | None]:
    candidates = [args.market_symbol] if args.market_symbol else ["SPX", "ES", "VIX"]
    for symbol in candidates:
        if not symbol:
            continue
        normalized = symbol.upper()
        if normalized not in available_symbols:
            continue
        try:
            profile = choose_data_profile(
                pg_settings,
                normalized,
                preferred_bar_size=args.market_bar_size or args.bar_size,
            )
            window = load_bars(
                pg_settings=pg_settings,
                symbol=normalized,
                bar_size=profile.bar_size,
                what_to_show=profile.what_to_show,
                use_rth=profile.use_rth,
                start_utc=args.start_utc,
                end_utc=args.end_utc,
                max_bars=args.max_bars,
            )
            if not args.quiet:
                print(
                    f"{normalized}: {len(window.bars)} market bars using "
                    f"{profile.bar_size} {profile.what_to_show} rth={profile.use_rth}"
                )
            return normalized, window.bars, profile
        except ValueError:
            continue
    return None, None, None


def load_futures_proxy(
    pg_settings: PostgresSettings,
    *,
    symbol: str,
    preferred_bar_size: str | None,
    start_utc: str | None,
    end_utc: str | None,
    max_bars: int,
    cache: dict[str, tuple[str | None, list[Bar] | None]],
) -> tuple[str | None, list[Bar] | None]:
    for futures_symbol in index_futures_for_symbol(symbol):
        if futures_symbol in cache:
            cached_symbol, cached_bars = cache[futures_symbol]
            if cached_bars is not None:
                return cached_symbol, cached_bars
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
            cache[futures_symbol] = (futures_symbol, window.bars)
            return futures_symbol, window.bars
        except ValueError:
            cache[futures_symbol] = (None, None)
            continue
    return None, None


def latest_period_bars(
    bars: list[Bar],
    *,
    period: str,
    bucket_timezone: str,
) -> list[Bar]:
    if not bars:
        return []
    bucket_zone = bucket_zone_from_name(bucket_timezone)
    sorted_bars = sorted(bars, key=lambda bar: parse_utc(bar.timestamp_utc))
    latest_bucket = period_bucket(
        parse_utc(sorted_bars[-1].timestamp_utc),
        period,
        bucket_zone,
    )
    return [
        bar
        for bar in sorted_bars
        if period_bucket(parse_utc(bar.timestamp_utc), period, bucket_zone) == latest_bucket
    ]


def filter_bars_to_window(
    bars: list[Bar] | None,
    reference_bars: list[Bar],
) -> list[Bar] | None:
    if not bars or not reference_bars:
        return None
    start = parse_utc(reference_bars[0].timestamp_utc)
    end = parse_utc(reference_bars[-1].timestamp_utc)
    output = [
        bar
        for bar in bars
        if start <= parse_utc(bar.timestamp_utc) <= end
    ]
    return output or None


def bucket_zone_from_name(bucket_timezone: str) -> ZoneInfo:
    if bucket_timezone.upper() == "UTC":
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(bucket_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unsupported bucket timezone {bucket_timezone!r}") from exc


def period_bucket(timestamp: datetime, period: str, bucket_zone: ZoneInfo) -> str:
    local_timestamp = timestamp.astimezone(bucket_zone)
    if period == "day":
        return local_timestamp.date().isoformat()
    if period == "week":
        iso_year, iso_week, _ = local_timestamp.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if period == "month":
        return f"{local_timestamp.year}-{local_timestamp.month:02d}"
    raise ValueError(f"Unsupported period {period!r}")


def parse_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build current instrument regime vectors for PostgreSQL historical_bars symbols."
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
        help="Build vectors for every symbol with PostgreSQL historical bars.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Limit the number of selected symbols. Intended for smoke tests.",
    )
    parser.add_argument(
        "--bar-size",
        default=None,
        help="Preferred bar size. Defaults to the best available PostgreSQL profile.",
    )
    parser.add_argument(
        "--market-symbol",
        default=None,
        help="Market proxy symbol. Defaults to the first available of SPX, ES, VIX.",
    )
    parser.add_argument(
        "--market-bar-size",
        default=None,
        help="Preferred market proxy bar size. Defaults to --bar-size.",
    )
    parser.add_argument(
        "--futures-bar-size",
        default=None,
        help="Preferred index futures bar size. Defaults to each symbol's selected bar size.",
    )
    parser.add_argument(
        "--normalization-period",
        choices=("day", "week", "month"),
        default="day",
        help="Period used for instrument-normalized slope and volatility thresholds.",
    )
    parser.add_argument(
        "--bucket-timezone",
        default="UTC",
        help="IANA timezone used for normalization buckets. Default: UTC.",
    )
    parser.add_argument("--start-utc", default=None)
    parser.add_argument("--end-utc", default=None)
    parser.add_argument(
        "--max-bars",
        type=int,
        default=5000,
        help="Use latest N bars per symbol. Default: 5000.",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=2,
        help="Minimum bars required for normalization buckets. Default: 2.",
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


if __name__ == "__main__":
    raise SystemExit(main())
