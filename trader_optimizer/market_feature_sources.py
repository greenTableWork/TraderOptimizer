from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trader_optimizer.data import Bar, choose_data_profile, load_bars
from trader_optimizer.market_features import (
    OptionTrade,
    build_market_feature_summary,
    index_futures_for_symbols,
)
from trader_optimizer.postgres import PostgresSettings, postgres_connection


DEFAULT_OPTION_TRADE_TABLES = (
    "option_trades",
    "options_trades",
    "historical_option_trades",
    "historical_options_trades",
)


@dataclass(frozen=True)
class OptionTradeTableMapping:
    schema: str
    table: str
    underlying_column: str
    time_column: str
    side_column: str
    premium_column: str | None
    price_column: str | None
    volume_column: str | None
    expiration_days_column: str | None
    expiration_date_column: str | None
    moneyness_column: str | None
    strike_column: str | None
    underlying_price_column: str | None
    open_interest_column: str | None
    implied_volatility_column: str | None


def build_market_feature_summary_from_postgres(
    pg_settings: PostgresSettings,
    symbol_bars: Mapping[str, Sequence[Bar]],
    *,
    preferred_bar_size: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    max_bars: int = 50000,
    option_trade_limit: int = 50000,
) -> dict[str, object]:
    symbols = tuple(symbol_bars.keys())
    index_futures_bars = load_available_index_futures_bars(
        pg_settings,
        symbols,
        preferred_bar_size=preferred_bar_size,
        start_utc=start_utc,
        end_utc=end_utc,
        max_bars=max_bars,
    )
    option_trades = load_available_option_trades(
        pg_settings,
        symbols,
        start_utc=start_utc,
        end_utc=end_utc,
        limit=option_trade_limit,
    )
    return build_market_feature_summary(
        symbol_bars,
        index_futures_bars=index_futures_bars,
        option_trades=option_trades,
    )


def load_available_index_futures_bars(
    pg_settings: PostgresSettings,
    symbols: Sequence[str],
    *,
    preferred_bar_size: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    max_bars: int = 50000,
) -> dict[str, list[Bar]]:
    futures_bars: dict[str, list[Bar]] = {}
    for future_symbol in index_futures_for_symbols(symbols):
        try:
            profile = choose_data_profile(
                pg_settings,
                future_symbol,
                preferred_bar_size=preferred_bar_size,
            )
            futures_bars[future_symbol] = load_bars(
                pg_settings=pg_settings,
                symbol=future_symbol,
                bar_size=profile.bar_size,
                what_to_show=profile.what_to_show,
                use_rth=profile.use_rth,
                start_utc=start_utc,
                end_utc=end_utc,
                max_bars=max_bars,
            ).bars
        except ValueError:
            continue
    return futures_bars


def load_available_option_trades(
    pg_settings: PostgresSettings,
    underlyings: Sequence[str],
    *,
    start_utc: str | None = None,
    end_utc: str | None = None,
    limit: int = 50000,
    table_names: Sequence[str] = DEFAULT_OPTION_TRADE_TABLES,
) -> list[OptionTrade]:
    normalized_underlyings = sorted(
        {str(underlying).upper() for underlying in underlyings if str(underlying)}
    )
    if not normalized_underlyings or limit <= 0:
        return []

    with postgres_connection(pg_settings) as connection:
        mapping = _discover_option_trade_table_mapping(connection, table_names)
        if mapping is None:
            return []
        query, params = _option_trade_query(
            mapping,
            normalized_underlyings,
            start_utc=start_utc,
            end_utc=end_utc,
            limit=limit,
        )
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    trades: list[OptionTrade] = []
    for row in rows:
        trade = _option_trade_from_row(row)
        if trade is not None:
            trades.append(trade)
    return trades


def _discover_option_trade_table_mapping(
    connection: Any,
    table_names: Sequence[str],
) -> OptionTradeTableMapping | None:
    lower_names = [name.lower() for name in table_names]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
              AND lower(table_name) = ANY(%s)
            """,
            (lower_names,),
        )
        table_rows = cursor.fetchall()

    def table_rank(row: tuple[str, str]) -> tuple[int, int, str]:
        schema, table = row
        try:
            name_rank = lower_names.index(str(table).lower())
        except ValueError:
            name_rank = len(lower_names)
        schema_rank = 0 if schema == "public" else 1
        return (name_rank, schema_rank, f"{schema}.{table}")

    for schema, table in sorted(table_rows, key=table_rank):
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
                """,
                (schema, table),
            )
            columns = {str(row[0]).lower(): str(row[0]) for row in cursor.fetchall()}
        mapping = _option_mapping_from_columns(str(schema), str(table), columns)
        if mapping is not None:
            return mapping
    return None


def _option_mapping_from_columns(
    schema: str,
    table: str,
    columns: Mapping[str, str],
) -> OptionTradeTableMapping | None:
    underlying = _pick_column(
        columns,
        ("underlying", "underlying_symbol", "root_symbol", "root", "symbol"),
    )
    time = _pick_column(
        columns,
        ("trade_time_utc", "timestamp_utc", "event_time_utc", "trade_time", "timestamp", "time"),
    )
    side = _pick_column(
        columns,
        ("side", "right", "option_side", "option_type", "contract_type", "put_call", "cp"),
    )
    premium = _pick_column(
        columns,
        ("premium", "trade_premium", "gross_premium", "notional", "trade_value"),
    )
    price = _pick_column(columns, ("price", "trade_price", "last", "mark", "mid"))
    volume = _pick_column(
        columns,
        ("volume", "trade_volume", "size", "quantity", "contract_volume"),
    )
    expiration_days = _pick_column(
        columns,
        ("expiration_days", "dte", "days_to_expiration"),
    )
    expiration_date = _pick_column(
        columns,
        ("expiration_date", "expiry_date", "expiry", "expiration", "last_trade_date"),
    )
    moneyness = _pick_column(columns, ("strike_moneyness", "moneyness"))
    strike = _pick_column(columns, ("strike", "strike_price"))
    underlying_price = _pick_column(
        columns,
        ("underlying_price", "underlying_last", "spot_price", "spot"),
    )
    if not all((underlying, time, side)):
        return None
    if premium is None and price is None:
        return None
    if expiration_days is None and expiration_date is None:
        return None
    if moneyness is None and (strike is None or underlying_price is None):
        return None
    return OptionTradeTableMapping(
        schema=schema,
        table=table,
        underlying_column=underlying,
        time_column=time,
        side_column=side,
        premium_column=premium,
        price_column=price,
        volume_column=volume,
        expiration_days_column=expiration_days,
        expiration_date_column=expiration_date,
        moneyness_column=moneyness,
        strike_column=strike,
        underlying_price_column=underlying_price,
        open_interest_column=_pick_column(columns, ("open_interest", "openinterest", "oi")),
        implied_volatility_column=_pick_column(
            columns,
            ("implied_volatility", "impliedvolatility", "iv"),
        ),
    )


def _option_trade_query(
    mapping: OptionTradeTableMapping,
    underlyings: Sequence[str],
    *,
    start_utc: str | None,
    end_utc: str | None,
    limit: int,
) -> tuple[Any, list[object]]:
    from psycopg2 import sql

    time_expr = sql.SQL("{}::timestamptz").format(sql.Identifier(mapping.time_column))
    volume_expr = (
        _numeric_column_sql(mapping.volume_column)
        if mapping.volume_column is not None
        else sql.SQL("1.0")
    )
    premium_expr = (
        _numeric_column_sql(mapping.premium_column)
        if mapping.premium_column is not None
        else sql.SQL("({} * {})").format(
            _numeric_column_sql(mapping.price_column),
            volume_expr,
        )
    )
    expiration_expr = (
        _numeric_column_sql(mapping.expiration_days_column)
        if mapping.expiration_days_column is not None
        else sql.SQL("({}::date - {}::date)").format(
            sql.Identifier(mapping.expiration_date_column),
            time_expr,
        )
    )
    moneyness_expr = (
        _numeric_column_sql(mapping.moneyness_column)
        if mapping.moneyness_column is not None
        else sql.SQL("({} / NULLIF({}, 0))").format(
            _numeric_column_sql(mapping.strike_column),
            _numeric_column_sql(mapping.underlying_price_column),
        )
    )
    open_interest_expr = (
        _numeric_column_sql(mapping.open_interest_column)
        if mapping.open_interest_column is not None
        else sql.SQL("NULL")
    )
    implied_volatility_expr = (
        _numeric_column_sql(mapping.implied_volatility_column)
        if mapping.implied_volatility_column is not None
        else sql.SQL("NULL")
    )

    where_parts = [
        sql.SQL("upper({}::text) = ANY(%s)").format(
            sql.Identifier(mapping.underlying_column)
        )
    ]
    params: list[object] = [list(underlyings)]
    if start_utc:
        where_parts.append(sql.SQL("{} >= %s::timestamptz").format(time_expr))
        params.append(start_utc)
    if end_utc:
        where_parts.append(sql.SQL("{} <= %s::timestamptz").format(time_expr))
        params.append(end_utc)
    params.append(limit)

    query = sql.SQL(
        """
        SELECT
            upper({underlying}::text) AS underlying,
            to_char({time_expr} AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"') AS trade_time_utc,
            {expiration_expr} AS expiration_days,
            {moneyness_expr} AS strike_moneyness,
            {side}::text AS side,
            {premium_expr} AS premium,
            {volume_expr} AS volume,
            {open_interest_expr} AS open_interest,
            {implied_volatility_expr} AS implied_volatility
        FROM {table}
        WHERE {where_clause}
        ORDER BY {time_expr} DESC
        LIMIT %s
        """
    ).format(
        underlying=sql.Identifier(mapping.underlying_column),
        time_expr=time_expr,
        expiration_expr=expiration_expr,
        moneyness_expr=moneyness_expr,
        side=sql.Identifier(mapping.side_column),
        premium_expr=premium_expr,
        volume_expr=volume_expr,
        open_interest_expr=open_interest_expr,
        implied_volatility_expr=implied_volatility_expr,
        table=sql.Identifier(mapping.schema, mapping.table),
        where_clause=sql.SQL(" AND ").join(where_parts),
    )
    return query, params


def _option_trade_from_row(row: Sequence[Any]) -> OptionTrade | None:
    try:
        side = _normalize_option_side(str(row[4]))
        expiration_days = int(float(row[2]))
        moneyness = float(row[3])
        premium = float(row[5])
        volume = float(row[6])
        if not side or volume <= 0:
            return None
        return OptionTrade(
            underlying=str(row[0]).upper(),
            trade_time_utc=str(row[1]),
            expiration_days=max(0, expiration_days),
            strike_moneyness=moneyness,
            side=side,
            premium=premium,
            volume=volume,
            open_interest=_float_or_none(row[7]),
            implied_volatility=_float_or_none(row[8]),
        )
    except (TypeError, ValueError):
        return None


def _numeric_column_sql(column: str | None) -> Any:
    from psycopg2 import sql

    if column is None:
        return sql.SQL("NULL")
    return sql.SQL("NULLIF({}::text, '')::double precision").format(
        sql.Identifier(column)
    )


def _pick_column(
    columns: Mapping[str, str],
    aliases: Sequence[str],
) -> str | None:
    for alias in aliases:
        column = columns.get(alias.lower())
        if column is not None:
            return column
    return None


def _normalize_option_side(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"C", "CALL", "CE"}:
        return "CALL"
    if normalized in {"P", "PUT", "PE"}:
        return "PUT"
    return normalized


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
