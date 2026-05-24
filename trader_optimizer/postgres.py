from __future__ import annotations

import getpass
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus


@dataclass(frozen=True)
class PostgresSettings:
    conninfo: str = ""
    host: str = "127.0.0.1"
    port: int = 5432
    database: str = "trader"
    user: str | None = None
    password: str | None = None
    optuna_storage_url: str | None = None

    @property
    def display(self) -> str:
        if self.conninfo:
            return self.conninfo
        user = self.user or getpass.getuser()
        return f"postgresql://{user}@{self.host}:{self.port}/{self.database}"


def postgres_settings_from_env() -> PostgresSettings:
    return PostgresSettings(
        conninfo=os.getenv("TRADER_PG_CONNINFO", ""),
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5432")),
        database=os.getenv("PGDATABASE", "trader"),
        user=os.getenv("PGUSER") or None,
        password=os.getenv("PGPASSWORD") or None,
        optuna_storage_url=os.getenv("TRADER_OPTIMIZER_OPTUNA_STORAGE") or None,
    )


def connect_postgres(settings: PostgresSettings):
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError(
            "Missing psycopg2. Install TraderOptimizer PostgreSQL dependencies with "
            "`python3 -m pip install psycopg2-binary`."
        ) from exc

    if settings.conninfo:
        return psycopg2.connect(settings.conninfo)

    kwargs: dict[str, object] = {
        "host": settings.host,
        "port": settings.port,
        "dbname": settings.database,
    }
    if settings.user:
        kwargs["user"] = settings.user
    if settings.password:
        kwargs["password"] = settings.password
    return psycopg2.connect(**kwargs)


@contextmanager
def postgres_connection(settings: PostgresSettings):
    conn = connect_postgres(settings)
    try:
        yield conn
    finally:
        conn.close()


def optuna_storage_url(settings: PostgresSettings) -> str:
    if settings.optuna_storage_url:
        return settings.optuna_storage_url
    if settings.conninfo:
        raise ValueError(
            "Optuna requires a SQLAlchemy PostgreSQL URL. Pass --optuna-storage-url "
            "or set TRADER_OPTIMIZER_OPTUNA_STORAGE when using --pg-conninfo."
        )

    user = quote_plus(settings.user or getpass.getuser())
    password = f":{quote_plus(settings.password)}" if settings.password else ""
    host = quote_plus(settings.host)
    database = quote_plus(settings.database)
    return f"postgresql+psycopg2://{user}{password}@{host}:{settings.port}/{database}"


def ensure_optimizer_schema(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            DO $$
            BEGIN
                CREATE DOMAIN trader_currency_code AS TEXT
                    CHECK (VALUE ~ '^[A-Z][A-Z0-9_]{1,15}$');
            EXCEPTION WHEN duplicate_object THEN NULL;
            END
            $$;

            DO $$
            BEGIN
                CREATE DOMAIN trader_currency_amount AS NUMERIC(38, 12);
            EXCEPTION WHEN duplicate_object THEN NULL;
            END
            $$;

            DO $$
            BEGIN
                CREATE DOMAIN trader_position_quantity AS NUMERIC(38, 12);
            EXCEPTION WHEN duplicate_object THEN NULL;
            END
            $$;

            CREATE TABLE IF NOT EXISTS optimizer_runs (
                id BIGSERIAL PRIMARY KEY,
                study_name TEXT NOT NULL,
                run_kind TEXT NOT NULL,
                symbol TEXT NOT NULL DEFAULT '',
                strategy_name TEXT NOT NULL DEFAULT '',
                strategy_type TEXT NOT NULL DEFAULT '',
                variant TEXT NOT NULL DEFAULT '',
                output_dir TEXT NOT NULL,
                config_path TEXT NOT NULL DEFAULT '',
                summary_path TEXT NOT NULL DEFAULT '',
                best_value DOUBLE PRECISION,
                data_source TEXT NOT NULL,
                bar_size TEXT NOT NULL DEFAULT '',
                what_to_show TEXT NOT NULL DEFAULT '',
                use_rth BOOLEAN,
                first_timestamp TIMESTAMPTZ,
                last_timestamp TIMESTAMPTZ,
                bars BIGINT NOT NULL DEFAULT 0,
                metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS optimizer_trials (
                run_id BIGINT NOT NULL REFERENCES optimizer_runs(id) ON DELETE CASCADE,
                number INTEGER NOT NULL,
                value DOUBLE PRECISION,
                state TEXT NOT NULL,
                params JSONB NOT NULL,
                user_attrs JSONB NOT NULL,
                PRIMARY KEY (run_id, number)
            );

            CREATE TABLE IF NOT EXISTS optimizer_fills (
                run_id BIGINT NOT NULL REFERENCES optimizer_runs(id) ON DELETE CASCADE,
                fill_index INTEGER NOT NULL,
                tick BIGINT,
                timestamp_utc TIMESTAMPTZ,
                action TEXT NOT NULL DEFAULT '',
                step BIGINT,
                quantity trader_position_quantity,
                price trader_currency_amount,
                commission trader_currency_amount,
                PRIMARY KEY (run_id, fill_index)
            );

            CREATE TABLE IF NOT EXISTS optimizer_batch_results (
                id BIGSERIAL PRIMARY KEY,
                batch_name TEXT NOT NULL,
                name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                variant TEXT NOT NULL DEFAULT '',
                symbols TEXT[] NOT NULL DEFAULT '{}',
                source_config TEXT NOT NULL,
                status TEXT NOT NULL,
                output_dir TEXT NOT NULL DEFAULT '',
                best_config TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                best_value DOUBLE PRECISION,
                strategy_return_pct DOUBLE PRECISION,
                benchmark_return_pct DOUBLE PRECISION,
                excess_return_pct DOUBLE PRECISION,
                reason TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            ALTER TABLE optimizer_batch_results
                ADD COLUMN IF NOT EXISTS strategy_return_pct DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS benchmark_return_pct DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS excess_return_pct DOUBLE PRECISION;

            ALTER TABLE optimizer_fills
                ALTER COLUMN quantity TYPE trader_position_quantity
                    USING quantity::numeric::trader_position_quantity,
                ALTER COLUMN price TYPE trader_currency_amount
                    USING price::numeric::trader_currency_amount,
                ALTER COLUMN commission TYPE trader_currency_amount
                    USING commission::numeric::trader_currency_amount;
            """
        )
    conn.commit()


def insert_optimizer_run(
    conn,
    *,
    study_name: str,
    run_kind: str,
    symbol: str,
    strategy_name: str = "",
    strategy_type: str = "",
    variant: str = "",
    output_dir: Path,
    config_path: Path,
    summary_path: Path,
    best_value: float | None,
    data_source: str,
    bar_size: str,
    what_to_show: str,
    use_rth: int | bool,
    first_timestamp: str,
    last_timestamp: str,
    bars: int,
    metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> int:
    from psycopg2.extras import Json

    ensure_optimizer_schema(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO optimizer_runs (
                study_name, run_kind, symbol, strategy_name, strategy_type, variant,
                output_dir, config_path, summary_path, best_value, data_source,
                bar_size, what_to_show, use_rth, first_timestamp, last_timestamp,
                bars, metrics, hyperparameters
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s
            )
            RETURNING id
            """,
            (
                study_name,
                run_kind,
                symbol,
                strategy_name,
                strategy_type,
                variant,
                str(output_dir),
                str(config_path),
                str(summary_path),
                best_value,
                data_source,
                bar_size,
                what_to_show,
                bool(use_rth),
                first_timestamp,
                last_timestamp,
                bars,
                Json(metrics),
                Json(hyperparameters),
            ),
        )
        row = cursor.fetchone()
    conn.commit()
    return int(row[0])


def insert_optimizer_trials(conn, run_id: int, trials: list[Any]) -> None:
    from psycopg2.extras import Json, execute_values

    rows = [
        (
            run_id,
            trial.number,
            trial.value,
            trial.state.name,
            Json(trial.params),
            Json(trial.user_attrs),
        )
        for trial in trials
    ]
    if not rows:
        return
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO optimizer_trials (
                run_id, number, value, state, params, user_attrs
            )
            VALUES %s
            ON CONFLICT (run_id, number)
            DO UPDATE SET
                value = excluded.value,
                state = excluded.state,
                params = excluded.params,
                user_attrs = excluded.user_attrs
            """,
            rows,
        )
    conn.commit()


def insert_optimizer_fills(conn, run_id: int, fills: list[Any]) -> None:
    from psycopg2.extras import execute_values

    rows = [
        (
            run_id,
            index,
            getattr(fill, "tick", None),
            getattr(fill, "timestamp_utc", None),
            getattr(fill, "action", ""),
            getattr(fill, "step", None),
            getattr(fill, "quantity", None),
            getattr(fill, "price", None),
            getattr(fill, "commission", None),
        )
        for index, fill in enumerate(fills)
    ]
    if not rows:
        return
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO optimizer_fills (
                run_id, fill_index, tick, timestamp_utc, action,
                step, quantity, price, commission
            )
            VALUES %s
            ON CONFLICT (run_id, fill_index)
            DO UPDATE SET
                tick = excluded.tick,
                timestamp_utc = excluded.timestamp_utc,
                action = excluded.action,
                step = excluded.step,
                quantity = excluded.quantity,
                price = excluded.price,
                commission = excluded.commission
            """,
            rows,
        )
    conn.commit()


def insert_optimizer_batch_results(conn, batch_name: str, results: list[Any]) -> None:
    from psycopg2.extras import execute_values

    ensure_optimizer_schema(conn)
    rows = [
        (
            batch_name,
            result.name,
            result.strategy_type,
            result.variant,
            list(result.symbols),
            result.source_config,
            result.status,
            result.output_dir or "",
            result.best_config or "",
            result.summary or "",
            result.best_value,
            result.strategy_return_pct,
            result.benchmark_return_pct,
            result.excess_return_pct,
            result.reason or "",
        )
        for result in results
    ]
    if not rows:
        return
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO optimizer_batch_results (
                batch_name, name, strategy_type, variant, symbols, source_config,
                status, output_dir, best_config, summary, best_value,
                strategy_return_pct, benchmark_return_pct, excess_return_pct, reason
            )
            VALUES %s
            """,
            rows,
        )
    conn.commit()
