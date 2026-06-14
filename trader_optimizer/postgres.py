from __future__ import annotations

import getpass
import os
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
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


def _numeric_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


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
                best_value NUMERIC(38, 12),
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
                value NUMERIC(38, 12),
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
                best_value NUMERIC(38, 12),
                strategy_return_pct NUMERIC(38, 12),
                benchmark_return_pct NUMERIC(38, 12),
                excess_return_pct NUMERIC(38, 12),
                reason TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS optimizer_sweep_candidates (
                id BIGSERIAL PRIMARY KEY,
                report_name TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                selected BOOLEAN NOT NULL,
                total_return NUMERIC(38, 12) NOT NULL,
                max_drawdown NUMERIC(38, 12) NOT NULL,
                sharpe NUMERIC(38, 12) NOT NULL,
                trade_count BIGINT NOT NULL,
                bars BIGINT NOT NULL,
                config JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS optimizer_sweep_candidates_report_idx
                ON optimizer_sweep_candidates (report_name, strategy_id);

            DO $$
            BEGIN
                IF to_regclass('public.historical_bars') IS NOT NULL
                    AND EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'historical_bars'
                          AND column_name IN ('open', 'high', 'low', 'close', 'volume', 'wap')
                          AND (
                              data_type <> 'numeric'
                              OR numeric_precision IS DISTINCT FROM 38
                              OR numeric_scale IS DISTINCT FROM 12
                          )
                    )
                THEN
                    ALTER TABLE historical_bars
                        ALTER COLUMN open TYPE NUMERIC(38, 12)
                            USING open::numeric(38, 12),
                        ALTER COLUMN high TYPE NUMERIC(38, 12)
                            USING high::numeric(38, 12),
                        ALTER COLUMN low TYPE NUMERIC(38, 12)
                            USING low::numeric(38, 12),
                        ALTER COLUMN close TYPE NUMERIC(38, 12)
                            USING close::numeric(38, 12),
                        ALTER COLUMN volume TYPE NUMERIC(38, 12)
                            USING volume::numeric(38, 12),
                        ALTER COLUMN wap TYPE NUMERIC(38, 12)
                            USING wap::numeric(38, 12);
                END IF;
            END
            $$;

            ALTER TABLE optimizer_runs
                ADD COLUMN IF NOT EXISTS best_value NUMERIC(38, 12);

            ALTER TABLE optimizer_trials
                ADD COLUMN IF NOT EXISTS value NUMERIC(38, 12);

            ALTER TABLE optimizer_batch_results
                ADD COLUMN IF NOT EXISTS best_value NUMERIC(38, 12),
                ADD COLUMN IF NOT EXISTS strategy_return_pct NUMERIC(38, 12),
                ADD COLUMN IF NOT EXISTS benchmark_return_pct NUMERIC(38, 12),
                ADD COLUMN IF NOT EXISTS excess_return_pct NUMERIC(38, 12);

            ALTER TABLE optimizer_sweep_candidates
                ADD COLUMN IF NOT EXISTS total_return NUMERIC(38, 12),
                ADD COLUMN IF NOT EXISTS max_drawdown NUMERIC(38, 12),
                ADD COLUMN IF NOT EXISTS sharpe NUMERIC(38, 12);

            ALTER TABLE optimizer_runs
                ALTER COLUMN best_value TYPE NUMERIC(38, 12)
                    USING best_value::numeric;

            ALTER TABLE optimizer_trials
                ALTER COLUMN value TYPE NUMERIC(38, 12)
                    USING value::numeric;

            ALTER TABLE optimizer_batch_results
                ALTER COLUMN best_value TYPE NUMERIC(38, 12)
                    USING best_value::numeric,
                ALTER COLUMN strategy_return_pct TYPE NUMERIC(38, 12)
                    USING strategy_return_pct::numeric,
                ALTER COLUMN benchmark_return_pct TYPE NUMERIC(38, 12)
                    USING benchmark_return_pct::numeric,
                ALTER COLUMN excess_return_pct TYPE NUMERIC(38, 12)
                    USING excess_return_pct::numeric;

            ALTER TABLE optimizer_sweep_candidates
                ALTER COLUMN total_return TYPE NUMERIC(38, 12)
                    USING total_return::numeric,
                ALTER COLUMN max_drawdown TYPE NUMERIC(38, 12)
                    USING max_drawdown::numeric,
                ALTER COLUMN sharpe TYPE NUMERIC(38, 12)
                    USING sharpe::numeric;

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


def ensure_live_regime_schema(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS regime_detector_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                detector_spec_version TEXT NOT NULL DEFAULT '',
                started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at TIMESTAMPTZ,
                status TEXT NOT NULL DEFAULT 'running',
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            );

            CREATE TABLE IF NOT EXISTS live_regime_vectors (
                symbol TEXT NOT NULL,
                window_key TEXT NOT NULL,
                detector_run_id TEXT NOT NULL,
                regime_cell_id TEXT NOT NULL,
                vector JSONB NOT NULL,
                detection JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (symbol, window_key)
            );

            CREATE TABLE IF NOT EXISTS regime_vector_history (
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                window_key TEXT NOT NULL,
                detector_run_id TEXT NOT NULL,
                regime_cell_id TEXT NOT NULL,
                vector JSONB NOT NULL,
                detection JSONB NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS regime_vector_history_symbol_time_idx
                ON regime_vector_history (symbol, observed_at DESC);

            CREATE TABLE IF NOT EXISTS regime_transition_events (
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                window_key TEXT NOT NULL,
                detector_run_id TEXT NOT NULL,
                from_regime_cell_id TEXT NOT NULL DEFAULT '',
                to_regime_cell_id TEXT NOT NULL,
                transition_status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                confidence NUMERIC(12, 8),
                detection JSONB NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS strategy_regime_config_map (
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                regime_cell_id TEXT NOT NULL,
                regime_cell JSONB NOT NULL,
                strategy_name TEXT NOT NULL,
                config_path TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                excess_return_pct NUMERIC(38, 12),
                spx_excess_return_pct NUMERIC(38, 12),
                same_stock_excess_return_pct NUMERIC(38, 12),
                source TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (symbol, regime_cell_id, strategy_name, config_path)
            );

            CREATE INDEX IF NOT EXISTS strategy_regime_config_map_lookup_idx
                ON strategy_regime_config_map (
                    symbol, regime_cell_id, validation_status
                );

            CREATE TABLE IF NOT EXISTS strategy_selection_decisions (
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                window_key TEXT NOT NULL,
                detector_run_id TEXT NOT NULL,
                regime_cell_id TEXT NOT NULL,
                selection_status TEXT NOT NULL,
                selected_config_path TEXT NOT NULL DEFAULT '',
                decision JSONB NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
    conn.commit()


def insert_live_regime_detections(
    conn,
    *,
    run_id: str,
    mode: str,
    detector_spec_version: str,
    detections: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    from psycopg2.extras import Json, execute_values

    ensure_live_regime_schema(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO regime_detector_runs (
                run_id, mode, detector_spec_version, status, metadata
            )
            VALUES (%s, %s, %s, 'completed', %s)
            ON CONFLICT (run_id)
            DO UPDATE SET
                mode = excluded.mode,
                detector_spec_version = excluded.detector_spec_version,
                completed_at = now(),
                status = excluded.status,
                metadata = excluded.metadata
            """,
            (
                run_id,
                mode,
                detector_spec_version,
                Json(metadata or {}),
            ),
        )

    live_rows = []
    history_rows = []
    transition_rows = []
    selection_rows = []
    for detection in detections:
        symbol = str(detection.get("symbol") or "")
        state = detection.get("state")
        transition = detection.get("transition")
        selection = detection.get("strategySelection")
        source_vector = detection.get("sourceVector")
        window_key = ""
        if isinstance(state, dict):
            window_key = str(state.get("stateKey") or "")
        regime_cell = str(detection.get("activeRegimeCellId") or "")
        live_rows.append(
            (
                symbol,
                window_key,
                run_id,
                regime_cell,
                Json(source_vector if isinstance(source_vector, dict) else {}),
                Json(detection),
            )
        )
        history_rows.append(
            (
                symbol,
                window_key,
                run_id,
                regime_cell,
                Json(source_vector if isinstance(source_vector, dict) else {}),
                Json(detection),
            )
        )
        if isinstance(transition, dict) and transition.get("status") in {
            "initialized",
            "switched",
        }:
            transition_rows.append(
                (
                    symbol,
                    window_key,
                    run_id,
                    str(transition.get("fromRegimeCellId") or ""),
                    str(transition.get("toRegimeCellId") or regime_cell),
                    str(transition.get("status") or ""),
                    str(transition.get("reason") or ""),
                    _numeric_or_none(
                        state.get("changePointConfidence")
                        if isinstance(state, dict)
                        else None
                    ),
                    Json(detection),
                )
            )
        selected_config = ""
        selection_status = "unknown"
        if isinstance(selection, dict):
            selection_status = str(selection.get("status") or "unknown")
            selected = selection.get("selected")
            if isinstance(selected, dict):
                selected_config = str(selected.get("configPath") or "")
        selection_rows.append(
            (
                symbol,
                window_key,
                run_id,
                regime_cell,
                selection_status,
                selected_config,
                Json(selection if isinstance(selection, dict) else {}),
            )
        )

    with conn.cursor() as cursor:
        if live_rows:
            execute_values(
                cursor,
                """
                INSERT INTO live_regime_vectors (
                    symbol, window_key, detector_run_id, regime_cell_id,
                    vector, detection
                )
                VALUES %s
                ON CONFLICT (symbol, window_key)
                DO UPDATE SET
                    detector_run_id = excluded.detector_run_id,
                    regime_cell_id = excluded.regime_cell_id,
                    vector = excluded.vector,
                    detection = excluded.detection,
                    updated_at = now()
                """,
                live_rows,
            )
        if history_rows:
            execute_values(
                cursor,
                """
                INSERT INTO regime_vector_history (
                    symbol, window_key, detector_run_id, regime_cell_id,
                    vector, detection
                )
                VALUES %s
                """,
                history_rows,
            )
        if transition_rows:
            execute_values(
                cursor,
                """
                INSERT INTO regime_transition_events (
                    symbol, window_key, detector_run_id, from_regime_cell_id,
                    to_regime_cell_id, transition_status, reason, confidence,
                    detection
                )
                VALUES %s
                """,
                transition_rows,
            )
        if selection_rows:
            execute_values(
                cursor,
                """
                INSERT INTO strategy_selection_decisions (
                    symbol, window_key, detector_run_id, regime_cell_id,
                    selection_status, selected_config_path, decision
                )
                VALUES %s
                """,
                selection_rows,
            )
    conn.commit()


def insert_strategy_regime_config_map(
    conn,
    *,
    entries: list[dict[str, Any]],
) -> None:
    from psycopg2.extras import Json, execute_values

    ensure_live_regime_schema(conn)
    rows = []
    for entry in entries:
        rows.append(
            (
                str(entry.get("symbol") or "").upper(),
                str(entry.get("regimeCellId") or ""),
                Json(entry.get("regimeCell") if isinstance(entry.get("regimeCell"), dict) else {}),
                str(entry.get("strategyName") or ""),
                str(entry.get("configPath") or ""),
                str(entry.get("validationStatus") or ""),
                _numeric_or_none(entry.get("excessReturnPct")),
                _numeric_or_none(entry.get("spxExcessReturnPct")),
                _numeric_or_none(entry.get("sameStockExcessReturnPct")),
                str(entry.get("source") or ""),
            )
        )
    if not rows:
        return
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO strategy_regime_config_map (
                symbol, regime_cell_id, regime_cell, strategy_name, config_path,
                validation_status, excess_return_pct, spx_excess_return_pct,
                same_stock_excess_return_pct, source
            )
            VALUES %s
            ON CONFLICT (symbol, regime_cell_id, strategy_name, config_path)
            DO UPDATE SET
                regime_cell = excluded.regime_cell,
                validation_status = excluded.validation_status,
                excess_return_pct = excluded.excess_return_pct,
                spx_excess_return_pct = excluded.spx_excess_return_pct,
                same_stock_excess_return_pct = excluded.same_stock_excess_return_pct,
                source = excluded.source
            """,
            rows,
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
    use_rth: int | bool | None,
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
                _numeric_or_none(best_value),
                data_source,
                bar_size,
                what_to_show,
                None if use_rth is None else bool(use_rth),
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
            _numeric_or_none(trial.value),
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
            _numeric_or_none(getattr(fill, "quantity", None)),
            _numeric_or_none(getattr(fill, "price", None)),
            _numeric_or_none(getattr(fill, "commission", None)),
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
            _numeric_or_none(result.best_value),
            _numeric_or_none(result.strategy_return_pct),
            _numeric_or_none(result.benchmark_return_pct),
            _numeric_or_none(result.excess_return_pct),
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


def insert_optimizer_sweep_report(
    conn,
    report_name: str,
    candidates: list[Any],
    selected: list[Any],
) -> None:
    import json

    from psycopg2.extras import Json, execute_values

    ensure_optimizer_schema(conn)
    selected_keys = {
        (
            str(candidate.strategy_id),
            json.dumps(candidate.config, sort_keys=True),
        )
        for candidate in selected
    }
    rows = []
    for candidate in candidates:
        config_json = json.dumps(candidate.config, sort_keys=True)
        rows.append(
            (
                report_name,
                str(candidate.strategy_id),
                (str(candidate.strategy_id), config_json) in selected_keys,
                _numeric_or_none(candidate.total_return),
                _numeric_or_none(candidate.max_drawdown),
                _numeric_or_none(candidate.sharpe),
                int(candidate.trade_count),
                int(candidate.bars),
                Json(candidate.config),
            )
        )
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM optimizer_sweep_candidates WHERE report_name = %s",
            (report_name,),
        )
        if rows:
            execute_values(
                cursor,
                """
                INSERT INTO optimizer_sweep_candidates (
                    report_name, strategy_id, selected, total_return, max_drawdown,
                    sharpe, trade_count, bars, config
                )
                VALUES %s
                """,
                rows,
            )
    conn.commit()
