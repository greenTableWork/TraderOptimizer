from decimal import Decimal

from trader_optimizer.postgres import (
    PostgresSettings,
    _numeric_or_none,
    ensure_optimizer_schema,
    optuna_storage_url,
)


def test_postgres_display_defaults_to_local_database() -> None:
    settings = PostgresSettings(user="trader_user", database="trader_test")

    assert settings.display == "postgresql://trader_user@127.0.0.1:5432/trader_test"


def test_optuna_storage_url_is_postgresql_sqlalchemy_url() -> None:
    settings = PostgresSettings(
        host="localhost",
        port=5433,
        database="trader test",
        user="trader user",
        password="p@ss word",
    )

    assert (
        optuna_storage_url(settings)
        == "postgresql+psycopg2://trader+user:p%40ss+word@localhost:5433/trader+test"
    )


def test_optimizer_schema_uses_money_and_position_domains() -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, sql: str) -> None:
            self.statements.append(sql)

    class FakeConnection:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()
            self.commits = 0

        def cursor(self):
            return self.cursor_instance

        def commit(self) -> None:
            self.commits += 1

    conn = FakeConnection()

    ensure_optimizer_schema(conn)

    schema_sql = conn.cursor_instance.statements[0]
    assert "CREATE DOMAIN trader_currency_amount" in schema_sql
    assert "CREATE DOMAIN trader_position_quantity" in schema_sql
    assert "quantity trader_position_quantity" in schema_sql
    assert "price trader_currency_amount" in schema_sql
    assert "commission trader_currency_amount" in schema_sql
    assert "ALTER TABLE optimizer_fills" in schema_sql
    assert conn.commits == 1


def test_optimizer_schema_uses_numeric_storage_not_double_precision() -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, sql: str) -> None:
            self.statements.append(sql)

    class FakeConnection:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()

        def cursor(self):
            return self.cursor_instance

        def commit(self) -> None:
            pass

    conn = FakeConnection()

    ensure_optimizer_schema(conn)

    schema_sql = conn.cursor_instance.statements[0]
    assert "DOUBLE PRECISION" not in schema_sql
    assert "best_value NUMERIC(38, 12)" in schema_sql
    assert "value NUMERIC(38, 12)" in schema_sql
    assert "strategy_return_pct NUMERIC(38, 12)" in schema_sql
    assert "optimizer_sweep_candidates" in schema_sql
    assert "total_return NUMERIC(38, 12)" in schema_sql
    assert "ALTER COLUMN best_value TYPE NUMERIC(38, 12)" in schema_sql
    assert "ALTER COLUMN value TYPE NUMERIC(38, 12)" in schema_sql
    assert "ALTER COLUMN strategy_return_pct TYPE NUMERIC(38, 12)" in schema_sql


def test_postgres_numeric_values_use_decimal_strings() -> None:
    assert _numeric_or_none(None) is None
    assert _numeric_or_none(0.1) == Decimal("0.1")
    assert _numeric_or_none(Decimal("12.3400")) == Decimal("12.3400")
