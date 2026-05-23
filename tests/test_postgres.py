from trader_optimizer.postgres import PostgresSettings, optuna_storage_url


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
