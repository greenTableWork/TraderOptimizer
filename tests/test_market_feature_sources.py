from trader_optimizer.data import Bar, BarWindow, DataProfile
from trader_optimizer.market_feature_sources import (
    _option_mapping_from_columns,
    _option_trade_from_row,
    load_available_index_futures_bars,
)
from trader_optimizer.postgres import PostgresSettings


def _bar(index: int, close: float) -> Bar:
    return Bar(
        timestamp_utc=f"2026-05-26T13:{30 + index:02d}:00+00:00",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100,
    )


def test_load_available_index_futures_bars_uses_symbol_proxy_set(monkeypatch) -> None:
    loaded_symbols: list[str] = []

    def fake_choose_data_profile(pg_settings, symbol, preferred_bar_size=None):
        if symbol == "ES":
            raise ValueError("ES is not loaded")
        return DataProfile(
            symbol=symbol,
            bar_size=preferred_bar_size or "1 min",
            what_to_show="TRADES",
            use_rth=1,
            count=2,
            first_timestamp="2026-05-26T13:30:00+00:00",
            last_timestamp="2026-05-26T13:31:00+00:00",
        )

    def fake_load_bars(**kwargs):
        loaded_symbols.append(kwargs["symbol"])
        return BarWindow(
            symbol=kwargs["symbol"],
            bar_size=kwargs["bar_size"],
            what_to_show=kwargs["what_to_show"],
            use_rth=kwargs["use_rth"],
            data_source="test",
            bars=[_bar(0, 100.0), _bar(1, 101.0)],
        )

    monkeypatch.setattr(
        "trader_optimizer.market_feature_sources.choose_data_profile",
        fake_choose_data_profile,
    )
    monkeypatch.setattr(
        "trader_optimizer.market_feature_sources.load_bars",
        fake_load_bars,
    )

    bars = load_available_index_futures_bars(
        PostgresSettings(),
        ["AAPL"],
        preferred_bar_size="1 min",
    )

    assert loaded_symbols == ["NQ"]
    assert list(bars) == ["NQ"]
    assert len(bars["NQ"]) == 2


def test_option_mapping_accepts_price_size_and_expiration_date_shape() -> None:
    mapping = _option_mapping_from_columns(
        "public",
        "option_trades",
        {
            "underlying_symbol": "underlying_symbol",
            "trade_time_utc": "trade_time_utc",
            "right": "right",
            "price": "price",
            "size": "size",
            "expiration_date": "expiration_date",
            "strike": "strike",
            "underlying_price": "underlying_price",
            "open_interest": "open_interest",
            "iv": "iv",
        },
    )

    assert mapping is not None
    assert mapping.price_column == "price"
    assert mapping.volume_column == "size"
    assert mapping.expiration_date_column == "expiration_date"
    assert mapping.strike_column == "strike"


def test_option_trade_from_row_normalizes_option_side() -> None:
    trade = _option_trade_from_row(
        (
            "aapl",
            "2026-05-26T14:00:00+00:00",
            21,
            1.03,
            "c",
            2500,
            10,
            100,
            0.42,
        )
    )

    assert trade is not None
    assert trade.underlying == "AAPL"
    assert trade.side == "CALL"
    assert trade.expiration_days == 21
