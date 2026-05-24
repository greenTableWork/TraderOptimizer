import pytest

from trader_optimizer.data import Bar
from trader_optimizer.simple_strategies import (
    simulate_buy_and_hold,
    simulate_equal_weight_buy_and_hold,
)


def _bar(day: int, close: float) -> Bar:
    return Bar(
        timestamp_utc=f"2026-01-{day:02d}T00:00:00+00:00",
        open=close,
        high=close,
        low=close,
        close=close,
    )


def test_buy_and_hold_uses_same_quantity_and_commission() -> None:
    result = simulate_buy_and_hold([_bar(1, 100.0), _bar(2, 110.0)], quantity=10)

    assert result.fills == 1
    assert result.allocated_capital == 1000.0
    assert result.commissions == 1.0
    assert result.net_pnl == 99.0
    assert result.return_pct == pytest.approx(0.099)


def test_equal_weight_buy_and_hold_allocates_across_symbols() -> None:
    result = simulate_equal_weight_buy_and_hold(
        {
            "AAPL": [_bar(1, 100.0), _bar(2, 110.0)],
            "MSFT": [_bar(1, 200.0), _bar(2, 220.0)],
        },
        portfolio_notional=1000.0,
    )

    assert result.fills == 2
    assert result.allocated_capital == 1000.0
    assert result.commissions == 2.0
    assert result.net_pnl == 98.0
    assert result.return_pct == pytest.approx(0.098)
