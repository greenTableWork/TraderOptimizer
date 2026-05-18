from trader_optimizer.data import Bar
from trader_optimizer.simulator import StrategyParams, simulate_constant_step_offset


def _bar(index: int, price: float) -> Bar:
    return Bar(
        timestamp_utc=f"2026-01-01T00:{index:02d}:00+00:00",
        open=price,
        high=price,
        low=price,
        close=price,
    )


def test_constant_step_offset_buys_drop_and_sells_recovery() -> None:
    bars = [
        _bar(0, 100.0),
        _bar(1, 99.1),
        _bar(2, 98.9),
        _bar(3, 98.8),
        _bar(4, 100.0),
        _bar(5, 101.1),
        _bar(6, 101.2),
    ]
    params = StrategyParams(
        baseline=100.0,
        step_delta=1.0,
        execution_limit_offset=5.0,
        state_transition_threshold=0.2,
        order_quantity_usd=100.0,
        order_quantity=1.0,
    )

    result, fills = simulate_constant_step_offset(bars, params)

    assert result.buys == 1
    assert result.sells == 1
    assert result.final_position == 0.0
    assert result.net_pnl > 0.0
    assert [fill.action for fill in fills] == ["BUY", "SELL"]


def test_constant_step_offset_penalizes_open_inventory_in_metrics() -> None:
    bars = [_bar(0, 100.0), _bar(1, 99.1), _bar(2, 98.9), _bar(3, 98.8)]
    params = StrategyParams(
        baseline=100.0,
        step_delta=1.0,
        execution_limit_offset=5.0,
        state_transition_threshold=0.2,
        order_quantity_usd=100.0,
        order_quantity=1.0,
    )

    result, _ = simulate_constant_step_offset(bars, params)

    assert result.buys == 1
    assert result.sells == 0
    assert result.final_position > 0.0
    assert result.final_position_value > 0.0
