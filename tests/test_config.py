from trader_optimizer.config import build_constant_step_offset_config
from trader_optimizer.simulator import StrategyParams


def test_build_constant_step_offset_config_has_tradercore_fields() -> None:
    config = build_constant_step_offset_config(
        "AAPL",
        StrategyParams(
            baseline=100.0,
            step_delta=1.0,
            execution_limit_offset=10.0,
            state_transition_threshold=0.25,
            order_quantity_usd=500.0,
            order_quantity=5.0,
        ),
    )

    assert config["strategy_type"] == "ConstantStepOffset"
    assert config["baseline"] == 100.0
    assert config["stepDelta"] == 1.0
    assert config["executionLimitOffset"] == 10.0
    assert config["stateTransitionThreshold"] == 0.25
    assert config["orderQuantityInUSD"] == 500.0
    assert config["orderQuantity"] == 5.0
    assert config["contract"] == config["price_contract"]
    assert config["contract"]["exchange"] == "BACKTESTER"
