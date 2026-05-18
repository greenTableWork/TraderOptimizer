import json

from trader_optimizer.strategy_configs import load_strategy_candidate


def test_load_strategy_candidate_detects_cso_without_strategy_type(tmp_path) -> None:
    path = tmp_path / "constant_step_offset_test.json"
    path.write_text(
        json.dumps(
            {
                "baseline": 100.0,
                "stepDelta": 1.0,
                "contract": {
                    "symbol": "AAPL",
                    "secType": "STOCK",
                    "currency": "USD",
                    "exchange": "BACKTESTER",
                },
            }
        )
    )

    candidate = load_strategy_candidate(path)

    assert candidate is not None
    assert candidate.strategy_type == "ConstantStepOffset"
    assert candidate.symbols == ("AAPL",)


def test_load_strategy_candidate_detects_portfolio_symbols(tmp_path) -> None:
    path = tmp_path / "portfolio.json"
    path.write_text(
        json.dumps(
            {
                "strategy_type": "PortfolioAllocation",
                "allocation_type": "QS-002",
                "contracts": [
                    {
                        "symbol": "AAPL",
                        "secType": "STOCK",
                        "currency": "USD",
                        "exchange": "BACKTESTER",
                    },
                    {
                        "symbol": "MSFT",
                        "secType": "STOCK",
                        "currency": "USD",
                        "exchange": "BACKTESTER",
                    },
                ],
            }
        )
    )

    candidate = load_strategy_candidate(path)

    assert candidate is not None
    assert candidate.strategy_type == "PortfolioAllocation"
    assert candidate.variant == "QS-002"
    assert candidate.symbols == ("AAPL", "MSFT")
