from __future__ import annotations

import json
from pathlib import Path

from trader_optimizer.simulator import StrategyParams


def build_constant_step_offset_config(
    symbol: str,
    params: StrategyParams,
    ledger_name: str | None = None,
) -> dict[str, object]:
    safe_symbol = symbol.replace(".", "_")
    ledger_stem = ledger_name or f"CSO_{safe_symbol}_OPTIMIZED"
    contract = {
        "symbol": symbol,
        "secType": "STOCK",
        "currency": "USD",
        "exchange": "BACKTESTER",
    }
    return {
        "strategy_type": "ConstantStepOffset",
        "baseline": round(params.baseline, 6),
        "stepDelta": round(params.step_delta, 6),
        "executionLimitOffset": round(params.execution_limit_offset, 6),
        "stateTransitionThreshold": round(params.state_transition_threshold, 6),
        "orderQuantityInUSD": round(params.order_quantity_usd, 6),
        "orderQuantity": round(params.order_quantity, 6),
        "contract": contract,
        "price_contract": contract.copy(),
        "ledgerPath": f"data/TraderLedger/{ledger_stem}",
        "ledgerContextCollection": f"{ledger_stem}_context",
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
