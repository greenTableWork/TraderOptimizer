from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    path: Path
    strategy_type: str
    config: dict[str, Any]
    symbols: tuple[str, ...]
    variant: str


DEFAULT_CONFIG_GLOBS = (
    "TraderCore/TraderLogicConfigs/TraderCore/configs/backtesting/**/*.json",
    "TraderCore/TraderLogicConfigs/TraderLab/configs/backtests/**/*.json",
)
OPTIMIZER_SUPPORTED_TECHNICAL_SIGNAL_VARIANTS = frozenset(
    {
        "TS-002",
        "EMACROSS",
        "EMA_CROSS",
        "TS-003",
        "BOLLINGERBREAKOUT",
        "BOLLINGER_BREAKOUT",
        "TS-004",
        "OPENINGRANGEBREAKOUT",
        "OPENING_RANGE_BREAKOUT",
        "ORB",
        "TS-005",
        "RSIDIVERGENCE",
        "RSI_DIVERGENCE",
    }
)
OPTIMIZER_SUPPORTED_PORTFOLIO_VARIANTS = frozenset(
    {
        "QS-001",
        "QS-002",
        "PAIRS-001",
    }
)


def discover_strategy_candidates(
    trader_root: Path,
    config_globs: list[str] | None = None,
) -> list[StrategyCandidate]:
    candidates: list[StrategyCandidate] = []
    seen_paths: set[Path] = set()
    for pattern in config_globs or list(DEFAULT_CONFIG_GLOBS):
        for path in sorted(trader_root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen_paths or not path.is_file():
                continue
            seen_paths.add(resolved)
            candidate = load_strategy_candidate(path)
            if candidate:
                candidates.append(candidate)
    return candidates


def is_optimizer_supported_candidate(candidate: StrategyCandidate) -> bool:
    if candidate.strategy_type in {"ConstantStepOffset", "MovingAverageCross"}:
        return True
    if candidate.strategy_type == "TechnicalSignal":
        return candidate.variant in OPTIMIZER_SUPPORTED_TECHNICAL_SIGNAL_VARIANTS
    if candidate.strategy_type == "PortfolioAllocation":
        return candidate.variant in OPTIMIZER_SUPPORTED_PORTFOLIO_VARIANTS
    return False


def load_strategy_candidate(path: Path) -> StrategyCandidate | None:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(config, dict):
        return None

    strategy_type = str(config.get("strategy_type") or "")
    if not strategy_type and {"baseline", "stepDelta"}.issubset(config):
        strategy_type = "ConstantStepOffset"
    if strategy_type not in {
        "ConstantStepOffset",
        "CSO",
        "MovingAverageCross",
        "TechnicalSignal",
        "PortfolioAllocation",
    }:
        return None

    symbols = _symbols_for_config(config)
    if not symbols:
        return None

    variant = ""
    if strategy_type == "TechnicalSignal":
        variant = str(config.get("signal_type", "technical")).upper()
    elif strategy_type == "PortfolioAllocation":
        variant = str(config.get("allocation_type", "portfolio")).upper()
    elif strategy_type in {"ConstantStepOffset", "CSO"}:
        variant = "CSO"
        strategy_type = "ConstantStepOffset"
    else:
        variant = strategy_type

    return StrategyCandidate(
        name=_safe_name(path),
        path=path,
        strategy_type=strategy_type,
        config=config,
        symbols=tuple(symbols),
        variant=variant,
    )


def _symbols_for_config(config: dict[str, Any]) -> list[str]:
    if "contracts" in config and isinstance(config["contracts"], list):
        symbols = []
        for contract in config["contracts"]:
            if isinstance(contract, dict) and contract.get("symbol"):
                symbols.append(str(contract["symbol"]))
        return symbols
    contract = config.get("price_contract") or config.get("contract")
    if isinstance(contract, dict) and contract.get("symbol"):
        return [str(contract["symbol"])]
    return []


def _safe_name(path: Path) -> str:
    return path.stem.replace(".", "_").replace("-", "_")
