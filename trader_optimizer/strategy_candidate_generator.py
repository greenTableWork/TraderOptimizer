from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trader_optimizer.config import write_json
from trader_optimizer.regime_tuning_universe import load_regime_vectors_jsonl
from trader_optimizer.strategy_configs import (
    DEFAULT_CONFIG_GLOBS,
    StrategyCandidate,
    discover_strategy_candidates,
    load_strategy_candidate,
)


STRATEGY_CANDIDATE_PACK_SCHEMA = "strategy_candidate_pack.v1"
DEFAULT_EXCLUDED_INSTRUMENT_SYMBOLS = frozenset({"BTC", "LTC", "ES", "SPX", "VIX"})


@dataclass(frozen=True)
class CandidateFamily:
    family: str
    strategy_type: str
    variant: str
    base_config: dict[str, object]


DEFAULT_CANDIDATE_FAMILIES: tuple[CandidateFamily, ...] = (
    CandidateFamily(
        family="mac",
        strategy_type="MovingAverageCross",
        variant="SMA_CROSS",
        base_config={
            "strategy_type": "MovingAverageCross",
            "fastWindow": 5,
            "slowWindow": 21,
            "orderQuantityInUSD": 100.0,
            "orderQuantity": 1,
        },
    ),
    CandidateFamily(
        family="matrend002",
        strategy_type="MovingAverageCross",
        variant="MATREND-002",
        base_config={
            "strategy_type": "MovingAverageCross",
            "trendMode": "MATREND-002",
            "fastWindow": 3,
            "middleWindow": 8,
            "slowWindow": 34,
            "orderQuantityInUSD": 100.0,
            "orderQuantity": 1,
        },
    ),
    CandidateFamily(
        family="ts002",
        strategy_type="TechnicalSignal",
        variant="TS-002",
        base_config={
            "strategy_type": "TechnicalSignal",
            "signal_type": "TS-002",
            "fastWindow": 13,
            "slowWindow": 34,
            "orderQuantity": 1,
        },
    ),
    CandidateFamily(
        family="ts003",
        strategy_type="TechnicalSignal",
        variant="TS-003",
        base_config={
            "strategy_type": "TechnicalSignal",
            "signal_type": "TS-003",
            "middleWindow": 20,
            "trendWindow": 50,
            "bandStddev": 2,
            "orderQuantity": 1,
        },
    ),
    CandidateFamily(
        family="ts004",
        strategy_type="TechnicalSignal",
        variant="TS-004",
        base_config={
            "strategy_type": "TechnicalSignal",
            "signal_type": "TS-004",
            "openingRangeBars": 15,
            "useAtrStop": False,
            "orderQuantity": 1,
        },
    ),
    CandidateFamily(
        family="ts005",
        strategy_type="TechnicalSignal",
        variant="TS-005",
        base_config={
            "strategy_type": "TechnicalSignal",
            "signal_type": "TS-005",
            "rsiPeriod": 14,
            "divergenceLookback": 8,
            "orderQuantity": 1,
        },
    ),
)


def generate_strategy_candidate_pack(
    *,
    regime_vectors_path: Path,
    output_dir: Path,
    count: int,
    trader_root: Path | None = None,
    existing_config_globs: list[str] | None = None,
    include_non_equity: bool = False,
    families: tuple[CandidateFamily, ...] = DEFAULT_CANDIDATE_FAMILIES,
    summary_output: Path | None = None,
) -> dict[str, object]:
    if count <= 0:
        raise ValueError("count must be positive")
    vectors = load_regime_vectors_jsonl(regime_vectors_path)
    all_symbols = _unique_symbols(vectors)
    excluded_symbols = (
        []
        if include_non_equity
        else [symbol for symbol in all_symbols if symbol in DEFAULT_EXCLUDED_INSTRUMENT_SYMBOLS]
    )
    symbols = [
        symbol
        for symbol in all_symbols
        if include_non_equity or symbol not in DEFAULT_EXCLUDED_INSTRUMENT_SYMBOLS
    ]
    if not symbols:
        raise ValueError("No eligible symbols found in regime vectors")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_candidates = _discover_existing_candidates(
        trader_root,
        existing_config_globs,
    )
    existing_keys = {_candidate_search_key(candidate) for candidate in existing_candidates}

    pack_records, pack_keys = _load_existing_pack(output_dir)
    records = list(pack_records)
    used_keys = set(existing_keys)
    used_keys.update(pack_keys)

    duplicate_existing = 0
    duplicate_pack = 0
    if len(records) >= count:
        return _write_summary(
            summary_output or output_dir.with_name(f"{output_dir.name}_summary.json"),
            regime_vectors_path=regime_vectors_path,
            output_dir=output_dir,
            requested_count=count,
            all_symbols=all_symbols,
            eligible_symbols=symbols,
            excluded_symbols=excluded_symbols,
            records=records[:count],
            duplicate_existing=duplicate_existing,
            duplicate_pack=duplicate_pack,
            existing_config_globs=existing_config_globs,
        )

    for symbol in symbols:
        for family in families:
            config = _candidate_config(symbol, family)
            key = strategy_search_key(config)
            if key in existing_keys:
                duplicate_existing += 1
                continue
            if key in used_keys:
                duplicate_pack += 1
                continue
            path = output_dir / _candidate_filename(symbol, family.family)
            write_json(path, config)
            used_keys.add(key)
            records.append(_candidate_record(path, symbol, family, key))
            if len(records) >= count:
                return _write_summary(
                    summary_output or output_dir.with_name(f"{output_dir.name}_summary.json"),
                    regime_vectors_path=regime_vectors_path,
                    output_dir=output_dir,
                    requested_count=count,
                    all_symbols=all_symbols,
                    eligible_symbols=symbols,
                    excluded_symbols=excluded_symbols,
                    records=records,
                    duplicate_existing=duplicate_existing,
                    duplicate_pack=duplicate_pack,
                    existing_config_globs=existing_config_globs,
                )

    raise ValueError(
        f"Could only build {len(records)} unique candidates; requested {count}. "
        "Add more symbols, families, or allow non-equity instruments."
    )


def strategy_search_key(config: dict[str, object]) -> tuple[object, ...]:
    strategy_type = str(config.get("strategy_type") or "")
    symbols = tuple(_symbols_for_config(config))
    if strategy_type == "MovingAverageCross":
        variant = str(config.get("trendMode") or "SMA_CROSS").upper()
    elif strategy_type == "TechnicalSignal":
        variant = str(config.get("signal_type") or "technical").upper()
    elif strategy_type == "PortfolioAllocation":
        variant = str(config.get("allocation_type") or "portfolio").upper()
    elif strategy_type in {"ConstantStepOffset", "CSO"}:
        strategy_type = "ConstantStepOffset"
        variant = "CSO"
    else:
        variant = strategy_type
    return (strategy_type, variant, symbols)


def _discover_existing_candidates(
    trader_root: Path | None,
    config_globs: list[str] | None,
) -> list[StrategyCandidate]:
    if trader_root is None:
        return []
    return discover_strategy_candidates(
        trader_root,
        config_globs or list(DEFAULT_CONFIG_GLOBS),
    )


def _load_existing_pack(output_dir: Path) -> tuple[list[dict[str, object]], set[tuple[object, ...]]]:
    records: list[dict[str, object]] = []
    keys: set[tuple[object, ...]] = set()
    for path in sorted(output_dir.glob("*.json")):
        if path.name.endswith("_summary.json"):
            continue
        candidate = load_strategy_candidate(path)
        if candidate is None:
            continue
        key = _candidate_search_key(candidate)
        keys.add(key)
        records.append(
            {
                "path": str(path.resolve()),
                "symbol": candidate.symbols[0],
                "family": _family_from_candidate(candidate),
                "strategyType": candidate.strategy_type,
                "variant": key[1],
                "searchKey": list(key),
                "reusedExistingPackFile": True,
            }
        )
    return records, keys


def _candidate_search_key(candidate: StrategyCandidate) -> tuple[object, ...]:
    return strategy_search_key(candidate.config)


def _candidate_config(symbol: str, family: CandidateFamily) -> dict[str, object]:
    config = dict(family.base_config)
    contract = {
        "symbol": symbol,
        "secType": "STOCK",
        "currency": "USD",
        "exchange": "BACKTESTER",
    }
    safe_symbol = _safe_symbol(symbol).upper()
    ledger_stem = f"REGIME_CANDIDATE_{family.family.upper()}_{safe_symbol}"
    config.update(
        {
            "contract": contract,
            "price_contract": contract.copy(),
            "ledgerPath": f"data/TraderLedger/{ledger_stem}",
            "ledgerContextCollection": f"{ledger_stem}_context",
        }
    )
    return config


def _candidate_filename(symbol: str, family: str) -> str:
    return f"{_safe_symbol(symbol).lower()}_{family}_regime_candidate.json"


def _candidate_record(
    path: Path,
    symbol: str,
    family: CandidateFamily,
    key: tuple[object, ...],
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "symbol": symbol,
        "family": family.family,
        "strategyType": family.strategy_type,
        "variant": family.variant,
        "searchKey": list(key),
        "reusedExistingPackFile": False,
    }


def _write_summary(
    summary_path: Path,
    *,
    regime_vectors_path: Path,
    output_dir: Path,
    requested_count: int,
    all_symbols: list[str],
    eligible_symbols: list[str],
    excluded_symbols: list[str],
    records: list[dict[str, object]],
    duplicate_existing: int,
    duplicate_pack: int,
    existing_config_globs: list[str] | None,
) -> dict[str, object]:
    family_counts = Counter(str(record["family"]) for record in records)
    strategy_type_counts = Counter(str(record["strategyType"]) for record in records)
    summary: dict[str, object] = {
        "schema": STRATEGY_CANDIDATE_PACK_SCHEMA,
        "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "regimeVectors": str(regime_vectors_path.resolve()),
        "outputDir": str(output_dir.resolve()),
        "requestedCount": requested_count,
        "candidateCount": len(records),
        "symbols": eligible_symbols,
        "sourceSymbols": all_symbols,
        "excludedSymbols": excluded_symbols,
        "familyCounts": dict(sorted(family_counts.items())),
        "strategyTypeCounts": dict(sorted(strategy_type_counts.items())),
        "duplicateExistingConfigsSkipped": duplicate_existing,
        "duplicatePackCandidatesSkipped": duplicate_pack,
        "existingConfigGlobs": existing_config_globs or list(DEFAULT_CONFIG_GLOBS),
        "candidates": records,
    }
    write_json(summary_path, summary)
    return summary


def _unique_symbols(vectors: list[dict[str, object]]) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for vector in vectors:
        symbol = str(vector.get("symbol") or "").upper()
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    return symbols


def _symbols_for_config(config: dict[str, object]) -> list[str]:
    contracts = config.get("contracts")
    if isinstance(contracts, list):
        return [
            str(contract["symbol"]).upper()
            for contract in contracts
            if isinstance(contract, dict) and contract.get("symbol")
        ]
    contract = config.get("price_contract") or config.get("contract")
    if isinstance(contract, dict) and contract.get("symbol"):
        return [str(contract["symbol"]).upper()]
    return []


def _family_from_candidate(candidate: StrategyCandidate) -> str:
    config = candidate.config
    if candidate.strategy_type == "MovingAverageCross":
        trend_mode = str(config.get("trendMode") or "").lower()
        return trend_mode.replace("-", "") or "mac"
    if candidate.strategy_type == "TechnicalSignal":
        return str(config.get("signal_type") or "technical").lower().replace("-", "")
    if candidate.strategy_type == "PortfolioAllocation":
        return str(config.get("allocation_type") or "portfolio").lower().replace("-", "")
    return candidate.strategy_type.lower()


def _safe_symbol(symbol: str) -> str:
    return (
        symbol.replace(".", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )
