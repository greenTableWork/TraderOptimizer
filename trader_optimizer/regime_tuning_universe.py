from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from trader_optimizer.config import write_json
from trader_optimizer.strategy_configs import StrategyCandidate


REGIME_TUNING_UNIVERSE_SCHEMA = "regime_tuning_universe.v1"
REGIME_TUNING_TASK_SCHEMA = "regime_tuning_task.v1"
DEFAULT_REGIME_DIMENSIONS: tuple[str, ...] = (
    "directionSign",
    "instrumentVolatilityRegime",
    "marketVolatilityRegime",
    "volumeRegime",
)
StrategyScope = Literal["matching-symbol", "all"]


def load_regime_vectors_jsonl(path: Path) -> list[dict[str, object]]:
    vectors: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Regime vector line {line_number} must be a JSON object")
            vectors.append(payload)
    return vectors


def build_regime_tuning_tasks(
    vectors: Sequence[Mapping[str, object]],
    candidates: Sequence[StrategyCandidate],
    *,
    trader_root: Path,
    output_root: Path,
    export_root: Path | None,
    regime_dimensions: Sequence[str] = DEFAULT_REGIME_DIMENSIONS,
    strategy_scope: StrategyScope = "matching-symbol",
    trials: int = 25,
    max_bars: int = 0,
    workers: int = 1,
    skip_backtester_build: bool = True,
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for vector in vectors:
        symbol = str(vector.get("symbol") or "").upper()
        if not symbol:
            continue
        start_utc = str(vector.get("startUtc") or "")
        end_utc = str(vector.get("endUtc") or "")
        bar_size = _optional_str(vector.get("barSize"))
        regime_cell = regime_cell_for_vector(vector, regime_dimensions)
        for candidate in candidates:
            candidate_symbols = tuple(symbol.upper() for symbol in candidate.symbols)
            symbol_matches = symbol in candidate_symbols
            if strategy_scope == "matching-symbol" and not symbol_matches:
                continue
            requires_retargeting = not symbol_matches
            task_id = regime_tuning_task_id(
                symbol=symbol,
                start_utc=start_utc,
                end_utc=end_utc,
                regime_cell=regime_cell,
                candidate_name=candidate.name,
                source_config=str(candidate.path),
            )
            task_output_dir = output_root / task_id
            task: dict[str, object] = {
                "schema": REGIME_TUNING_TASK_SCHEMA,
                "taskId": task_id,
                "symbol": symbol,
                "startUtc": start_utc,
                "endUtc": end_utc,
                "durationSeconds": _duration_seconds(start_utc, end_utc),
                "durationBars": int(vector.get("bars") or 0),
                "barSize": bar_size,
                "whatToShow": vector.get("whatToShow"),
                "useRth": vector.get("useRth"),
                "regimeCellId": regime_cell_id(regime_cell),
                "regimeCell": regime_cell,
                "regimeDimensions": list(regime_dimensions),
                "strategyName": candidate.name,
                "strategyType": candidate.strategy_type,
                "variant": candidate.variant,
                "strategySymbols": list(candidate_symbols),
                "sourceConfig": str(candidate.path),
                "sourceConfigGlob": _config_glob_for_candidate(trader_root, candidate),
                "requiresRetargeting": requires_retargeting,
                "runnableWithExistingConfig": not requires_retargeting,
                "optimizerCommand": _optimizer_command(
                    trader_root=trader_root,
                    source_config_glob=_config_glob_for_candidate(trader_root, candidate),
                    output_dir=task_output_dir,
                    export_dir=export_root / task_id if export_root else None,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    bar_size=bar_size,
                    trials=trials,
                    max_bars=max_bars,
                    workers=workers,
                    skip_backtester_build=skip_backtester_build,
                ),
                "vectorCore": vector.get("core", {}),
                "vectorAdvanced": vector.get("advanced", {}),
            }
            tasks.append(task)
    return tasks


def write_regime_tuning_universe(
    output_path: Path,
    summary_path: Path,
    tasks: Sequence[Mapping[str, object]],
    *,
    vector_count: int,
    strategy_count: int,
    strategy_scope: StrategyScope,
    regime_dimensions: Sequence[str] = DEFAULT_REGIME_DIMENSIONS,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, sort_keys=True))
            handle.write("\n")
    write_json(
        summary_path,
        summarize_regime_tuning_universe(
            tasks,
            vector_count=vector_count,
            strategy_count=strategy_count,
            strategy_scope=strategy_scope,
            regime_dimensions=regime_dimensions,
        ),
    )


def summarize_regime_tuning_universe(
    tasks: Sequence[Mapping[str, object]],
    *,
    vector_count: int,
    strategy_count: int,
    strategy_scope: StrategyScope,
    regime_dimensions: Sequence[str] = DEFAULT_REGIME_DIMENSIONS,
) -> dict[str, object]:
    by_cell = Counter(str(task.get("regimeCellId") or "") for task in tasks)
    by_symbol = Counter(str(task.get("symbol") or "") for task in tasks)
    by_strategy_type = Counter(str(task.get("strategyType") or "") for task in tasks)
    by_runnable = Counter(
        "runnable" if task.get("runnableWithExistingConfig") else "requires_retargeting"
        for task in tasks
    )
    return {
        "schema": REGIME_TUNING_UNIVERSE_SCHEMA,
        "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "strategyScope": strategy_scope,
        "regimeDimensions": list(regime_dimensions),
        "vectors": vector_count,
        "strategies": strategy_count,
        "tasks": len(tasks),
        "uniqueRegimeCells": len(by_cell),
        "runnableTasks": by_runnable.get("runnable", 0),
        "requiresRetargetingTasks": by_runnable.get("requires_retargeting", 0),
        "tasksBySymbol": dict(sorted(by_symbol.items())),
        "tasksByStrategyType": dict(sorted(by_strategy_type.items())),
        "tasksByRegimeCell": dict(sorted(by_cell.items())),
    }


def regime_cell_for_vector(
    vector: Mapping[str, object],
    regime_dimensions: Sequence[str] = DEFAULT_REGIME_DIMENSIONS,
) -> dict[str, object]:
    core = vector.get("core")
    if not isinstance(core, Mapping):
        core = {}
    return {dimension: core.get(dimension, "unknown") for dimension in regime_dimensions}


def regime_cell_id(regime_cell: Mapping[str, object]) -> str:
    payload = json.dumps(regime_cell, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"regime-cell-v1:{digest}"


def regime_tuning_task_id(
    *,
    symbol: str,
    start_utc: str,
    end_utc: str,
    regime_cell: Mapping[str, object],
    candidate_name: str,
    source_config: str,
) -> str:
    payload = json.dumps(
        {
            "symbol": symbol,
            "startUtc": start_utc,
            "endUtc": end_utc,
            "regimeCell": regime_cell,
            "candidateName": candidate_name,
            "sourceConfig": source_config,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    safe_candidate = candidate_name.replace("/", "_").replace(" ", "_")[:48]
    return f"{symbol.lower()}_{safe_candidate}_{digest}"


def _optimizer_command(
    *,
    trader_root: Path,
    source_config_glob: str,
    output_dir: Path,
    export_dir: Path | None,
    start_utc: str,
    end_utc: str,
    bar_size: str | None,
    trials: int,
    max_bars: int,
    workers: int,
    skip_backtester_build: bool,
) -> list[str]:
    command = [
        ".venv/bin/trader-optimizer",
        "optimize-existing",
        "--trader-root",
        str(trader_root),
        "--config-glob",
        source_config_glob,
        "--start-utc",
        start_utc,
        "--end-utc",
        end_utc,
        "--trials",
        str(trials),
        "--max-bars",
        str(max_bars),
        "--workers",
        str(workers),
        "--output-dir",
        str(output_dir),
        "--plan-path",
        str(output_dir.with_name(f"{output_dir.name}_plan.md")),
    ]
    if bar_size:
        command.extend(["--bar-size", bar_size])
    if skip_backtester_build:
        command.append("--skip-backtester-build")
    if export_dir is not None:
        command.extend(["--export-config-dir", str(export_dir)])
    return command


def _config_glob_for_candidate(trader_root: Path, candidate: StrategyCandidate) -> str:
    try:
        return str(candidate.path.resolve().relative_to(trader_root.resolve()))
    except ValueError:
        return str(candidate.path)


def _duration_seconds(start_utc: str, end_utc: str) -> float | None:
    if not start_utc or not end_utc:
        return None
    return (_parse_utc(end_utc) - _parse_utc(start_utc)).total_seconds()


def _parse_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
