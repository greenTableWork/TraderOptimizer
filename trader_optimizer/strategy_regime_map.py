from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trader_optimizer.config import write_json
from trader_optimizer.regime_tuning_universe import load_regime_vectors_jsonl


STRATEGY_REGIME_MAP_SCHEMA = "strategy_regime_map.v1"
STRATEGY_REGIME_MAP_SUMMARY_SCHEMA = "strategy_regime_map_summary.v1"
DEFAULT_VALIDATION_STATUSES = ("ok",)


def build_strategy_regime_map_from_run_summary(
    run_summary_path: Path,
    *,
    validation_statuses: Sequence[str] = DEFAULT_VALIDATION_STATUSES,
    cwd: Path | None = None,
) -> list[dict[str, object]]:
    run_summary_path = run_summary_path.resolve()
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    universe_path = resolve_run_artifact_path(
        str(run_summary.get("universe") or ""),
        run_summary_path=run_summary_path,
        cwd=cwd or Path.cwd(),
    )
    task_by_id = {
        str(task.get("taskId") or ""): task
        for task in load_regime_vectors_jsonl(universe_path)
    }
    allowed_statuses = {status.lower() for status in validation_statuses}
    entries: list[dict[str, object]] = []
    for outer_result in run_summary.get("results", []):
        if not isinstance(outer_result, Mapping):
            continue
        task_id = str(outer_result.get("task_id") or "")
        task = task_by_id.get(task_id, {})
        batch_summary_path = resolve_run_artifact_path(
            str(outer_result.get("summary_path") or ""),
            run_summary_path=run_summary_path,
            cwd=cwd or Path.cwd(),
            must_exist=False,
        )
        if not batch_summary_path.exists():
            continue
        batch_summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
        command = _string_list(outer_result.get("command"))
        export_dir = _command_option_path(command, "--export-config-dir")
        if export_dir is not None and not export_dir.is_absolute():
            export_dir = (cwd or Path.cwd()) / export_dir
        for row in batch_summary.get("results", []):
            if not isinstance(row, Mapping):
                continue
            status = str(row.get("status") or "")
            if status.lower() not in allowed_statuses:
                continue
            entry = strategy_regime_map_entry(
                outer_result=outer_result,
                task=task,
                row=row,
                batch_summary_path=batch_summary_path,
                export_dir=export_dir,
                run_summary_path=run_summary_path,
            )
            if entry is not None:
                entries.append(entry)
    return dedupe_strategy_regime_map_entries(entries)


def strategy_regime_map_entry(
    *,
    outer_result: Mapping[str, object],
    task: Mapping[str, object],
    row: Mapping[str, object],
    batch_summary_path: Path,
    export_dir: Path | None,
    run_summary_path: Path,
) -> dict[str, object] | None:
    regime_cell = task.get("regimeCell")
    if not isinstance(regime_cell, Mapping):
        return None
    symbol = str(task.get("symbol") or outer_result.get("symbol") or "").upper()
    if not symbol:
        return None
    strategy_name = str(row.get("name") or task.get("strategyName") or "")
    summary_path = _optional_path(row.get("summary"))
    best_summary = _load_json(summary_path)
    backtester = best_summary.get("backtester") if isinstance(best_summary, Mapping) else {}
    if not isinstance(backtester, Mapping):
        backtester = {}
    benchmarks = backtester.get("benchmarks")
    if not isinstance(benchmarks, Mapping):
        benchmarks = {}
    exported_config = exported_config_path(export_dir, row)
    run_config_path = _optional_path(row.get("best_config"))
    config_path = exported_config or run_config_path
    if config_path is None:
        return None
    same_stock = _benchmark_block(benchmarks, "same_stock_buy_and_hold")
    spx = _benchmark_block(benchmarks, "spx_buy_and_hold")
    positive = _benchmark_block(benchmarks, "positive_return")
    tuning_profile = row.get("tuning_profile")
    if not isinstance(tuning_profile, Mapping):
        tuning_profile = {}
    entry: dict[str, object] = {
        "schema": STRATEGY_REGIME_MAP_SCHEMA,
        "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "symbol": symbol,
        "strategyName": strategy_name,
        "strategyType": str(row.get("strategy_type") or task.get("strategyType") or ""),
        "variant": str(row.get("variant") or task.get("variant") or ""),
        "symbols": list(row.get("symbols") or task.get("strategySymbols") or [symbol]),
        "configPath": str(config_path),
        "runConfigPath": str(run_config_path) if run_config_path is not None else "",
        "summaryPath": str(summary_path) if summary_path is not None else "",
        "sourceConfig": str(row.get("source_config") or task.get("sourceConfig") or ""),
        "taskId": str(task.get("taskId") or outer_result.get("task_id") or ""),
        "regimeCellId": str(task.get("regimeCellId") or ""),
        "regimeCell": dict(regime_cell),
        "regimeDimensions": list(task.get("regimeDimensions") or []),
        "validationStatus": str(row.get("status") or ""),
        "strategyReturnPct": _float_or_none(row.get("strategy_return_pct")),
        "benchmarkReturnPct": _float_or_none(row.get("benchmark_return_pct")),
        "excessReturnPct": _float_or_none(row.get("excess_return_pct")),
        "sameStockExcessReturnPct": _float_or_none(same_stock.get("excess_return_pct")),
        "spxExcessReturnPct": _float_or_none(spx.get("excess_return_pct")),
        "positiveReturnPassed": positive.get("passed"),
        "sameStockPassed": same_stock.get("passed"),
        "spxPassed": spx.get("passed"),
        "bestValue": _float_or_none(row.get("best_value")),
        "mapScore": map_score(
            row_excess_return=_float_or_none(row.get("excess_return_pct")),
            same_stock_excess_return=_float_or_none(same_stock.get("excess_return_pct")),
            spx_excess_return=_float_or_none(spx.get("excess_return_pct")),
        ),
        "tuningCategories": list(tuning_profile.get("categoryLabels") or []),
        "optimizedFor": tuning_profile.get("optimizedFor") or {},
        "window": {
            "startUtc": task.get("startUtc"),
            "endUtc": task.get("endUtc"),
            "durationSeconds": task.get("durationSeconds"),
            "durationBars": task.get("durationBars"),
            "barSize": task.get("barSize"),
            "whatToShow": task.get("whatToShow"),
            "useRth": task.get("useRth"),
        },
        "source": str(run_summary_path),
        "batchSummaryPath": str(batch_summary_path),
    }
    return entry


def dedupe_strategy_regime_map_entries(
    entries: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    best_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for entry in entries:
        key = (
            str(entry.get("symbol") or "").upper(),
            str(entry.get("regimeCellId") or ""),
            str(entry.get("strategyName") or ""),
            str(entry.get("configPath") or ""),
        )
        candidate = dict(entry)
        current = best_by_key.get(key)
        if current is None or _map_sort_key(candidate) > _map_sort_key(current):
            best_by_key[key] = candidate
    return sorted(
        best_by_key.values(),
        key=lambda entry: (
            str(entry.get("symbol") or ""),
            str(entry.get("regimeCellId") or ""),
            -float(entry.get("mapScore") or 0.0),
            str(entry.get("strategyName") or ""),
        ),
    )


def summarize_strategy_regime_map(
    entries: Sequence[Mapping[str, object]],
    *,
    run_summary_path: Path,
) -> dict[str, object]:
    by_status = Counter(str(entry.get("validationStatus") or "") for entry in entries)
    by_symbol = Counter(str(entry.get("symbol") or "") for entry in entries)
    by_type = Counter(str(entry.get("strategyType") or "") for entry in entries)
    by_cell = Counter(str(entry.get("regimeCellId") or "") for entry in entries)
    best_by_symbol_cell: dict[str, dict[str, object]] = {}
    for entry in entries:
        key = f"{entry.get('symbol')}|{entry.get('regimeCellId')}"
        current = best_by_symbol_cell.get(key)
        if current is None or _map_sort_key(entry) > _map_sort_key(current):
            best_by_symbol_cell[key] = {
                "symbol": entry.get("symbol"),
                "regimeCellId": entry.get("regimeCellId"),
                "strategyName": entry.get("strategyName"),
                "configPath": entry.get("configPath"),
                "mapScore": entry.get("mapScore"),
                "strategyReturnPct": entry.get("strategyReturnPct"),
                "sameStockExcessReturnPct": entry.get("sameStockExcessReturnPct"),
                "spxExcessReturnPct": entry.get("spxExcessReturnPct"),
            }
    return {
        "schema": STRATEGY_REGIME_MAP_SUMMARY_SCHEMA,
        "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "sourceRunSummary": str(run_summary_path),
        "entries": len(entries),
        "statusCounts": dict(sorted(by_status.items())),
        "symbols": dict(sorted(by_symbol.items())),
        "strategyTypes": dict(sorted(by_type.items())),
        "regimeCells": dict(sorted(by_cell.items())),
        "bestBySymbolRegimeCell": dict(sorted(best_by_symbol_cell.items())),
    }


def write_strategy_regime_map_jsonl(
    path: Path,
    entries: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")


def write_strategy_regime_map_summary(
    path: Path,
    entries: Sequence[Mapping[str, object]],
    *,
    run_summary_path: Path,
) -> None:
    write_json(path, summarize_strategy_regime_map(entries, run_summary_path=run_summary_path))


def exported_config_path(export_dir: Path | None, row: Mapping[str, object]) -> Path | None:
    if export_dir is None:
        return None
    index_path = export_dir / "index.json"
    if not index_path.exists():
        return None
    payload = _load_json(index_path)
    configs = payload.get("configs") if isinstance(payload, Mapping) else None
    if not isinstance(configs, list):
        return None
    row_name = str(row.get("name") or "")
    for config in configs:
        if not isinstance(config, Mapping):
            continue
        if row_name and str(config.get("name") or "") != row_name:
            continue
        exported = config.get("exported_config")
        if exported:
            return Path(str(exported))
    for config in configs:
        if isinstance(config, Mapping) and config.get("exported_config"):
            return Path(str(config["exported_config"]))
    return None


def resolve_run_artifact_path(
    value: str,
    *,
    run_summary_path: Path,
    cwd: Path,
    must_exist: bool = True,
) -> Path:
    if not value:
        raise ValueError("Run summary did not include a required artifact path")
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                cwd / raw,
                run_summary_path.parent / raw,
                run_summary_path.parent.parent / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f"Could not resolve run artifact path {value!r}")
    return candidates[1].resolve() if len(candidates) > 1 else raw.resolve()


def map_score(
    *,
    row_excess_return: float | None,
    same_stock_excess_return: float | None,
    spx_excess_return: float | None,
) -> float:
    values = [
        value
        for value in (same_stock_excess_return, spx_excess_return, row_excess_return)
        if value is not None
    ]
    return sum(values) / len(values) if values else 0.0


def _map_sort_key(entry: Mapping[str, object]) -> tuple[float, float, float, str]:
    return (
        float(entry.get("mapScore") or 0.0),
        float(entry.get("spxExcessReturnPct") or 0.0),
        float(entry.get("sameStockExcessReturnPct") or 0.0),
        str(entry.get("configPath") or ""),
    )


def _benchmark_block(benchmarks: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = benchmarks.get(key)
    return value if isinstance(value, Mapping) else {}


def _command_option_path(command: Sequence[str], option: str) -> Path | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return Path(command[index + 1])


def _optional_path(value: object) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
