from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

from trader_optimizer.batch import BatchItemResult, BatchSettings, optimize_candidates
from trader_optimizer.config import write_json
from trader_optimizer.strategy_configs import (
    StrategyCandidate,
    discover_strategy_candidates,
    load_strategy_candidate,
)


@dataclass(frozen=True)
class BenchmarkPeriod:
    name: str
    months: int = 0
    days: int = 0


@dataclass(frozen=True)
class BenchmarkLoopSettings:
    trader_root: Path
    output_root: Path
    state_path: Path
    base_settings: BatchSettings
    config_globs: list[str] | None = None
    include_strategy_types: set[str] | None = None
    exclude_strategy_types: set[str] | None = None
    now: datetime | None = None
    periods: tuple[BenchmarkPeriod, ...] = ()
    sleep_seconds: float = 24 * 60 * 60
    once: bool = False


DEFAULT_BENCHMARK_PERIODS: tuple[BenchmarkPeriod, ...] = (
    BenchmarkPeriod("daily", days=1),
    BenchmarkPeriod("weekly", days=7),
    BenchmarkPeriod("biweekly", days=14),
    BenchmarkPeriod("monthly", months=1),
    BenchmarkPeriod("quarterly", months=3),
    BenchmarkPeriod("semiannual", months=6),
    BenchmarkPeriod("yearly", months=12),
)


def run_benchmark_loop(
    settings: BenchmarkLoopSettings,
    *,
    optimize: Callable[[list[StrategyCandidate], BatchSettings], list[BatchItemResult]] = optimize_candidates,
) -> dict[str, object]:
    """Run the rolling benchmark loop.

    With ``once=True`` this executes one cycle and returns its manifest. Otherwise
    it repeats forever, sleeping between cycles; each cycle benchmarks daily,
    weekly, biweekly, monthly, quarterly, semiannual, and yearly windows.
    """

    last_manifest: dict[str, object] = {}
    while True:
        last_manifest = run_benchmark_cycle(settings, optimize=optimize)
        if settings.once:
            return last_manifest
        time.sleep(settings.sleep_seconds)


def run_benchmark_cycle(
    settings: BenchmarkLoopSettings,
    *,
    optimize: Callable[[list[StrategyCandidate], BatchSettings], list[BatchItemResult]] = optimize_candidates,
) -> dict[str, object]:
    now = _as_utc(settings.now or datetime.now(UTC))
    cycle_id = now.strftime("%Y%m%dT%H%M%SZ")
    state = _read_state(settings.state_path)
    periods = settings.periods or DEFAULT_BENCHMARK_PERIODS
    base_candidates = _filtered_candidates(
        discover_strategy_candidates(settings.trader_root, settings.config_globs),
        include_types=settings.include_strategy_types,
        exclude_types=settings.exclude_strategy_types,
    )
    period_manifests: list[dict[str, object]] = []

    for period in periods:
        start = _period_start(now, period)
        output_dir = settings.output_root / cycle_id / period.name
        candidates = _candidate_set(base_candidates, _champion_candidates(settings.state_path.parent))
        period_settings = _period_settings(settings.base_settings, output_dir, start, now)
        if period_settings.verbose:
            print(
                f"benchmark period={period.name} start={_format_utc(start)} "
                f"end={_format_utc(now)} candidates={len(candidates)}"
            )
        results = optimize(candidates, period_settings)
        champion = _select_champion(results)
        champion_record = None
        if champion is not None:
            champion_record = _promote_champion(
                champion,
                period,
                settings.state_path.parent,
                state,
                cycle_id,
            )
        period_manifests.append(
            {
                "period": period.name,
                "start_utc": _format_utc(start),
                "end_utc": _format_utc(now),
                "output_dir": str(output_dir),
                "candidates": len(candidates),
                "ok_results": sum(1 for result in results if result.status == "ok"),
                "champion": champion_record,
            }
        )

    manifest = {
        "cycle_id": cycle_id,
        "created_at_utc": _format_utc(now),
        "periods": period_manifests,
    }
    state["last_cycle"] = manifest
    _write_state(settings.state_path, state)
    write_json(settings.output_root / cycle_id / "benchmark_manifest.json", manifest)
    return manifest


def _period_settings(
    base_settings: BatchSettings,
    output_dir: Path,
    start: datetime,
    end: datetime,
) -> BatchSettings:
    return BatchSettings(
        pg_settings=base_settings.pg_settings,
        optuna_storage_url=base_settings.optuna_storage_url,
        output_dir=output_dir,
        trials=base_settings.trials,
        max_bars=base_settings.max_bars,
        preferred_bar_size=base_settings.preferred_bar_size,
        train_fraction=base_settings.train_fraction,
        verbose=base_settings.verbose,
        export_config_dir=output_dir / "exported_configs",
        workers=base_settings.workers,
        backtester_settings=base_settings.backtester_settings,
        start_utc=_format_utc(start),
        end_utc=_format_utc(end),
        strategy_budget=base_settings.strategy_budget,
    )


def _candidate_set(
    base_candidates: Iterable[StrategyCandidate],
    champion_candidates: Iterable[StrategyCandidate],
) -> list[StrategyCandidate]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    output: list[StrategyCandidate] = []
    for candidate in [*base_candidates, *champion_candidates]:
        key = (candidate.name, candidate.strategy_type, candidate.symbols)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def _champion_candidates(state_dir: Path) -> list[StrategyCandidate]:
    champion_dir = state_dir / "champions"
    candidates: list[StrategyCandidate] = []
    if not champion_dir.exists():
        return candidates
    for path in sorted(champion_dir.glob("**/*.json")):
        candidate = load_strategy_candidate(path)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _filtered_candidates(
    candidates: Iterable[StrategyCandidate],
    *,
    include_types: set[str] | None,
    exclude_types: set[str] | None,
) -> list[StrategyCandidate]:
    include = {item.lower() for item in include_types or set()}
    exclude = {item.lower() for item in exclude_types or set()}
    output = []
    for candidate in candidates:
        strategy_type = candidate.strategy_type.lower()
        if include and strategy_type not in include:
            continue
        if strategy_type in exclude:
            continue
        output.append(candidate)
    return output


def _select_champion(results: Iterable[BatchItemResult]) -> BatchItemResult | None:
    ok = [result for result in results if result.status == "ok" and result.best_config]
    if not ok:
        return None
    return max(ok, key=_score_result)


def _score_result(result: BatchItemResult) -> float:
    for value in (
        result.excess_return_pct,
        result.strategy_return_pct,
        result.best_value,
    ):
        if value is not None:
            return float(value)
    return float("-inf")


def _promote_champion(
    result: BatchItemResult,
    period: BenchmarkPeriod,
    state_dir: Path,
    state: dict[str, object],
    cycle_id: str,
) -> dict[str, object] | None:
    score = _score_result(result)
    champions = state.setdefault("champions", {})
    assert isinstance(champions, dict)
    previous = champions.get(period.name)
    if isinstance(previous, dict) and score <= float(previous.get("score", float("-inf"))):
        return previous

    best_config = Path(str(result.best_config))
    destination = state_dir / "champions" / period.name / f"{result.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_config, destination)
    record = {
        "cycle_id": cycle_id,
        "name": result.name,
        "strategy_type": result.strategy_type,
        "variant": result.variant,
        "symbols": list(result.symbols),
        "score": score,
        "strategy_return_pct": result.strategy_return_pct,
        "benchmark_return_pct": result.benchmark_return_pct,
        "excess_return_pct": result.excess_return_pct,
        "tuning_profile": result.tuning_profile,
        "source_best_config": str(best_config),
        "champion_config": str(destination),
    }
    champions[period.name] = record
    return record


def _read_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"champions": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data.setdefault("champions", {})
        return data
    return {"champions": {}}


def _write_state(path: Path, state: dict[str, object]) -> None:
    write_json(path, state)


def _period_start(now: datetime, period: BenchmarkPeriod) -> datetime:
    start = now - timedelta(days=period.days)
    if period.months:
        start = _subtract_months(start, period.months)
    return start


def _subtract_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + (value.month - 1) - months
    year = month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _days_in_month(year, month))
    return value.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=UTC)
    this_month = datetime(year, month, 1, tzinfo=UTC)
    return (next_month - this_month).days


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")
