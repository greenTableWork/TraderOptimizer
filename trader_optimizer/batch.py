from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import optuna

from trader_optimizer.backtester import (
    BackTesterSettings,
    BacktestProfile,
    BacktestValidationResult,
    prepare_backtester,
    profile_for_bars,
    validate_with_backtester,
)
from trader_optimizer.config import write_json
from trader_optimizer.data import (
    Bar,
    choose_data_profile,
    load_bars,
    split_train_validation,
)
from trader_optimizer.optimizer import OptimizationSettings, run_optimization
from trader_optimizer.postgres import (
    PostgresSettings,
    insert_optimizer_batch_results,
    insert_optimizer_run,
    insert_optimizer_trials,
    postgres_connection,
)
from trader_optimizer.simple_strategies import (
    SimpleResult,
    bollinger_breakout_directions,
    ema_cross_directions,
    simulate_buy_and_hold,
    simulate_equal_weight_buy_and_hold,
    momentum_factor_weights,
    opening_range_breakout_directions,
    pairs_trading_weights,
    rsi_divergence_directions,
    simulate_portfolio,
    simulate_target_directions,
    sma_cross_directions,
    volatility_target_weights,
)
from trader_optimizer.strategy_configs import StrategyCandidate


@dataclass(frozen=True)
class BatchSettings:
    pg_settings: PostgresSettings
    optuna_storage_url: str
    output_dir: Path
    trials: int
    max_bars: int
    preferred_bar_size: str | None
    train_fraction: float
    verbose: bool
    export_config_dir: Path | None = None
    workers: int = 0
    backtester_settings: BackTesterSettings | None = None
    start_utc: str | None = None
    end_utc: str | None = None


@dataclass(frozen=True)
class BatchItemResult:
    name: str
    strategy_type: str
    variant: str
    symbols: tuple[str, ...]
    source_config: str
    status: str
    output_dir: str | None = None
    best_config: str | None = None
    summary: str | None = None
    best_value: float | None = None
    strategy_return_pct: float | None = None
    benchmark_return_pct: float | None = None
    excess_return_pct: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def optimize_candidates(
    candidates: list[StrategyCandidate],
    settings: BatchSettings,
) -> list[BatchItemResult]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if settings.backtester_settings is not None:
        settings = replace(
            settings,
            backtester_settings=prepare_backtester(settings.backtester_settings),
        )
    worker_count = _resolve_workers(settings.workers, len(candidates))

    if worker_count == 1:
        results = []
        for index, candidate in enumerate(candidates, start=1):
            _print_candidate_start(index, len(candidates), candidate, settings)
            result = _optimize_candidate(candidate, settings)
            _print_candidate_finish(index, len(candidates), result, settings)
            results.append(result)
    else:
        results_by_index: list[BatchItemResult | None] = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for index, candidate in enumerate(candidates, start=1):
                _print_candidate_start(index, len(candidates), candidate, settings)
                future = executor.submit(_optimize_candidate, candidate, settings)
                futures[future] = index
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results_by_index[index - 1] = result
                _print_candidate_finish(index, len(candidates), result, settings)
        results = [result for result in results_by_index if result is not None]

    _write_batch_summary(settings, results)
    if settings.export_config_dir:
        _export_best_configs(settings.export_config_dir, results)
    return results


def _resolve_workers(requested_workers: int, candidate_count: int) -> int:
    if candidate_count <= 1:
        return 1
    if requested_workers > 0:
        return min(requested_workers, candidate_count)
    return min(candidate_count, max(1, min(os.cpu_count() or 1, 4)))


def _optimize_candidate(
    candidate: StrategyCandidate,
    settings: BatchSettings,
) -> BatchItemResult:
    try:
        if candidate.strategy_type == "ConstantStepOffset":
            return _optimize_cso(candidate, settings)
        if candidate.strategy_type in {"MovingAverageCross", "TechnicalSignal"}:
            return _optimize_single_signal(candidate, settings)
        if candidate.strategy_type == "PortfolioAllocation":
            return _optimize_portfolio(candidate, settings)
        return _skipped(candidate, f"Unsupported strategy_type {candidate.strategy_type}")
    except Exception as exc:  # noqa: BLE001 - batch summaries should retain failures.
        return _skipped(candidate, str(exc))


def _print_candidate_start(
    index: int,
    total: int,
    candidate: StrategyCandidate,
    settings: BatchSettings,
) -> None:
    if not settings.verbose:
        return
    print(
        f"[{index}/{total}] start {candidate.name} "
        f"{candidate.strategy_type} {candidate.symbols}"
    )


def _print_candidate_finish(
    index: int,
    total: int,
    result: BatchItemResult,
    settings: BatchSettings,
) -> None:
    if not settings.verbose:
        return
    suffix = ""
    if result.excess_return_pct is not None:
        suffix = f" excess_return_pct={result.excess_return_pct:.6f}"
    elif result.reason:
        suffix = f" reason={result.reason}"
    print(f"[{index}/{total}] finish {result.name} status={result.status}{suffix}")


def write_optimization_plan(
    candidates: list[StrategyCandidate],
    settings: BatchSettings,
    plan_path: Path,
) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TraderOptimizer Config Optimization Plan",
        "",
        "This plan is generated from the current Trader strategy config JSON files.",
        "It describes what Optuna will tune, which PostgreSQL data profile will be used, "
        "and where the resulting artifacts will be written.",
        "",
        "## Run Settings",
        "",
        f"- PostgreSQL data: `{settings.pg_settings.display}`",
        f"- Optuna storage: `{settings.optuna_storage_url}`",
        f"- Output directory: `{settings.output_dir}`",
        f"- Exported configs: `{settings.export_config_dir}`"
        if settings.export_config_dir
        else "- Exported configs: not requested",
        f"- Trials per config: `{settings.trials}`",
        f"- Max bars per symbol: `{settings.max_bars}`",
        f"- Start UTC: `{settings.start_utc or 'auto'}`",
        f"- End UTC: `{settings.end_utc or 'auto'}`",
        f"- Train fraction: `{settings.train_fraction}`",
        f"- Preferred bar size: `{settings.preferred_bar_size or 'auto'}`",
        f"- Workers: `{_resolve_workers(settings.workers, len(candidates))}`",
        "",
        "## Search Spaces",
        "",
        "- `MovingAverageCross`: `fastWindow`, `slowWindow`, `orderQuantity`, and derived `orderQuantityInUSD`.",
        "- `TechnicalSignal` TS-002 EMA cross: `fastWindow`, `slowWindow`, and `orderQuantity`.",
        "- `TechnicalSignal` TS-003 Bollinger breakout: `middleWindow`, `trendWindow`, `bandStddev`, and `orderQuantity`.",
        "- `TechnicalSignal` TS-004 opening range breakout: `openingRangeBars`, `useAtrStop`, `atrWindow`, and `orderQuantity`.",
        "- `TechnicalSignal` TS-005 RSI divergence: `rsiPeriod`, `divergenceLookback`, and `orderQuantity`.",
        "- `PortfolioAllocation` QS-001 volatility targeting: `targetVolatility`, `volatilityWindow`, and `maxGrossExposure`.",
        "- `PortfolioAllocation` QS-002 momentum factor: `momentumLookback`, `momentumLegSize`, and `maxGrossExposure`.",
        "- `PortfolioAllocation` PAIRS-001 equity pairs: `pairWindow`, `pairEntryZ`, `pairExitZ`, and `maxGrossExposure`.",
        "",
        "## Objective",
        "",
        "The objective blends train and validation excess return versus a buy-and-hold benchmark for the same symbol set, "
        "then penalizes open inventory, drawdown, and no-trade configurations. "
        "Every exported config is validated with TraderCore BackTester and must have positive return, beat SPX, "
        "and beat same-stock buy-and-hold over the BackTester validation window.",
        "",
        "## Strategy Coverage",
        "",
        "| Config | Type | Variant | Symbols | Data profile | Tuned fields |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in candidates:
        lines.append(_plan_row(candidate, settings))
    lines.extend(
        [
            "",
            "## Validation Path",
            "",
            "1. Inspect the generated `best_summary.json`, `backtester` payload, and PostgreSQL `optimizer_trials` rows for each strategy.",
            "2. Promote only configs with a passing BackTester validation status.",
            "3. Use smaller `--max-bars`, `--start-utc`, or `--end-utc` windows when the BackTester validation cost is too high.",
            "",
        ]
    )
    plan_path.write_text("\n".join(lines), encoding="utf-8")


def _optimize_cso(
    candidate: StrategyCandidate,
    settings: BatchSettings,
) -> BatchItemResult:
    symbol = candidate.symbols[0]
    profile = choose_data_profile(
        settings.pg_settings,
        symbol,
        preferred_bar_size=settings.preferred_bar_size,
    )
    window = load_bars(
        pg_settings=settings.pg_settings,
        symbol=symbol,
        bar_size=profile.bar_size,
        what_to_show=profile.what_to_show,
        use_rth=profile.use_rth,
        start_utc=settings.start_utc,
        end_utc=settings.end_utc,
        max_bars=settings.max_bars,
    )
    output_dir = settings.output_dir / candidate.name
    artifacts = run_optimization(
        window,
        OptimizationSettings(
            trials=settings.trials,
            train_fraction=settings.train_fraction,
            output_dir=output_dir,
            study_name=f"{settings.output_dir.name}_{candidate.name}_cso",
            storage_url=settings.optuna_storage_url,
            pg_settings=settings.pg_settings,
            min_trades=2,
            verbose=False,
            backtester_settings=settings.backtester_settings,
        ),
    )
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    benchmark = summary.get("benchmark", {}).get("all", {})
    status, reason = _benchmark_status(benchmark)
    backtester_validation = summary.get("backtester")
    if isinstance(backtester_validation, dict):
        status = str(backtester_validation.get("status") or status)
        reason = (
            str(backtester_validation.get("reason"))
            if backtester_validation.get("reason")
            else reason
        )
        backtester_benchmarks = backtester_validation.get("benchmarks")
        if isinstance(backtester_benchmarks, dict):
            same_stock = backtester_benchmarks.get("same_stock_buy_and_hold")
            if isinstance(same_stock, dict):
                benchmark = {
                    "strategy_return_pct": backtester_validation.get("strategy", {}).get("return_pct")
                    if isinstance(backtester_validation.get("strategy"), dict)
                    else benchmark.get("strategy_return_pct"),
                    "benchmark_return_pct": same_stock.get("benchmark_return_pct"),
                    "excess_return_pct": same_stock.get("excess_return_pct"),
                }
    return BatchItemResult(
        name=candidate.name,
        strategy_type=candidate.strategy_type,
        variant=candidate.variant,
        symbols=candidate.symbols,
        source_config=str(candidate.path),
        status=status,
        output_dir=str(output_dir),
        best_config=str(artifacts.config_path),
        summary=str(artifacts.summary_path),
        best_value=artifacts.best_value,
        strategy_return_pct=_float_or_none(benchmark.get("strategy_return_pct")),
        benchmark_return_pct=_float_or_none(benchmark.get("benchmark_return_pct")),
        excess_return_pct=_float_or_none(benchmark.get("excess_return_pct")),
        reason=reason,
    )


def _optimize_single_signal(
    candidate: StrategyCandidate,
    settings: BatchSettings,
) -> BatchItemResult:
    symbol = candidate.symbols[0]
    profile = choose_data_profile(
        settings.pg_settings,
        symbol,
        preferred_bar_size=settings.preferred_bar_size,
    )
    bars = load_bars(
        pg_settings=settings.pg_settings,
        symbol=symbol,
        bar_size=profile.bar_size,
        what_to_show=profile.what_to_show,
        use_rth=profile.use_rth,
        start_utc=settings.start_utc,
        end_utc=settings.end_utc,
        max_bars=settings.max_bars,
    ).bars
    train_bars, validation_bars = split_train_validation(
        bars,
        settings.train_fraction,
    )
    output_dir = settings.output_dir / candidate.name
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"{settings.output_dir.name}_{candidate.name}_simple"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=settings.optuna_storage_url,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _single_signal_objective(
            candidate,
            train_bars,
            validation_bars,
            trial,
        ),
        n_trials=settings.trials,
        show_progress_bar=False,
    )
    best_config, all_result = _single_signal_config_and_result(
        candidate,
        bars,
        study.best_trial,
    )
    _, train_result = _single_signal_config_and_result(
        candidate,
        train_bars,
        study.best_trial,
    )
    _, validation_result = _single_signal_config_and_result(
        candidate,
        validation_bars,
        study.best_trial,
    )
    train_benchmark = simulate_buy_and_hold(
        train_bars,
        allocated_capital=train_result.allocated_capital,
    )
    validation_benchmark = simulate_buy_and_hold(
        validation_bars,
        allocated_capital=validation_result.allocated_capital,
    )
    all_benchmark = simulate_buy_and_hold(
        bars,
        allocated_capital=all_result.allocated_capital,
    )
    benchmark = {
        "name": "buy_and_hold",
        "train": _benchmark_comparison(train_result, train_benchmark),
        "validation": _benchmark_comparison(validation_result, validation_benchmark),
        "all": _benchmark_comparison(all_result, all_benchmark),
    }
    status, reason = _benchmark_status(benchmark["all"])
    best_config["ledgerPath"] = f"data/TraderLedger/{candidate.name}_OPTIMIZED"
    best_config["ledgerContextCollection"] = f"{candidate.name}_OPTIMIZED_context"

    config_path = output_dir / "best_config.json"
    summary_path = output_dir / "best_summary.json"
    metrics = {
        "train": train_result.to_dict(),
        "validation": validation_result.to_dict(),
        "all": all_result.to_dict(),
    }
    hyperparameters = dict(study.best_trial.params)
    write_json(config_path, best_config)
    backtester_validation = _validate_batch_backtester(
        candidate=candidate,
        config_path=config_path,
        profile=profile_for_bars(
            bar_size=profile.bar_size,
            what_to_show=profile.what_to_show,
            use_rth=profile.use_rth,
            symbol_bars={symbol: bars},
        ),
        symbol_bars={symbol: bars},
        settings=settings,
        output_dir=output_dir,
    )
    if backtester_validation is not None:
        status = backtester_validation.status
        reason = backtester_validation.reason
    write_json(
        summary_path,
        {
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number,
            "strategy_type": candidate.strategy_type,
            "variant": candidate.variant,
            "symbols": list(candidate.symbols),
            "source_config": str(candidate.path),
            "data_profile": profile.__dict__,
            "metrics": metrics,
            "benchmark": benchmark,
            "backtester": backtester_validation.to_dict()
            if backtester_validation is not None
            else None,
            "hyperparameters": hyperparameters,
        },
    )
    with postgres_connection(settings.pg_settings) as conn:
        run_id = insert_optimizer_run(
            conn,
            study_name=study_name,
            run_kind="batch",
            symbol=symbol,
            strategy_name=candidate.name,
            strategy_type=candidate.strategy_type,
            variant=candidate.variant,
            output_dir=output_dir,
            config_path=config_path,
            summary_path=summary_path,
            best_value=float(study.best_value),
            data_source=settings.pg_settings.display,
            bar_size=profile.bar_size,
            what_to_show=profile.what_to_show,
            use_rth=profile.use_rth,
            first_timestamp=bars[0].timestamp_utc,
            last_timestamp=bars[-1].timestamp_utc,
            bars=len(bars),
            metrics=metrics,
            hyperparameters=hyperparameters,
        )
        insert_optimizer_trials(conn, run_id, study.trials)
    return BatchItemResult(
        name=candidate.name,
        strategy_type=candidate.strategy_type,
        variant=candidate.variant,
        symbols=candidate.symbols,
        source_config=str(candidate.path),
        status=status,
        output_dir=str(output_dir),
        best_config=str(config_path),
        summary=str(summary_path),
        best_value=float(study.best_value),
        strategy_return_pct=(
            backtester_validation.strategy_return_pct
            if backtester_validation is not None
            else all_result.return_pct
        ),
        benchmark_return_pct=(
            backtester_validation.same_stock_return_pct
            if backtester_validation is not None
            else all_benchmark.return_pct
        ),
        excess_return_pct=(
            backtester_validation.same_stock_excess_return_pct
            if backtester_validation is not None
            else all_result.return_pct - all_benchmark.return_pct
        ),
        reason=reason,
    )


def _single_signal_objective(
    candidate: StrategyCandidate,
    train_bars: list[Bar],
    validation_bars: list[Bar],
    trial: optuna.Trial,
) -> float:
    _, train_result = _single_signal_config_and_result(candidate, train_bars, trial)
    _, validation_result = _single_signal_config_and_result(
        candidate,
        validation_bars,
        trial,
    )
    train_benchmark = simulate_buy_and_hold(
        train_bars,
        allocated_capital=train_result.allocated_capital,
    )
    validation_benchmark = simulate_buy_and_hold(
        validation_bars,
        allocated_capital=validation_result.allocated_capital,
    )
    train_excess_return = train_result.return_pct - train_benchmark.return_pct
    validation_excess_return = (
        validation_result.return_pct - validation_benchmark.return_pct
    )
    trial.set_user_attr("train_net_pnl", train_result.net_pnl)
    trial.set_user_attr("train_return_pct", train_result.return_pct)
    trial.set_user_attr("train_buy_hold_return_pct", train_benchmark.return_pct)
    trial.set_user_attr("train_excess_return_pct", train_excess_return)
    trial.set_user_attr("train_fills", train_result.fills)
    trial.set_user_attr("validation_net_pnl", validation_result.net_pnl)
    trial.set_user_attr("validation_return_pct", validation_result.return_pct)
    trial.set_user_attr(
        "validation_buy_hold_return_pct",
        validation_benchmark.return_pct,
    )
    trial.set_user_attr("validation_excess_return_pct", validation_excess_return)
    trial.set_user_attr("validation_fills", validation_result.fills)
    trial.set_user_attr(
        "validation_final_position_value",
        validation_result.final_position_value,
    )
    total_fills = train_result.fills + validation_result.fills
    no_trade_penalty = 0.02 if total_fills == 0 else 0.0
    inventory_penalty = (
        abs(validation_result.final_position_value)
        / max(validation_result.allocated_capital, 1.0)
        * 0.25
    )
    drawdown_penalty = (
        validation_result.max_drawdown
        / max(validation_result.allocated_capital, 1.0)
        * 0.10
    )
    return (
        train_excess_return * 0.70
        + validation_excess_return * 0.30
        - inventory_penalty
        - drawdown_penalty
        - no_trade_penalty
    )


def _single_signal_config_and_result(
    candidate: StrategyCandidate,
    bars: list[Bar],
    trial: optuna.Trial,
) -> tuple[dict[str, Any], SimpleResult]:
    config = dict(candidate.config)
    quantity = float(trial.suggest_int("orderQuantity", 1, 20))

    if candidate.strategy_type == "MovingAverageCross":
        fast = trial.suggest_int("fastWindow", 2, 30)
        slow = trial.suggest_int("slowWindow", fast + 1, 120)
        directions = sma_cross_directions(bars, fast, slow)
        config.update(
            {
                "fastWindow": fast,
                "slowWindow": slow,
                "orderQuantity": quantity,
                "orderQuantityInUSD": quantity * bars[0].close,
            }
        )
    else:
        signal_type = str(config.get("signal_type", "")).upper()
        if signal_type in {"TS-002", "EMACROSS", "EMA_CROSS"}:
            fast = trial.suggest_int("fastWindow", 2, 30)
            slow = trial.suggest_int("slowWindow", fast + 1, 160)
            directions = ema_cross_directions(bars, fast, slow)
            config.update({"fastWindow": fast, "slowWindow": slow})
        elif signal_type in {"TS-003", "BOLLINGERBREAKOUT", "BOLLINGER_BREAKOUT"}:
            middle = trial.suggest_int("middleWindow", 5, 80)
            trend = trial.suggest_int("trendWindow", middle + 1, 180)
            band = trial.suggest_float("bandStddev", 0.5, 3.5)
            directions = bollinger_breakout_directions(bars, middle, trend, band)
            config.update(
                {
                    "middleWindow": middle,
                    "trendWindow": trend,
                    "bandStddev": band,
                }
            )
        elif signal_type in {"TS-004", "OPENINGRANGEBREAKOUT", "OPENING_RANGE_BREAKOUT", "ORB"}:
            opening_range = trial.suggest_int("openingRangeBars", 3, 80)
            directions = opening_range_breakout_directions(bars, opening_range)
            config.update(
                {
                    "openingRangeBars": opening_range,
                    "useAtrStop": trial.suggest_categorical("useAtrStop", [False, True]),
                    "atrWindow": trial.suggest_int("atrWindow", 5, 80),
                }
            )
        elif signal_type in {"TS-005", "RSIDIVERGENCE", "RSI_DIVERGENCE"}:
            rsi_period = trial.suggest_int("rsiPeriod", 5, 40)
            lookback = trial.suggest_int("divergenceLookback", 3, 60)
            directions = rsi_divergence_directions(bars, rsi_period, lookback)
            config.update(
                {
                    "rsiPeriod": rsi_period,
                    "divergenceLookback": lookback,
                }
            )
        else:
            raise ValueError(f"Unsupported TechnicalSignal signal_type {signal_type}")
        config["orderQuantity"] = quantity

    result = simulate_target_directions(bars, directions, quantity)
    return config, result


def _optimize_portfolio(
    candidate: StrategyCandidate,
    settings: BatchSettings,
) -> BatchItemResult:
    symbol_bars: dict[str, list[Bar]] = {}
    profiles: dict[str, Any] = {}
    for symbol in candidate.symbols:
        profile = choose_data_profile(
            settings.pg_settings,
            symbol,
            preferred_bar_size=settings.preferred_bar_size,
        )
        profiles[symbol] = profile.__dict__
        symbol_bars[symbol] = load_bars(
            pg_settings=settings.pg_settings,
            symbol=symbol,
            bar_size=profile.bar_size,
            what_to_show=profile.what_to_show,
            use_rth=profile.use_rth,
            start_utc=settings.start_utc,
            end_utc=settings.end_utc,
            max_bars=settings.max_bars,
        ).bars

    train_symbol_bars: dict[str, list[Bar]] = {}
    validation_symbol_bars: dict[str, list[Bar]] = {}
    for symbol, bars in symbol_bars.items():
        train_bars, validation_bars = split_train_validation(
            bars,
            settings.train_fraction,
        )
        train_symbol_bars[symbol] = train_bars
        validation_symbol_bars[symbol] = validation_bars

    output_dir = settings.output_dir / candidate.name
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"{settings.output_dir.name}_{candidate.name}_portfolio"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=settings.optuna_storage_url,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _portfolio_objective(
            candidate,
            train_symbol_bars,
            validation_symbol_bars,
            trial,
        ),
        n_trials=settings.trials,
        show_progress_bar=False,
    )
    best_config, all_result = _portfolio_config_and_result(
        candidate,
        symbol_bars,
        study.best_trial,
    )
    _, train_result = _portfolio_config_and_result(
        candidate,
        train_symbol_bars,
        study.best_trial,
    )
    _, validation_result = _portfolio_config_and_result(
        candidate,
        validation_symbol_bars,
        study.best_trial,
    )
    train_benchmark = simulate_equal_weight_buy_and_hold(
        train_symbol_bars,
        train_result.allocated_capital,
    )
    validation_benchmark = simulate_equal_weight_buy_and_hold(
        validation_symbol_bars,
        validation_result.allocated_capital,
    )
    all_benchmark = simulate_equal_weight_buy_and_hold(
        symbol_bars,
        all_result.allocated_capital,
    )
    benchmark = {
        "name": "equal_weight_buy_and_hold",
        "train": _benchmark_comparison(train_result, train_benchmark),
        "validation": _benchmark_comparison(validation_result, validation_benchmark),
        "all": _benchmark_comparison(all_result, all_benchmark),
    }
    status, reason = _benchmark_status(benchmark["all"])
    best_config["ledgerPath"] = f"data/TraderLedger/{candidate.name}_OPTIMIZED"
    best_config["ledgerContextCollection"] = f"{candidate.name}_OPTIMIZED_context"
    config_path = output_dir / "best_config.json"
    summary_path = output_dir / "best_summary.json"
    metrics = {
        "train": train_result.to_dict(),
        "validation": validation_result.to_dict(),
        "all": all_result.to_dict(),
    }
    hyperparameters = dict(study.best_trial.params)
    write_json(config_path, best_config)
    backtester_validation = _validate_batch_backtester(
        candidate=candidate,
        config_path=config_path,
        profile=_portfolio_backtest_profile(profiles, symbol_bars),
        symbol_bars=symbol_bars,
        settings=settings,
        output_dir=output_dir,
    )
    if backtester_validation is not None:
        status = backtester_validation.status
        reason = backtester_validation.reason
    write_json(
        summary_path,
        {
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number,
            "strategy_type": candidate.strategy_type,
            "variant": candidate.variant,
            "symbols": list(candidate.symbols),
            "source_config": str(candidate.path),
            "data_profiles": profiles,
            "metrics": metrics,
            "benchmark": benchmark,
            "backtester": backtester_validation.to_dict()
            if backtester_validation is not None
            else None,
            "hyperparameters": hyperparameters,
        },
    )
    with postgres_connection(settings.pg_settings) as conn:
        run_id = insert_optimizer_run(
            conn,
            study_name=study_name,
            run_kind="batch",
            symbol=",".join(candidate.symbols),
            strategy_name=candidate.name,
            strategy_type=candidate.strategy_type,
            variant=candidate.variant,
            output_dir=output_dir,
            config_path=config_path,
            summary_path=summary_path,
            best_value=float(study.best_value),
            data_source=settings.pg_settings.display,
            bar_size=_profile_value_list(profiles, "bar_size"),
            what_to_show=_profile_value_list(profiles, "what_to_show"),
            use_rth=_profile_use_rth(profiles),
            first_timestamp=_first_timestamp(symbol_bars),
            last_timestamp=_last_timestamp(symbol_bars),
            bars=_total_bars(symbol_bars),
            metrics=metrics,
            hyperparameters=hyperparameters,
        )
        insert_optimizer_trials(conn, run_id, study.trials)
    return BatchItemResult(
        name=candidate.name,
        strategy_type=candidate.strategy_type,
        variant=candidate.variant,
        symbols=candidate.symbols,
        source_config=str(candidate.path),
        status=status,
        output_dir=str(output_dir),
        best_config=str(config_path),
        summary=str(summary_path),
        best_value=float(study.best_value),
        strategy_return_pct=(
            backtester_validation.strategy_return_pct
            if backtester_validation is not None
            else all_result.return_pct
        ),
        benchmark_return_pct=(
            backtester_validation.same_stock_return_pct
            if backtester_validation is not None
            else all_benchmark.return_pct
        ),
        excess_return_pct=(
            backtester_validation.same_stock_excess_return_pct
            if backtester_validation is not None
            else all_result.return_pct - all_benchmark.return_pct
        ),
        reason=reason,
    )


def _portfolio_objective(
    candidate: StrategyCandidate,
    train_symbol_bars: dict[str, list[Bar]],
    validation_symbol_bars: dict[str, list[Bar]],
    trial: optuna.Trial,
) -> float:
    _, train_result = _portfolio_config_and_result(
        candidate,
        train_symbol_bars,
        trial,
    )
    _, validation_result = _portfolio_config_and_result(
        candidate,
        validation_symbol_bars,
        trial,
    )
    train_benchmark = simulate_equal_weight_buy_and_hold(
        train_symbol_bars,
        train_result.allocated_capital,
    )
    validation_benchmark = simulate_equal_weight_buy_and_hold(
        validation_symbol_bars,
        validation_result.allocated_capital,
    )
    train_excess_return = train_result.return_pct - train_benchmark.return_pct
    validation_excess_return = (
        validation_result.return_pct - validation_benchmark.return_pct
    )
    trial.set_user_attr("train_net_pnl", train_result.net_pnl)
    trial.set_user_attr("train_return_pct", train_result.return_pct)
    trial.set_user_attr("train_buy_hold_return_pct", train_benchmark.return_pct)
    trial.set_user_attr("train_excess_return_pct", train_excess_return)
    trial.set_user_attr("train_fills", train_result.fills)
    trial.set_user_attr("validation_net_pnl", validation_result.net_pnl)
    trial.set_user_attr("validation_return_pct", validation_result.return_pct)
    trial.set_user_attr(
        "validation_buy_hold_return_pct",
        validation_benchmark.return_pct,
    )
    trial.set_user_attr("validation_excess_return_pct", validation_excess_return)
    trial.set_user_attr("validation_fills", validation_result.fills)
    trial.set_user_attr(
        "validation_final_position_value",
        validation_result.final_position_value,
    )
    inventory_penalty = (
        abs(validation_result.final_position_value)
        / max(validation_result.allocated_capital, 1.0)
        * 0.05
    )
    drawdown_penalty = (
        validation_result.max_drawdown
        / max(validation_result.allocated_capital, 1.0)
        * 0.10
    )
    no_trade_penalty = 0.02 if train_result.fills + validation_result.fills == 0 else 0.0
    return (
        train_excess_return * 0.70
        + validation_excess_return * 0.30
        - inventory_penalty
        - drawdown_penalty
        - no_trade_penalty
    )


def _portfolio_config_and_result(
    candidate: StrategyCandidate,
    symbol_bars: dict[str, list[Bar]],
    trial: optuna.Trial,
) -> tuple[dict[str, Any], SimpleResult]:
    config = dict(candidate.config)
    allocation_type = str(config.get("allocation_type", "")).upper()
    notional = float(config.get("portfolioNotional", 10000.0))
    max_gross = trial.suggest_float("maxGrossExposure", 0.25, 2.0)
    if allocation_type == "QS-001":
        target_vol = trial.suggest_float("targetVolatility", 0.03, 0.50)
        vol_window = trial.suggest_int("volatilityWindow", 3, 80)
        weight_fn = volatility_target_weights(vol_window, target_vol)
        config.update(
            {
                "targetVolatility": target_vol,
                "volatilityWindow": vol_window,
                "maxGrossExposure": max_gross,
            }
        )
    elif allocation_type == "QS-002":
        lookback = trial.suggest_int("momentumLookback", 2, 80)
        leg_size = trial.suggest_int("momentumLegSize", 1, max(1, len(candidate.symbols) // 2))
        weight_fn = momentum_factor_weights(lookback, leg_size)
        config.update(
            {
                "momentumLookback": lookback,
                "momentumLegSize": leg_size,
                "maxGrossExposure": max_gross,
            }
        )
    elif allocation_type == "PAIRS-001":
        pair_window = trial.suggest_int("pairWindow", 4, 120)
        entry_z = trial.suggest_float("pairEntryZ", 0.5, 3.0)
        exit_z = trial.suggest_float("pairExitZ", 0.05, min(0.8, entry_z * 0.8))
        pairs = [
            (str(item["left"]), str(item["right"]))
            for item in config.get("pairs", [])
            if isinstance(item, dict) and item.get("left") and item.get("right")
        ]
        weight_fn = pairs_trading_weights(pairs, pair_window, entry_z, exit_z)
        config.update(
            {
                "pairWindow": pair_window,
                "pairEntryZ": entry_z,
                "pairExitZ": exit_z,
                "maxGrossExposure": max_gross,
            }
        )
    else:
        raise ValueError(f"Unsupported allocation_type {allocation_type}")

    result = simulate_portfolio(
        symbol_bars,
        weight_fn,
        portfolio_notional=notional,
        max_gross_exposure=max_gross,
        min_trade_quantity=float(config.get("minTradeQuantity", 0.0001)),
    )
    return config, result


def _benchmark_comparison(
    strategy_result: Any,
    benchmark_result: SimpleResult,
) -> dict[str, object]:
    return {
        "benchmark": benchmark_result.to_dict(),
        "strategy_return_pct": strategy_result.return_pct,
        "benchmark_return_pct": benchmark_result.return_pct,
        "excess_return_pct": strategy_result.return_pct - benchmark_result.return_pct,
        "strategy_net_pnl": strategy_result.net_pnl,
        "benchmark_net_pnl": benchmark_result.net_pnl,
        "excess_net_pnl": strategy_result.net_pnl - benchmark_result.net_pnl,
    }


def _validate_batch_backtester(
    *,
    candidate: StrategyCandidate,
    config_path: Path,
    profile: BacktestProfile,
    symbol_bars: dict[str, list[Bar]],
    settings: BatchSettings,
    output_dir: Path,
) -> BacktestValidationResult | None:
    if settings.backtester_settings is None:
        return None
    return validate_with_backtester(
        strategy_config_path=config_path,
        strategy_name=candidate.name,
        symbols=candidate.symbols,
        profile=profile,
        symbol_bars=symbol_bars,
        settings=settings.backtester_settings,
        output_dir=output_dir / "backtester",
    )


def _portfolio_backtest_profile(
    profiles: dict[str, Any],
    symbol_bars: dict[str, list[Bar]],
) -> BacktestProfile:
    values = list(profiles.values())
    if not values:
        raise ValueError("No data profiles for portfolio BackTester validation")
    bar_size = str(values[0]["bar_size"])
    what_to_show = str(values[0]["what_to_show"])
    use_rth = int(values[0]["use_rth"])
    for profile in values[1:]:
        if (
            str(profile["bar_size"]) != bar_size
            or str(profile["what_to_show"]) != what_to_show
            or int(profile["use_rth"]) != use_rth
        ):
            raise ValueError(
                "BackTester validation requires one common data profile for "
                f"portfolio symbols, got {profiles}"
            )
    return profile_for_bars(
        bar_size=bar_size,
        what_to_show=what_to_show,
        use_rth=use_rth,
        symbol_bars=symbol_bars,
    )


def _benchmark_status(comparison: dict[str, object]) -> tuple[str, str | None]:
    strategy_return = _float_or_none(comparison.get("strategy_return_pct"))
    benchmark_return = _float_or_none(comparison.get("benchmark_return_pct"))
    excess_return = _float_or_none(comparison.get("excess_return_pct"))
    if excess_return is not None and excess_return > 0.0:
        return "ok", None
    if strategy_return is None or benchmark_return is None:
        return "benchmark_failed", "Missing buy-and-hold benchmark comparison"
    return (
        "benchmark_failed",
        "Best simulated return "
        f"{strategy_return:.6f} did not beat buy-and-hold {benchmark_return:.6f}",
    )


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _skipped(candidate: StrategyCandidate, reason: str) -> BatchItemResult:
    return BatchItemResult(
        name=candidate.name,
        strategy_type=candidate.strategy_type,
        variant=candidate.variant,
        symbols=candidate.symbols,
        source_config=str(candidate.path),
        status="skipped",
        reason=reason,
    )


def _write_batch_summary(
    settings: BatchSettings,
    results: list[BatchItemResult],
) -> None:
    output_dir = settings.output_dir
    write_json(
        output_dir / "batch_summary.json",
        {"results": [result.to_dict() for result in results]},
    )
    with postgres_connection(settings.pg_settings) as conn:
        insert_optimizer_batch_results(conn, output_dir.name, results)


def _export_best_configs(
    export_dir: Path,
    results: list[BatchItemResult],
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for stale_config in export_dir.glob("*.optimized.json"):
        stale_config.unlink()
    index: list[dict[str, object]] = []
    for result in results:
        if result.status != "ok" or not result.best_config:
            continue
        source = Path(result.best_config)
        exported = export_dir / f"{result.name}.optimized.json"
        shutil.copyfile(source, exported)
        index.append(
            {
                "name": result.name,
                "strategy_type": result.strategy_type,
                "variant": result.variant,
                "symbols": list(result.symbols),
                "source_config": result.source_config,
                "best_value": result.best_value,
                "strategy_return_pct": result.strategy_return_pct,
                "benchmark_return_pct": result.benchmark_return_pct,
                "excess_return_pct": result.excess_return_pct,
                "exported_config": str(exported),
                "run_config": result.best_config,
                "summary": result.summary,
            }
        )
    write_json(export_dir / "index.json", {"configs": index})


def _plan_row(candidate: StrategyCandidate, settings: BatchSettings) -> str:
    symbols = ", ".join(candidate.symbols)
    tuned_fields = ", ".join(_tuned_fields(candidate))
    try:
        profiles = []
        for symbol in candidate.symbols:
            profile = choose_data_profile(
                settings.pg_settings,
                symbol,
                preferred_bar_size=settings.preferred_bar_size,
            )
            profiles.append(
                f"{symbol}: {profile.bar_size} {profile.what_to_show} rth={profile.use_rth}"
            )
        data_profile = "<br>".join(profiles)
    except Exception as exc:  # noqa: BLE001 - plan should surface missing data.
        data_profile = f"unavailable: {exc}"
    return (
        f"| `{candidate.name}` | `{candidate.strategy_type}` | `{candidate.variant}` | "
        f"{symbols} | {data_profile} | {tuned_fields} |"
    )


def _tuned_fields(candidate: StrategyCandidate) -> list[str]:
    if candidate.strategy_type == "MovingAverageCross":
        return ["fastWindow", "slowWindow", "orderQuantity", "orderQuantityInUSD"]
    if candidate.strategy_type == "TechnicalSignal":
        variant = candidate.variant.upper()
        if variant in {"TS-002", "EMACROSS", "EMA_CROSS"}:
            return ["fastWindow", "slowWindow", "orderQuantity"]
        if variant in {"TS-003", "BOLLINGERBREAKOUT", "BOLLINGER_BREAKOUT"}:
            return ["middleWindow", "trendWindow", "bandStddev", "orderQuantity"]
        if variant in {"TS-004", "OPENINGRANGEBREAKOUT", "OPENING_RANGE_BREAKOUT", "ORB"}:
            return ["openingRangeBars", "useAtrStop", "atrWindow", "orderQuantity"]
        if variant in {"TS-005", "RSIDIVERGENCE", "RSI_DIVERGENCE"}:
            return ["rsiPeriod", "divergenceLookback", "orderQuantity"]
        return ["orderQuantity"]
    if candidate.strategy_type == "PortfolioAllocation":
        variant = candidate.variant.upper()
        if variant == "QS-001":
            return ["targetVolatility", "volatilityWindow", "maxGrossExposure"]
        if variant == "QS-002":
            return ["momentumLookback", "momentumLegSize", "maxGrossExposure"]
        if variant == "PAIRS-001":
            return ["pairWindow", "pairEntryZ", "pairExitZ", "maxGrossExposure"]
        return ["maxGrossExposure"]
    if candidate.strategy_type == "ConstantStepOffset":
        return [
            "baseline_quantile",
            "step_delta_pct",
            "execution_steps",
            "threshold_pct_of_step",
            "order_quantity_usd",
        ]
    return []


def _profile_value_list(profiles: dict[str, Any], field: str) -> str:
    return ",".join(
        sorted(
            {
                str(profile[field])
                for profile in profiles.values()
                if profile.get(field) is not None
            }
        )
    )


def _profile_use_rth(profiles: dict[str, Any]) -> int | None:
    values = {
        int(profile["use_rth"])
        for profile in profiles.values()
        if profile.get("use_rth") is not None
    }
    if len(values) == 1:
        return next(iter(values))
    return None


def _first_timestamp(symbol_bars: dict[str, list[Bar]]) -> str:
    return min(bar.timestamp_utc for bars in symbol_bars.values() for bar in bars)


def _last_timestamp(symbol_bars: dict[str, list[Bar]]) -> str:
    return max(bar.timestamp_utc for bars in symbol_bars.values() for bar in bars)


def _total_bars(symbol_bars: dict[str, list[Bar]]) -> int:
    return sum(len(bars) for bars in symbol_bars.values())
