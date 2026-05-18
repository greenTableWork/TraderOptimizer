from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

from trader_optimizer.config import write_json
from trader_optimizer.data import (
    Bar,
    choose_data_profile,
    load_bars,
    split_train_validation,
)
from trader_optimizer.optimizer import OptimizationSettings, run_optimization
from trader_optimizer.simple_strategies import (
    SimpleResult,
    bollinger_breakout_directions,
    ema_cross_directions,
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
    db_path: Path
    output_dir: Path
    trials: int
    max_bars: int
    preferred_bar_size: str | None
    train_fraction: float
    verbose: bool
    export_config_dir: Path | None = None


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
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def optimize_candidates(
    candidates: list[StrategyCandidate],
    settings: BatchSettings,
) -> list[BatchItemResult]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[BatchItemResult] = []
    for index, candidate in enumerate(candidates, start=1):
        if settings.verbose:
            print(
                f"[{index}/{len(candidates)}] {candidate.name} "
                f"{candidate.strategy_type} {candidate.symbols}"
            )
        try:
            if candidate.strategy_type == "ConstantStepOffset":
                result = _optimize_cso(candidate, settings)
            elif candidate.strategy_type in {"MovingAverageCross", "TechnicalSignal"}:
                result = _optimize_single_signal(candidate, settings)
            elif candidate.strategy_type == "PortfolioAllocation":
                result = _optimize_portfolio(candidate, settings)
            else:
                result = _skipped(candidate, f"Unsupported strategy_type {candidate.strategy_type}")
        except Exception as exc:  # noqa: BLE001 - batch summaries should retain failures.
            result = _skipped(candidate, str(exc))
        results.append(result)

    _write_batch_summary(settings.output_dir, results)
    if settings.export_config_dir:
        _export_best_configs(settings.export_config_dir, results)
    return results


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
        "It describes what Optuna will tune, which SQLite data profile will be used, "
        "and where the resulting artifacts will be written.",
        "",
        "## Run Settings",
        "",
        f"- SQLite data: `{settings.db_path}`",
        f"- Output directory: `{settings.output_dir}`",
        f"- Exported configs: `{settings.export_config_dir}`"
        if settings.export_config_dir
        else "- Exported configs: not requested",
        f"- Trials per config: `{settings.trials}`",
        f"- Max bars per symbol: `{settings.max_bars}`",
        f"- Train fraction: `{settings.train_fraction}`",
        f"- Preferred bar size: `{settings.preferred_bar_size or 'auto'}`",
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
        "The objective blends train and validation simulated return while penalizing open inventory, drawdown, and no-trade configurations. "
        "The simulator includes the local stock commission model and uses close-price fills for the non-CSO strategy families.",
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
            "1. Inspect the generated `best_summary.json` and `trials.csv` for each strategy.",
            "2. Run promising `best_config.json` files through TraderCore `BackTester`; the Python simulator is a fast search harness, not the execution source of truth.",
            "3. Promote only configs that survive the C++ backtest with acceptable fees, trade count, and drawdown.",
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
        settings.db_path,
        symbol,
        preferred_bar_size=settings.preferred_bar_size,
    )
    window = load_bars(
        db_path=settings.db_path,
        symbol=symbol,
        bar_size=profile.bar_size,
        what_to_show=profile.what_to_show,
        use_rth=profile.use_rth,
        max_bars=settings.max_bars,
    )
    output_dir = settings.output_dir / candidate.name
    artifacts = run_optimization(
        window,
        OptimizationSettings(
            trials=settings.trials,
            train_fraction=settings.train_fraction,
            output_dir=output_dir,
            study_name=f"{candidate.name}_cso",
            storage_path=output_dir / "optuna-study.db",
            min_trades=2,
            verbose=False,
        ),
    )
    return BatchItemResult(
        name=candidate.name,
        strategy_type=candidate.strategy_type,
        variant=candidate.variant,
        symbols=candidate.symbols,
        source_config=str(candidate.path),
        status="ok",
        output_dir=str(output_dir),
        best_config=str(artifacts.config_path),
        summary=str(artifacts.summary_path),
        best_value=artifacts.best_value,
    )


def _optimize_single_signal(
    candidate: StrategyCandidate,
    settings: BatchSettings,
) -> BatchItemResult:
    symbol = candidate.symbols[0]
    profile = choose_data_profile(
        settings.db_path,
        symbol,
        preferred_bar_size=settings.preferred_bar_size,
    )
    bars = load_bars(
        db_path=settings.db_path,
        symbol=symbol,
        bar_size=profile.bar_size,
        what_to_show=profile.what_to_show,
        use_rth=profile.use_rth,
        max_bars=settings.max_bars,
    ).bars
    train_bars, validation_bars = split_train_validation(
        bars,
        settings.train_fraction,
    )
    output_dir = settings.output_dir / candidate.name
    output_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{candidate.name}_simple",
        storage=f"sqlite:///{output_dir / 'optuna-study.db'}",
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
    best_config["ledgerPath"] = f"data/TraderLedger/{candidate.name}_OPTIMIZED.sqlite"
    best_config["ledgerContextCollection"] = f"{candidate.name}_OPTIMIZED_context"

    config_path = output_dir / "best_config.json"
    summary_path = output_dir / "best_summary.json"
    trials_path = output_dir / "trials.csv"
    write_json(config_path, best_config)
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
            "metrics": {
                "train": train_result.to_dict(),
                "validation": validation_result.to_dict(),
                "all": all_result.to_dict(),
            },
            "hyperparameters": study.best_trial.params,
        },
    )
    _write_trials(trials_path, study.trials)
    return BatchItemResult(
        name=candidate.name,
        strategy_type=candidate.strategy_type,
        variant=candidate.variant,
        symbols=candidate.symbols,
        source_config=str(candidate.path),
        status="ok",
        output_dir=str(output_dir),
        best_config=str(config_path),
        summary=str(summary_path),
        best_value=float(study.best_value),
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
    trial.set_user_attr("train_net_pnl", train_result.net_pnl)
    trial.set_user_attr("train_return_pct", train_result.return_pct)
    trial.set_user_attr("train_fills", train_result.fills)
    trial.set_user_attr("validation_net_pnl", validation_result.net_pnl)
    trial.set_user_attr("validation_return_pct", validation_result.return_pct)
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
        train_result.return_pct * 0.70
        + validation_result.return_pct * 0.30
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
            settings.db_path,
            symbol,
            preferred_bar_size=settings.preferred_bar_size,
        )
        profiles[symbol] = profile.__dict__
        symbol_bars[symbol] = load_bars(
            db_path=settings.db_path,
            symbol=symbol,
            bar_size=profile.bar_size,
            what_to_show=profile.what_to_show,
            use_rth=profile.use_rth,
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
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{candidate.name}_portfolio",
        storage=f"sqlite:///{output_dir / 'optuna-study.db'}",
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
    best_config["ledgerPath"] = f"data/TraderLedger/{candidate.name}_OPTIMIZED.sqlite"
    best_config["ledgerContextCollection"] = f"{candidate.name}_OPTIMIZED_context"
    config_path = output_dir / "best_config.json"
    summary_path = output_dir / "best_summary.json"
    trials_path = output_dir / "trials.csv"
    write_json(config_path, best_config)
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
            "metrics": {
                "train": train_result.to_dict(),
                "validation": validation_result.to_dict(),
                "all": all_result.to_dict(),
            },
            "hyperparameters": study.best_trial.params,
        },
    )
    _write_trials(trials_path, study.trials)
    return BatchItemResult(
        name=candidate.name,
        strategy_type=candidate.strategy_type,
        variant=candidate.variant,
        symbols=candidate.symbols,
        source_config=str(candidate.path),
        status="ok",
        output_dir=str(output_dir),
        best_config=str(config_path),
        summary=str(summary_path),
        best_value=float(study.best_value),
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
    trial.set_user_attr("train_net_pnl", train_result.net_pnl)
    trial.set_user_attr("train_return_pct", train_result.return_pct)
    trial.set_user_attr("train_fills", train_result.fills)
    trial.set_user_attr("validation_net_pnl", validation_result.net_pnl)
    trial.set_user_attr("validation_return_pct", validation_result.return_pct)
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
        train_result.return_pct * 0.70
        + validation_result.return_pct * 0.30
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
    output_dir: Path,
    results: list[BatchItemResult],
) -> None:
    write_json(
        output_dir / "batch_summary.json",
        {"results": [result.to_dict() for result in results]},
    )
    with (output_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "strategy_type",
                "variant",
                "symbols",
                "status",
                "best_value",
                "best_config",
                "summary",
                "reason",
                "source_config",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            row["symbols"] = ",".join(result.symbols)
            writer.writerow(row)


def _export_best_configs(
    export_dir: Path,
    results: list[BatchItemResult],
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
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
                settings.db_path,
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


def _write_trials(path: Path, trials: list[optuna.trial.FrozenTrial]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["number", "value", "state", "params_json", "user_attrs_json"],
        )
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "state": trial.state.name,
                    "params_json": json.dumps(trial.params, sort_keys=True),
                    "user_attrs_json": json.dumps(trial.user_attrs, sort_keys=True),
                }
            )
