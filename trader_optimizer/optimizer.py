from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

from trader_optimizer.config import build_constant_step_offset_config, write_json
from trader_optimizer.data import Bar, BarWindow, split_train_validation
from trader_optimizer.postgres import (
    PostgresSettings,
    insert_optimizer_fills,
    insert_optimizer_run,
    insert_optimizer_trials,
    postgres_connection,
)
from trader_optimizer.simple_strategies import simulate_buy_and_hold
from trader_optimizer.simulator import (
    SimulationResult,
    StrategyParams,
    simulate_constant_step_offset,
)


@dataclass(frozen=True)
class OptimizationSettings:
    trials: int
    train_fraction: float
    output_dir: Path
    study_name: str
    storage_url: str
    pg_settings: PostgresSettings
    min_trades: int
    verbose: bool


@dataclass(frozen=True)
class OptimizationArtifacts:
    output_dir: Path
    config_path: Path
    summary_path: Path
    study_storage: str
    best_value: float


def run_optimization(
    window: BarWindow,
    settings: OptimizationSettings,
) -> OptimizationArtifacts:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_bars, validation_bars = split_train_validation(
        window.bars,
        settings.train_fraction,
    )
    if settings.verbose:
        print("Preparing Optuna study")
        print(f"  output_dir: {settings.output_dir}")
        print(f"  study_name: {settings.study_name}")
        print(f"  storage: {settings.storage_url}")
        print(f"  train bars: {len(train_bars)}")
        print(f"  validation bars: {len(validation_bars)}")

    study = optuna.create_study(
        direction="maximize",
        study_name=settings.study_name,
        storage=settings.storage_url,
        load_if_exists=True,
    )

    objective = _Objective(
        train_bars=train_bars,
        validation_bars=validation_bars,
        min_trades=settings.min_trades,
    )
    study.optimize(objective, n_trials=settings.trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = _params_from_trial(best_trial, train_bars)
    train_result, _ = simulate_constant_step_offset(train_bars, best_params)
    validation_result, _ = simulate_constant_step_offset(validation_bars, best_params)
    all_result, fills = simulate_constant_step_offset(window.bars, best_params)
    train_benchmark = simulate_buy_and_hold(
        train_bars,
        allocated_capital=train_result.allocated_capital,
    )
    validation_benchmark = simulate_buy_and_hold(
        validation_bars,
        allocated_capital=validation_result.allocated_capital,
    )
    all_benchmark = simulate_buy_and_hold(
        window.bars,
        allocated_capital=all_result.allocated_capital,
    )

    config = build_constant_step_offset_config(window.symbol, best_params)
    config_path = settings.output_dir / "best_config.json"
    summary_path = settings.output_dir / "best_summary.json"
    metrics = {
        "train": train_result.to_dict(),
        "validation": validation_result.to_dict(),
        "all": all_result.to_dict(),
    }
    benchmark = {
        "name": "buy_and_hold",
        "train": _benchmark_comparison(train_result, train_benchmark),
        "validation": _benchmark_comparison(validation_result, validation_benchmark),
        "all": _benchmark_comparison(all_result, all_benchmark),
    }
    hyperparameters = dict(best_trial.params)

    write_json(config_path, config)
    write_json(
        summary_path,
        {
            "best_value": best_trial.value,
            "best_trial_number": best_trial.number,
            "symbol": window.symbol,
            "bar_size": window.bar_size,
            "what_to_show": window.what_to_show,
            "use_rth": window.use_rth,
            "data_source": window.data_source,
            "first_timestamp": window.first_timestamp,
            "last_timestamp": window.last_timestamp,
            "bars": len(window.bars),
            "hyperparameters": hyperparameters,
            "resolved_strategy_params": _strategy_params_dict(best_params),
            "metrics": metrics,
            "benchmark": benchmark,
        },
    )
    with postgres_connection(settings.pg_settings) as conn:
        run_id = insert_optimizer_run(
            conn,
            study_name=settings.study_name,
            run_kind="optimize",
            symbol=window.symbol,
            output_dir=settings.output_dir,
            config_path=config_path,
            summary_path=summary_path,
            best_value=float(best_trial.value or 0.0),
            data_source=window.data_source,
            bar_size=window.bar_size,
            what_to_show=window.what_to_show,
            use_rth=window.use_rth,
            first_timestamp=window.first_timestamp,
            last_timestamp=window.last_timestamp,
            bars=len(window.bars),
            metrics=metrics,
            hyperparameters=hyperparameters,
        )
        insert_optimizer_trials(conn, run_id, study.trials)
        insert_optimizer_fills(conn, run_id, fills)

    if settings.verbose:
        print("Best trial")
        print(f"  number: {best_trial.number}")
        print(f"  value: {best_trial.value:.8f}")
        print(f"  train_return_pct: {train_result.return_pct:.6f}")
        print(f"  validation_return_pct: {validation_result.return_pct:.6f}")
        print(f"  buy_hold_return_pct: {all_benchmark.return_pct:.6f}")
        print(
            "  excess_return_pct: "
            f"{all_result.return_pct - all_benchmark.return_pct:.6f}"
        )
        print(f"  all_net_pnl: {all_result.net_pnl:.2f}")
        print(f"  all_fills: {all_result.fills}")
        print(f"  config: {config_path}")
        print(f"  summary: {summary_path}")
        print("  pg_tables: optimizer_runs, optimizer_trials, optimizer_fills")

    return OptimizationArtifacts(
        output_dir=settings.output_dir,
        config_path=config_path,
        summary_path=summary_path,
        study_storage=settings.storage_url,
        best_value=float(best_trial.value or 0.0),
    )


class _Objective:
    def __init__(
        self,
        train_bars: list[Bar],
        validation_bars: list[Bar],
        min_trades: int,
    ) -> None:
        self.train_bars = train_bars
        self.validation_bars = validation_bars
        self.min_trades = min_trades

    def __call__(self, trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, self.train_bars)
        train_result, _ = simulate_constant_step_offset(self.train_bars, params)
        validation_result, _ = simulate_constant_step_offset(
            self.validation_bars,
            params,
        )
        train_benchmark = simulate_buy_and_hold(
            self.train_bars,
            allocated_capital=train_result.allocated_capital,
        )
        validation_benchmark = simulate_buy_and_hold(
            self.validation_bars,
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
        trial.set_user_attr("final_position_value", validation_result.final_position_value)
        trial.set_user_attr("max_drawdown", validation_result.max_drawdown)

        no_trade_penalty = 0.0
        total_fills = train_result.fills + validation_result.fills
        if total_fills < self.min_trades:
            no_trade_penalty = (self.min_trades - total_fills) * 0.01

        inventory_penalty = (
            abs(validation_result.final_position_value)
            / max(validation_result.allocated_capital, 1.0)
        ) * 0.25
        drawdown_penalty = (
            validation_result.max_drawdown
            / max(validation_result.allocated_capital, 1.0)
        ) * 0.10

        return (
            train_excess_return * 0.70
            + validation_excess_return * 0.30
            - no_trade_penalty
            - inventory_penalty
            - drawdown_penalty
        )


def _params_from_trial(trial: optuna.Trial, train_bars: list[Bar]) -> StrategyParams:
    closes = [bar.close for bar in train_bars]
    baseline_quantile = trial.suggest_float("baseline_quantile", 0.10, 0.90)
    baseline = _quantile(closes, baseline_quantile)
    step_delta_pct = trial.suggest_float("step_delta_pct", 0.001, 0.03, log=True)
    step_delta = max(0.01, baseline * step_delta_pct)
    execution_steps = trial.suggest_int("execution_steps", 3, 30)
    execution_limit_offset = step_delta * (execution_steps + 1)
    threshold_pct = trial.suggest_float("threshold_pct_of_step", 0.0, 0.90)
    state_transition_threshold = step_delta * threshold_pct
    order_quantity_usd = trial.suggest_float(
        "order_quantity_usd",
        100.0,
        5000.0,
        log=True,
    )
    order_quantity = max(1.0, round(order_quantity_usd / baseline))

    return StrategyParams(
        baseline=baseline,
        step_delta=step_delta,
        execution_limit_offset=execution_limit_offset,
        state_transition_threshold=state_transition_threshold,
        order_quantity_usd=order_quantity_usd,
        order_quantity=order_quantity,
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _strategy_params_dict(params: StrategyParams) -> dict[str, float | int]:
    return {
        "baseline": params.baseline,
        "stepDelta": params.step_delta,
        "executionLimitOffset": params.execution_limit_offset,
        "stateTransitionThreshold": params.state_transition_threshold,
        "orderQuantityInUSD": params.order_quantity_usd,
        "orderQuantity": params.order_quantity,
        "maxSteps": params.max_steps,
    }


def _benchmark_comparison(
    strategy_result: SimulationResult,
    benchmark_result: Any,
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
