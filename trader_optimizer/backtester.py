from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from trader_optimizer.config import write_json
from trader_optimizer.data import Bar, load_bars
from trader_optimizer.postgres import PostgresSettings
from trader_optimizer.simple_strategies import (
    SimpleResult,
    simulate_buy_and_hold,
    simulate_equal_weight_buy_and_hold,
)


@dataclass(frozen=True)
class BackTesterSettings:
    trader_root: Path
    pg_settings: PostgresSettings
    preset: str = "debug"
    backtester: Path | None = None
    skip_build: bool = False
    timeout_seconds: int = 300
    benchmark_symbol: str = "SPX"
    starting_cash: float = 100000.0

    @property
    def trader_core_root(self) -> Path:
        return self.trader_root / "TraderCore"


@dataclass(frozen=True)
class BacktestProfile:
    bar_size: str
    what_to_show: str
    use_rth: int
    start_utc: str
    end_utc: str


@dataclass(frozen=True)
class BackTesterRunResult:
    bars: int
    first_timestamp: str
    last_timestamp: str
    net_pnl: float
    return_pct: float
    max_drawdown: float
    fills: int
    commissions: float
    final_position_value: float
    ending_equity: float
    allocated_capital: float
    summary: dict[str, Any]
    summary_path: str
    run_config_path: str
    strategy_config_path: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class BacktestValidationResult:
    status: str
    reason: str | None
    strategy: BackTesterRunResult
    same_stock_benchmark: SimpleResult
    spx_benchmark: SimpleResult

    @property
    def strategy_return_pct(self) -> float:
        return self.strategy.return_pct

    @property
    def same_stock_return_pct(self) -> float:
        return self.same_stock_benchmark.return_pct

    @property
    def spx_return_pct(self) -> float:
        return self.spx_benchmark.return_pct

    @property
    def same_stock_excess_return_pct(self) -> float:
        return self.strategy_return_pct - self.same_stock_return_pct

    @property
    def spx_excess_return_pct(self) -> float:
        return self.strategy_return_pct - self.spx_return_pct

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "strategy": self.strategy.to_dict(),
            "benchmarks": {
                "positive_return": {
                    "passed": self.strategy_return_pct > 0.0,
                    "strategy_return_pct": self.strategy_return_pct,
                },
                "same_stock_buy_and_hold": {
                    "passed": self.same_stock_excess_return_pct > 0.0,
                    "benchmark": self.same_stock_benchmark.to_dict(),
                    "benchmark_return_pct": self.same_stock_return_pct,
                    "excess_return_pct": self.same_stock_excess_return_pct,
                },
                "spx_buy_and_hold": {
                    "passed": self.spx_excess_return_pct > 0.0,
                    "benchmark": self.spx_benchmark.to_dict(),
                    "benchmark_return_pct": self.spx_return_pct,
                    "excess_return_pct": self.spx_excess_return_pct,
                },
            },
        }


def prepare_backtester(settings: BackTesterSettings) -> BackTesterSettings:
    trader_core_root = settings.trader_core_root.resolve()
    if settings.backtester is not None:
        return replace(settings, backtester=settings.backtester.resolve())

    backtester = trader_core_root / "build" / settings.preset / "BackTesting" / "BackTester"
    if not settings.skip_build:
        subprocess.run(
            ["cmake", "--build", "--preset", settings.preset, "--target", "BackTester"],
            cwd=trader_core_root,
            check=True,
        )
    return replace(settings, backtester=backtester.resolve(), skip_build=True)


def validate_with_backtester(
    *,
    strategy_config_path: Path,
    strategy_name: str,
    symbols: tuple[str, ...],
    profile: BacktestProfile,
    symbol_bars: dict[str, list[Bar]],
    settings: BackTesterSettings,
    output_dir: Path,
) -> BacktestValidationResult:
    prepared = prepare_backtester(settings)
    strategy_result = run_backtester(
        strategy_config_path=strategy_config_path,
        strategy_name=strategy_name,
        symbols=symbols,
        profile=profile,
        settings=prepared,
        output_dir=output_dir,
    )
    same_stock_benchmark = _same_stock_benchmark(symbol_bars, settings.starting_cash)
    spx_bars = load_bars(
        pg_settings=settings.pg_settings,
        symbol=settings.benchmark_symbol,
        bar_size=profile.bar_size,
        what_to_show=profile.what_to_show,
        use_rth=profile.use_rth,
        start_utc=profile.start_utc,
        end_utc=profile.end_utc,
        max_bars=0,
    ).bars
    spx_benchmark = simulate_buy_and_hold(
        spx_bars,
        allocated_capital=settings.starting_cash,
    )
    status, reason = _benchmark_status(
        strategy_result.return_pct,
        same_stock_benchmark.return_pct,
        spx_benchmark.return_pct,
    )
    return BacktestValidationResult(
        status=status,
        reason=reason,
        strategy=strategy_result,
        same_stock_benchmark=same_stock_benchmark,
        spx_benchmark=spx_benchmark,
    )


def run_backtester(
    *,
    strategy_config_path: Path,
    strategy_name: str,
    symbols: tuple[str, ...],
    profile: BacktestProfile,
    settings: BackTesterSettings,
    output_dir: Path,
) -> BackTesterRunResult:
    backtester = settings.backtester or (
        settings.trader_core_root / "build" / settings.preset / "BackTesting" / "BackTester"
    )
    if not backtester.is_file() or not os.access(backtester, os.X_OK):
        raise FileNotFoundError(f"BackTester binary not found or not executable: {backtester}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_name(strategy_name)
    run_id = f"{safe_name}_backtester"
    run_config_path = output_dir / "backtest_config.json"
    run_output_dir = output_dir / "outputs"
    summary_path = run_output_dir / f"{run_id}_summary.json"
    write_json(
        run_config_path,
        _backtest_config(
            strategy_config_path=strategy_config_path,
            symbols=symbols,
            profile=profile,
            settings=settings,
            run_name=safe_name,
            run_id=run_id,
            output_dir=run_output_dir,
        ),
    )

    command = [str(backtester), "--backtest-config", str(run_config_path)]
    env = os.environ.copy()
    env["OPENTRADER_ROOT"] = str(settings.trader_core_root)
    completed = subprocess.run(
        command,
        cwd=settings.trader_core_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=settings.timeout_seconds,
        check=False,
    )
    log_path = output_dir / "backtester.log"
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n" + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"BackTester failed for {strategy_name} with exit code "
            f"{completed.returncode}; log={log_path}"
        )
    if not summary_path.is_file():
        raise FileNotFoundError(f"BackTester summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    starting_cash = _float(summary.get("startingCash"), settings.starting_cash)
    ending_equity = _float(summary.get("endingEquity"), starting_cash)
    net_pnl = ending_equity - starting_cash
    return BackTesterRunResult(
        bars=0,
        first_timestamp=profile.start_utc,
        last_timestamp=profile.end_utc,
        net_pnl=net_pnl,
        return_pct=net_pnl / max(starting_cash, 1.0),
        max_drawdown=0.0,
        fills=0,
        commissions=0.0,
        final_position_value=0.0,
        ending_equity=ending_equity,
        allocated_capital=starting_cash,
        summary=summary,
        summary_path=str(summary_path),
        run_config_path=str(run_config_path),
        strategy_config_path=str(strategy_config_path),
    )


def profile_for_bars(
    *,
    bar_size: str,
    what_to_show: str,
    use_rth: int,
    symbol_bars: dict[str, list[Bar]],
) -> BacktestProfile:
    if not symbol_bars:
        raise ValueError("symbol_bars cannot be empty")
    first = max(bars[0].timestamp_utc for bars in symbol_bars.values() if bars)
    last = min(bars[-1].timestamp_utc for bars in symbol_bars.values() if bars)
    if first > last:
        raise ValueError(f"No overlapping backtest window: {first} > {last}")
    return BacktestProfile(
        bar_size=bar_size,
        what_to_show=what_to_show,
        use_rth=use_rth,
        start_utc=first,
        end_utc=last,
    )


def _backtest_config(
    *,
    strategy_config_path: Path,
    symbols: tuple[str, ...],
    profile: BacktestProfile,
    settings: BackTesterSettings,
    run_name: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "startingCash": settings.starting_cash,
        "cashCurrency": "USD",
        "account": {
            "cashReserve": 0,
            "leverage": 1,
            "marginRequirement": 1,
            "allowShorting": False,
            "fractionalTrading": True,
            "lotSize": 0,
            "tickSize": 0,
        },
        "startUtc": profile.start_utc,
        "endUtc": profile.end_utc,
        "timezone": "UTC",
        "strategyConfigPath": str(strategy_config_path.resolve()),
        "dataSource": {
            "type": "postgres",
            "conninfo": _postgres_conninfo(settings.pg_settings),
            "barSize": profile.bar_size,
            "whatToShow": profile.what_to_show,
            "useRth": bool(profile.use_rth),
            "symbols": list(symbols),
        },
        "brokerage": {
            "name": "IBKR",
            "feeStructurePath": "data/fee_structures/ibkr_tws_stock_fees.json",
            "slippageBps": 0,
            "spreadBps": 0,
        },
        "fillModel": {
            "limitFillMode": "reference",
            "marketOrderFillPrice": "bar_average",
            "allowSameBarFills": True,
            "partialFills": False,
            "volumeParticipationRate": 1,
        },
        "execution": {
            "warmupBars": 0,
            "orderLatencyTicks": 0,
            "orderExpiryTicks": 0,
            "cancelOpenOrdersAtSessionEnd": False,
            "cancelOpenOrdersAtEnd": True,
            "finalizeOpenPositionsAtEnd": False,
        },
        "risk": {
            "maxGrossExposure": 0,
            "maxNetExposure": 0,
            "maxPositionQuantity": 0,
            "maxOrderNotional": 0,
            "maxDailyLoss": 0,
        },
        "reporting": {
            "runName": run_name,
            "runId": run_id,
            "outputDirectory": str(output_dir),
            "benchmarkSymbol": settings.benchmark_symbol,
            "metricsSet": ["ending_cash", "ending_equity", "positions"],
            "randomSeed": 0,
        },
    }


def _same_stock_benchmark(
    symbol_bars: dict[str, list[Bar]],
    allocated_capital: float,
) -> SimpleResult:
    if len(symbol_bars) == 1:
        return simulate_buy_and_hold(
            next(iter(symbol_bars.values())),
            allocated_capital=allocated_capital,
        )
    return simulate_equal_weight_buy_and_hold(symbol_bars, allocated_capital)


def _benchmark_status(
    strategy_return: float,
    same_stock_return: float,
    spx_return: float,
) -> tuple[str, str | None]:
    if strategy_return <= 0.0:
        return (
            "benchmark_failed",
            f"BackTester return {strategy_return:.6f} was not positive",
        )
    if strategy_return <= spx_return:
        return (
            "benchmark_failed",
            f"BackTester return {strategy_return:.6f} did not beat SPX {spx_return:.6f}",
        )
    if strategy_return <= same_stock_return:
        return (
            "benchmark_failed",
            "BackTester return "
            f"{strategy_return:.6f} did not beat same-stock hold {same_stock_return:.6f}",
        )
    return "ok", None


def _postgres_conninfo(settings: PostgresSettings) -> str:
    if settings.conninfo:
        return settings.conninfo
    parts = [
        f"host={settings.host}",
        f"port={settings.port}",
        f"dbname={settings.database}",
    ]
    if settings.user:
        parts.append(f"user={settings.user}")
    if settings.password:
        parts.append(f"password={settings.password}")
    return " ".join(parts)


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._") or "backtest"


def _float(value: object, default: float) -> float:
    if value is None:
        return default
    return float(value)
