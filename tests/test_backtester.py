from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from trader_optimizer.backtester import (
    BackTesterRunResult,
    BackTesterSettings,
    BacktestProfile,
    run_backtester,
    validate_with_backtester,
)
from trader_optimizer.data import Bar, BarWindow, DataProfile
from trader_optimizer.postgres import PostgresSettings


def _bar(timestamp: str, close: float) -> Bar:
    return Bar(timestamp, close, close, close, close)


def test_run_backtester_writes_config_and_reads_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trader_root = tmp_path / "Trader"
    trader_core = trader_root / "TraderCore"
    trader_core.mkdir(parents=True)
    backtester = tmp_path / "BackTester"
    backtester.write_text("#!/bin/sh\n", encoding="utf-8")
    backtester.chmod(0o700)
    strategy_config = tmp_path / "best_config.json"
    strategy_config.write_text(
        json.dumps({"contract": {"symbol": "AAPL", "exchange": "IBKR"}}),
        encoding="utf-8",
    )

    def fake_run(*args, **kwargs):
        config_path = Path(args[0][2])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        output_dir = Path(config["reporting"]["outputDirectory"])
        output_dir.mkdir(parents=True)
        summary_path = output_dir / f"{config['reporting']['runId']}_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "startingCash": 100000,
                    "endingEquity": 101500,
                    "endingCash": 101500,
                    "positions": {},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args[0], 0, "ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_backtester(
        strategy_config_path=strategy_config,
        strategy_name="AAPL test",
        symbols=("AAPL",),
        profile=BacktestProfile(
            bar_size="10 secs",
            what_to_show="TRADES",
            use_rth=1,
            start_utc="2026-05-15T13:30:00+00:00",
            end_utc="2026-05-15T13:31:00+00:00",
        ),
        settings=BackTesterSettings(
            trader_root=trader_root,
            pg_settings=PostgresSettings(database="trader"),
            backtester=backtester,
            skip_build=True,
        ),
        output_dir=tmp_path / "out",
    )

    run_config = json.loads(Path(result.run_config_path).read_text(encoding="utf-8"))
    assert run_config["dataSource"]["conninfo"] == "host=127.0.0.1 port=5432 dbname=trader"
    assert run_config["dataSource"]["symbols"] == ["AAPL"]
    rewritten_config = json.loads(Path(result.strategy_config_path).read_text(encoding="utf-8"))
    original_config = json.loads(strategy_config.read_text(encoding="utf-8"))
    assert rewritten_config["contract"]["exchange"] == "BACKTESTER"
    assert original_config["contract"]["exchange"] == "IBKR"
    assert result.return_pct == pytest.approx(0.015)


def test_validate_with_backtester_requires_positive_spx_and_same_stock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bars = {
        "AAPL": [
            _bar("2026-05-15T13:30:00+00:00", 100.0),
            _bar("2026-05-15T13:31:00+00:00", 105.0),
        ]
    }

    def fake_run_backtester(**kwargs):
        return BackTesterRunResult(
            bars=0,
            first_timestamp="2026-05-15T13:30:00+00:00",
            last_timestamp="2026-05-15T13:31:00+00:00",
            net_pnl=7000.0,
            return_pct=0.07,
            max_drawdown=0.0,
            fills=0,
            commissions=0.0,
            final_position_value=0.0,
            ending_equity=107000.0,
            allocated_capital=100000.0,
            summary={},
            summary_path="summary.json",
            run_config_path="run.json",
            strategy_config_path="config.json",
        )

    def fake_load_bars(**kwargs):
        return BarWindow(
            symbol="SPX",
            bar_size="10 secs",
            what_to_show="TRADES",
            use_rth=1,
            data_source="test",
            bars=[
                _bar("2026-05-15T13:30:00+00:00", 1000.0),
                _bar("2026-05-15T13:31:00+00:00", 1020.0),
            ],
        )

    monkeypatch.setattr("trader_optimizer.backtester.run_backtester", fake_run_backtester)
    monkeypatch.setattr("trader_optimizer.backtester.load_bars", fake_load_bars)
    result = validate_with_backtester(
        strategy_config_path=tmp_path / "best_config.json",
        strategy_name="AAPL",
        symbols=("AAPL",),
        profile=BacktestProfile("10 secs", "TRADES", 1, bars["AAPL"][0].timestamp_utc, bars["AAPL"][-1].timestamp_utc),
        symbol_bars=bars,
        settings=BackTesterSettings(
            trader_root=tmp_path,
            pg_settings=PostgresSettings(),
            backtester=tmp_path / "BackTester",
            skip_build=True,
        ),
        output_dir=tmp_path / "out",
    )

    assert result.status == "ok"
    assert result.strategy_return_pct > result.same_stock_return_pct
    assert result.strategy_return_pct > result.spx_return_pct


def test_validate_with_backtester_falls_back_to_available_spx_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bars = {
        "BTC": [
            _bar("2026-05-26T13:30:00+00:00", 100_000.0),
            _bar("2026-05-26T13:31:00+00:00", 101_000.0),
        ]
    }
    load_calls: list[dict[str, object]] = []

    def fake_run_backtester(**kwargs):
        return BackTesterRunResult(
            bars=0,
            first_timestamp="2026-05-26T13:30:00+00:00",
            last_timestamp="2026-05-26T13:31:00+00:00",
            net_pnl=3000.0,
            return_pct=0.03,
            max_drawdown=0.0,
            fills=0,
            commissions=0.0,
            final_position_value=0.0,
            ending_equity=103000.0,
            allocated_capital=100000.0,
            summary={},
            summary_path="summary.json",
            run_config_path="run.json",
            strategy_config_path="config.json",
        )

    def fake_load_bars(**kwargs):
        load_calls.append(kwargs)
        if kwargs["bar_size"] == "1 min":
            raise ValueError("No bars matched exact crypto benchmark profile")
        return BarWindow(
            symbol="SPX",
            bar_size="10 secs",
            what_to_show="TRADES",
            use_rth=1,
            data_source="test",
            bars=[
                _bar("2026-05-26T13:30:00+00:00", 1000.0),
                _bar("2026-05-26T13:31:00+00:00", 1010.0),
            ],
        )

    def fake_choose_data_profile(pg_settings, symbol, preferred_bar_size=None):
        assert symbol == "SPX"
        assert preferred_bar_size == "1 min"
        return DataProfile(
            symbol="SPX",
            bar_size="10 secs",
            what_to_show="TRADES",
            use_rth=1,
            count=2,
            first_timestamp="2026-05-26T13:30:00+00:00",
            last_timestamp="2026-05-26T13:31:00+00:00",
        )

    monkeypatch.setattr("trader_optimizer.backtester.run_backtester", fake_run_backtester)
    monkeypatch.setattr("trader_optimizer.backtester.load_bars", fake_load_bars)
    monkeypatch.setattr(
        "trader_optimizer.backtester.choose_data_profile",
        fake_choose_data_profile,
    )

    result = validate_with_backtester(
        strategy_config_path=tmp_path / "best_config.json",
        strategy_name="BTC",
        symbols=("BTC",),
        profile=BacktestProfile(
            "1 min",
            "AGGTRADES",
            0,
            bars["BTC"][0].timestamp_utc,
            bars["BTC"][-1].timestamp_utc,
        ),
        symbol_bars=bars,
        settings=BackTesterSettings(
            trader_root=tmp_path,
            pg_settings=PostgresSettings(),
            backtester=tmp_path / "BackTester",
            skip_build=True,
        ),
        output_dir=tmp_path / "out",
    )

    assert result.status == "ok"
    assert [call["bar_size"] for call in load_calls] == ["1 min", "10 secs"]
