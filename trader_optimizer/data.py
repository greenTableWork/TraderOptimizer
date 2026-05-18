from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Bar:
    timestamp_utc: str
    open: float
    high: float
    low: float
    close: float

    @property
    def backtest_price(self) -> float:
        return mean((self.open, self.high, self.low, self.close))


@dataclass(frozen=True)
class BarWindow:
    symbol: str
    bar_size: str
    what_to_show: str
    use_rth: int
    db_path: Path
    bars: list[Bar]

    @property
    def first_timestamp(self) -> str:
        return self.bars[0].timestamp_utc

    @property
    def last_timestamp(self) -> str:
        return self.bars[-1].timestamp_utc

    @property
    def closes(self) -> list[float]:
        return [bar.close for bar in self.bars]


def find_trader_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "TraderLab").is_dir() and (candidate / "TraderCore").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find Trader root. Pass --trader-root explicitly."
    )


def default_market_db(trader_root: Path) -> Path:
    return trader_root / "TraderLab" / "Data" / "tws_historical.sqlite"


def load_bars(
    db_path: Path,
    symbol: str,
    bar_size: str,
    what_to_show: str,
    use_rth: int,
    start_utc: str | None = None,
    end_utc: str | None = None,
    max_bars: int = 50000,
) -> BarWindow:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    filters = [
        "symbol = ?",
        "bar_size = ?",
        "what_to_show = ?",
        "use_rth = ?",
    ]
    params: list[object] = [symbol, bar_size, what_to_show, use_rth]
    if start_utc:
        filters.append("bar_time_utc >= ?")
        params.append(start_utc)
    if end_utc:
        filters.append("bar_time_utc <= ?")
        params.append(end_utc)

    where_clause = " AND ".join(filters)
    if max_bars > 0:
        query = f"""
            SELECT bar_time_utc, open, high, low, close
            FROM historical_bars
            WHERE {where_clause}
            ORDER BY bar_time_utc DESC
            LIMIT ?
        """
        params.append(max_bars)
    else:
        query = f"""
            SELECT bar_time_utc, open, high, low, close
            FROM historical_bars
            WHERE {where_clause}
            ORDER BY bar_time_utc ASC
        """

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    if max_bars > 0:
        rows = list(reversed(rows))

    bars = [
        Bar(
            timestamp_utc=str(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
        )
        for row in rows
    ]
    if not bars:
        raise ValueError(
            "No bars matched "
            f"symbol={symbol}, bar_size={bar_size}, what_to_show={what_to_show}, "
            f"use_rth={use_rth}, start_utc={start_utc}, end_utc={end_utc}"
        )

    return BarWindow(
        symbol=symbol,
        bar_size=bar_size,
        what_to_show=what_to_show,
        use_rth=use_rth,
        db_path=db_path,
        bars=bars,
    )


def split_train_validation(
    bars: list[Bar],
    train_fraction: float,
) -> tuple[list[Bar], list[Bar]]:
    if not 0.1 <= train_fraction <= 0.95:
        raise ValueError("train_fraction must be between 0.1 and 0.95")
    if len(bars) < 20:
        raise ValueError("Need at least 20 bars to split train/validation data")
    split_at = max(1, min(len(bars) - 1, int(len(bars) * train_fraction)))
    return bars[:split_at], bars[split_at:]
