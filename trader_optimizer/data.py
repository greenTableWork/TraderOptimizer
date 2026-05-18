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


@dataclass(frozen=True)
class DataProfile:
    symbol: str
    bar_size: str
    what_to_show: str
    use_rth: int
    count: int
    first_timestamp: str
    last_timestamp: str


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


def available_profiles(db_path: Path, symbol: str | None = None) -> list[DataProfile]:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    params: list[object] = []
    filter_sql = ""
    if symbol is not None:
        filter_sql = "WHERE symbol = ?"
        params.append(symbol)

    query = f"""
        SELECT
            symbol,
            bar_size,
            what_to_show,
            use_rth,
            COUNT(*) AS row_count,
            MIN(bar_time_utc),
            MAX(bar_time_utc)
        FROM historical_bars
        {filter_sql}
        GROUP BY symbol, bar_size, what_to_show, use_rth
        ORDER BY symbol, bar_size, what_to_show, use_rth
    """
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        DataProfile(
            symbol=str(row[0]),
            bar_size=str(row[1]),
            what_to_show=str(row[2]),
            use_rth=int(row[3]),
            count=int(row[4]),
            first_timestamp=str(row[5]),
            last_timestamp=str(row[6]),
        )
        for row in rows
    ]


def choose_data_profile(
    db_path: Path,
    symbol: str,
    preferred_bar_size: str | None = None,
) -> DataProfile:
    profiles = available_profiles(db_path, symbol)
    if not profiles:
        raise ValueError(f"No SQLite bars available for {symbol}")

    preferred_sizes = [preferred_bar_size] if preferred_bar_size else []
    preferred_sizes.extend(["1 min", "10 secs", "5 mins", "1 day"])

    def score(profile: DataProfile) -> tuple[int, int, int]:
        try:
            size_rank = preferred_sizes.index(profile.bar_size)
        except ValueError:
            size_rank = len(preferred_sizes)
        what_rank = 0 if profile.what_to_show == "TRADES" else 1
        rth_rank = 0 if profile.use_rth == 1 else 1
        return (size_rank, what_rank + rth_rank, -profile.count)

    return sorted(profiles, key=score)[0]


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
