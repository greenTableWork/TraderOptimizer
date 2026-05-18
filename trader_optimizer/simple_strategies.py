from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite, log, sqrt
from statistics import mean, pstdev
from typing import Callable

from trader_optimizer.data import Bar
from trader_optimizer.simulator import stock_commission


@dataclass(frozen=True)
class SimpleResult:
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

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def simulate_target_directions(
    bars: list[Bar],
    directions: list[int],
    quantity: float,
) -> SimpleResult:
    if len(bars) != len(directions):
        raise ValueError("bars and directions must have the same length")
    if not bars:
        raise ValueError("bars cannot be empty")

    cash = 0.0
    position = 0.0
    commissions = 0.0
    fills = 0
    equity_curve: list[float] = []
    for bar, direction in zip(bars, directions):
        price = bar.close
        target_position = float(direction) * quantity
        delta = target_position - position
        if abs(delta) > 1e-9:
            fee = stock_commission(abs(delta), price)
            cash -= delta * price
            cash -= fee
            commissions += fee
            position = target_position
            fills += 1
        equity_curve.append(cash + position * price)

    final_position_value = position * bars[-1].close
    ending_equity = cash + final_position_value
    allocated_capital = max(quantity * bars[0].close, 1.0)
    return SimpleResult(
        bars=len(bars),
        first_timestamp=bars[0].timestamp_utc,
        last_timestamp=bars[-1].timestamp_utc,
        net_pnl=ending_equity,
        return_pct=ending_equity / allocated_capital,
        max_drawdown=_max_drawdown(equity_curve),
        fills=fills,
        commissions=commissions,
        final_position_value=final_position_value,
        ending_equity=ending_equity,
        allocated_capital=allocated_capital,
    )


def sma_cross_directions(
    bars: list[Bar],
    fast_window: int,
    slow_window: int,
) -> list[int]:
    closes = [bar.close for bar in bars]
    directions: list[int] = []
    for index in range(len(closes)):
        if index + 1 < slow_window or slow_window <= fast_window:
            directions.append(0)
            continue
        fast = mean(closes[index + 1 - fast_window : index + 1])
        slow = mean(closes[index + 1 - slow_window : index + 1])
        directions.append(1 if fast > slow else 0)
    return directions


def ema_cross_directions(
    bars: list[Bar],
    fast_window: int,
    slow_window: int,
) -> list[int]:
    closes = [bar.close for bar in bars]
    fast_ema = _ema_series(closes, fast_window)
    slow_ema = _ema_series(closes, slow_window)
    directions = []
    for fast, slow in zip(fast_ema, slow_ema):
        if fast is None or slow is None or slow_window <= fast_window:
            directions.append(0)
        else:
            directions.append(1 if fast > slow else 0)
    return directions


def bollinger_breakout_directions(
    bars: list[Bar],
    middle_window: int,
    trend_window: int,
    band_stddev: float,
) -> list[int]:
    closes = [bar.close for bar in bars]
    target = 0
    directions: list[int] = []
    for index, close in enumerate(closes):
        required = max(middle_window, trend_window)
        if index + 1 < required or middle_window <= 1:
            directions.append(target)
            continue
        middle_values = closes[index + 1 - middle_window : index + 1]
        trend_values = closes[index + 1 - trend_window : index + 1]
        middle = mean(middle_values)
        stddev = pstdev(middle_values)
        trend = mean(trend_values)
        trend_up = middle > trend
        if close > middle + band_stddev * stddev and trend_up:
            target = 1
        elif close < middle or not trend_up:
            target = 0
        directions.append(target)
    return directions


def rsi_divergence_directions(
    bars: list[Bar],
    rsi_period: int,
    divergence_lookback: int,
) -> list[int]:
    closes = [bar.close for bar in bars]
    rsi_values = _rsi_series(closes, rsi_period)
    target = 0
    directions: list[int] = []
    for index, close in enumerate(closes):
        rsi = rsi_values[index]
        if (
            rsi is None
            or index <= divergence_lookback + 1
            or divergence_lookback < 2
        ):
            directions.append(target)
            continue
        previous_prices = closes[index - divergence_lookback - 1 : index]
        previous_rs = [
            value
            for value in rsi_values[index - divergence_lookback - 1 : index]
            if value is not None
        ]
        if not previous_rs:
            directions.append(target)
            continue
        if close <= min(previous_prices) and rsi > min(previous_rs) and rsi < 50.0:
            target = 1
        elif close >= max(previous_prices) and rsi < max(previous_rs) and rsi > 50.0:
            target = -1
        elif target > 0 and rsi >= 55.0:
            target = 0
        elif target < 0 and rsi <= 45.0:
            target = 0
        directions.append(target)
    return directions


def opening_range_breakout_directions(
    bars: list[Bar],
    opening_range_bars: int,
) -> list[int]:
    current_session = ""
    session_count = 0
    opening_high = 0.0
    opening_low = 0.0
    target = 0
    directions: list[int] = []
    for bar in bars:
        session = bar.timestamp_utc[:10]
        if session != current_session:
            current_session = session
            session_count = 0
            opening_high = bar.high
            opening_low = bar.low
            target = 0
            directions.append(target)
            continue
        if session_count < opening_range_bars:
            opening_high = max(opening_high, bar.high)
            opening_low = min(opening_low, bar.low)
            session_count += 1
            directions.append(target)
            continue
        if target > 0 and bar.close < opening_low:
            target = 0
        elif target == 0 and bar.close > opening_high:
            target = 1
        directions.append(target)
    return directions


def simulate_portfolio(
    symbol_bars: dict[str, list[Bar]],
    weight_function: Callable[[dict[str, list[float]], dict[str, float]], dict[str, float]],
    portfolio_notional: float,
    max_gross_exposure: float,
    min_trade_quantity: float = 0.0001,
) -> SimpleResult:
    rows = _aligned_rows(symbol_bars)
    if not rows:
        raise ValueError("No aligned multi-symbol rows")
    cash = 0.0
    positions = defaultdict(float)
    histories = defaultdict(list)
    commissions = 0.0
    fills = 0
    equity_curve: list[float] = []

    for timestamp, prices in rows:
        for symbol, price in prices.items():
            histories[symbol].append(price)
        weights = _normalize_gross(
            weight_function(histories, prices),
            max_gross_exposure,
        )
        for symbol, price in prices.items():
            target_weight = weights.get(symbol, 0.0)
            target_quantity = target_weight * portfolio_notional / price
            delta = target_quantity - positions[symbol]
            if abs(delta) < min_trade_quantity:
                continue
            fee = stock_commission(abs(delta), price)
            cash -= delta * price
            cash -= fee
            commissions += fee
            positions[symbol] = target_quantity
            fills += 1
        equity_curve.append(cash + sum(positions[s] * prices[s] for s in prices))

    _, final_prices = rows[-1]
    final_position_value = sum(
        positions[symbol] * final_prices.get(symbol, 0.0)
        for symbol in positions
    )
    ending_equity = cash + final_position_value
    return SimpleResult(
        bars=len(rows),
        first_timestamp=rows[0][0],
        last_timestamp=rows[-1][0],
        net_pnl=ending_equity,
        return_pct=ending_equity / max(portfolio_notional, 1.0),
        max_drawdown=_max_drawdown(equity_curve),
        fills=fills,
        commissions=commissions,
        final_position_value=final_position_value,
        ending_equity=ending_equity,
        allocated_capital=portfolio_notional,
    )


def volatility_target_weights(
    volatility_window: int,
    target_volatility: float,
) -> Callable[[dict[str, list[float]], dict[str, float]], dict[str, float]]:
    basket_returns: deque[float] = deque()
    previous_basket = 0.0

    def weights(histories: dict[str, list[float]], prices: dict[str, float]) -> dict[str, float]:
        nonlocal previous_basket
        basket = mean(prices.values())
        if previous_basket > 0.0:
            basket_returns.append(basket / previous_basket - 1.0)
            while len(basket_returns) > volatility_window:
                basket_returns.popleft()
        previous_basket = basket
        exposure = 0.50
        if len(basket_returns) >= 2:
            realized = pstdev(basket_returns) * sqrt(13.0 * 252.0)
            if isfinite(realized) and realized > 0:
                exposure = min(1.0, max(0.25, target_volatility / realized))
        per_symbol = exposure / len(prices)
        return {symbol: per_symbol for symbol in prices}

    return weights


def momentum_factor_weights(
    lookback: int,
    leg_size: int,
) -> Callable[[dict[str, list[float]], dict[str, float]], dict[str, float]]:
    def weights(histories: dict[str, list[float]], prices: dict[str, float]) -> dict[str, float]:
        scores = []
        for symbol, history in histories.items():
            if len(history) <= lookback or history[-lookback - 1] <= 0:
                continue
            scores.append((history[-1] / history[-lookback - 1] - 1.0, symbol))
        if len(scores) < 2:
            return {}
        scores.sort()
        size = min(leg_size, len(scores) // 2)
        if size <= 0:
            return {}
        output = {}
        for index in range(size):
            output[scores[index][1]] = -0.50 / size
            output[scores[-1 - index][1]] = 0.50 / size
        return output

    return weights


def pairs_trading_weights(
    pairs: list[tuple[str, str]],
    pair_window: int,
    entry_z: float,
    exit_z: float,
) -> Callable[[dict[str, list[float]], dict[str, float]], dict[str, float]]:
    pair_state = {pair: 0 for pair in pairs}

    def weights(histories: dict[str, list[float]], prices: dict[str, float]) -> dict[str, float]:
        output = defaultdict(float)
        if not pairs:
            return {}
        pair_budget = 1.0 / len(pairs)
        for left, right in pairs:
            left_history = histories[left]
            right_history = histories[right]
            count = min(pair_window, len(left_history), len(right_history))
            if count < 4:
                continue
            spreads = [
                log(left_history[-count + idx]) - log(right_history[-count + idx])
                for idx in range(count)
                if left_history[-count + idx] > 0 and right_history[-count + idx] > 0
            ]
            if len(spreads) < 4:
                continue
            spread_std = pstdev(spreads)
            if spread_std == 0:
                continue
            zscore = (spreads[-1] - mean(spreads)) / spread_std
            state = pair_state[(left, right)]
            if abs(zscore) < exit_z:
                state = 0
            elif zscore > entry_z:
                state = -1
            elif zscore < -entry_z:
                state = 1
            pair_state[(left, right)] = state
            if state:
                output[left] += state * pair_budget / 2.0
                output[right] -= state * pair_budget / 2.0
        return dict(output)

    return weights


def _ema_series(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        return [None] * len(values)
    output: list[float | None] = []
    seed_sum = 0.0
    ema_value = 0.0
    ready = False
    smoothing = 2.0 / (window + 1.0)
    for index, value in enumerate(values, start=1):
        if index <= window:
            seed_sum += value
            if index == window:
                ema_value = seed_sum / window
                ready = True
                output.append(ema_value)
            else:
                output.append(None)
            continue
        ema_value += smoothing * (value - ema_value)
        ready = True
        output.append(ema_value if ready else None)
    return output


def _rsi_series(values: list[float], period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period + 1:
        return output
    gains = []
    losses = []
    for index in range(1, period + 1):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    output[period] = _rsi(avg_gain, avg_loss)
    for index in range(period + 1, len(values)):
        change = values[index] - values[index - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        output[index] = _rsi(avg_gain, avg_loss)
    return output


def _rsi(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _aligned_rows(symbol_bars: dict[str, list[Bar]]) -> list[tuple[str, dict[str, float]]]:
    timestamp_prices: dict[str, dict[str, float]] = defaultdict(dict)
    symbols = set(symbol_bars)
    for symbol, bars in symbol_bars.items():
        for bar in bars:
            timestamp_prices[bar.timestamp_utc][symbol] = bar.close
    rows = []
    for timestamp in sorted(timestamp_prices):
        prices = timestamp_prices[timestamp]
        if set(prices) == symbols:
            rows.append((timestamp, prices))
    return rows


def _normalize_gross(weights: dict[str, float], max_gross: float) -> dict[str, float]:
    gross = sum(abs(value) for value in weights.values())
    if gross == 0.0 or gross <= max_gross:
        return dict(weights)
    scale = max_gross / gross
    return {symbol: value * scale for symbol, value in weights.items()}


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst
