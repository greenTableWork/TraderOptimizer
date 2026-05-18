from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor

from trader_optimizer.data import Bar


@dataclass(frozen=True)
class StrategyParams:
    baseline: float
    step_delta: float
    execution_limit_offset: float
    state_transition_threshold: float
    order_quantity_usd: float
    order_quantity: float

    @property
    def max_steps(self) -> int:
        return int(self.execution_limit_offset / self.step_delta) - 1

    def validate(self) -> None:
        if self.baseline <= 0:
            raise ValueError("baseline must be positive")
        if self.step_delta <= 0:
            raise ValueError("step_delta must be positive")
        if self.execution_limit_offset < self.state_transition_threshold:
            raise ValueError(
                "execution_limit_offset must be >= state_transition_threshold"
            )
        if self.max_steps < 1:
            raise ValueError("execution_limit_offset must allow at least one step")
        if self.order_quantity_usd <= 0 or self.order_quantity <= 0:
            raise ValueError("order quantity must be positive")


@dataclass(frozen=True)
class Fill:
    tick: int
    timestamp_utc: str
    action: str
    step: int
    quantity: float
    price: float
    commission: float


@dataclass(frozen=True)
class SimulationResult:
    bars: int
    first_timestamp: str
    last_timestamp: str
    net_pnl: float
    gross_pnl: float
    commissions: float
    return_pct: float
    max_drawdown: float
    fills: int
    buys: int
    sells: int
    final_position: float
    final_position_value: float
    ending_equity: float
    allocated_capital: float
    open_buy_step: int | None
    open_sell_step: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PendingOrder:
    action: str
    step: int
    limit_price: float


def stock_commission(
    quantity: float,
    price: float,
    per_share: float = 0.005,
    minimum: float = 1.0,
    max_trade_value_multiple: float = 0.01,
) -> float:
    trade_value = abs(quantity * price)
    return min(max(quantity * per_share, minimum), trade_value * max_trade_value_multiple)


def simulate_constant_step_offset(
    bars: list[Bar],
    params: StrategyParams,
) -> tuple[SimulationResult, list[Fill]]:
    params.validate()
    if not bars:
        raise ValueError("bars cannot be empty")

    cash = 0.0
    gross_cash = 0.0
    position = 0.0
    commissions = 0.0
    fills: list[Fill] = []
    executed_buy_steps: set[int] = set()
    pending_buy: PendingOrder | None = None
    pending_sell: PendingOrder | None = None
    equity_curve: list[float] = []

    for tick, bar in enumerate(bars, start=1):
        price = bar.backtest_price

        if pending_buy and price <= pending_buy.limit_price:
            fee = stock_commission(params.order_quantity, price)
            gross_cash -= params.order_quantity * price
            cash -= params.order_quantity * price + fee
            position += params.order_quantity
            commissions += fee
            executed_buy_steps.add(pending_buy.step)
            fills.append(
                Fill(
                    tick=tick,
                    timestamp_utc=bar.timestamp_utc,
                    action="BUY",
                    step=pending_buy.step,
                    quantity=params.order_quantity,
                    price=price,
                    commission=fee,
                )
            )
            pending_buy = None

        if pending_sell and price >= pending_sell.limit_price:
            fee = stock_commission(params.order_quantity, price)
            gross_cash += params.order_quantity * price
            cash += params.order_quantity * price - fee
            position -= params.order_quantity
            commissions += fee
            executed_buy_steps.discard(pending_sell.step)
            fills.append(
                Fill(
                    tick=tick,
                    timestamp_utc=bar.timestamp_utc,
                    action="SELL",
                    step=pending_sell.step,
                    quantity=params.order_quantity,
                    price=price,
                    commission=fee,
                )
            )
            pending_sell = None

        raw_step = abs(price - params.baseline) / params.step_delta
        current_step = floor(raw_step)
        if 0 < current_step <= params.max_steps:
            buy_zone = (
                price
                <= params.baseline
                - (current_step * params.step_delta)
                + params.state_transition_threshold
            )
            sell_zone = (
                price
                >= params.baseline
                + (current_step * params.step_delta)
                - params.state_transition_threshold
            )

            if (
                buy_zone
                and pending_buy is None
                and current_step not in executed_buy_steps
            ):
                pending_buy = PendingOrder(
                    action="BUY",
                    step=current_step,
                    limit_price=params.baseline - current_step * params.step_delta,
                )
            elif (
                sell_zone
                and pending_sell is None
                and current_step in executed_buy_steps
            ):
                pending_sell = PendingOrder(
                    action="SELL",
                    step=current_step,
                    limit_price=params.baseline + current_step * params.step_delta,
                )

        equity_curve.append(cash + position * price)

    last_price = bars[-1].backtest_price
    final_position_value = position * last_price
    ending_equity = cash + final_position_value
    gross_pnl = gross_cash + final_position_value
    allocated_capital = (
        max(params.order_quantity_usd, params.order_quantity * params.baseline)
        * params.max_steps
    )
    return_pct = ending_equity / allocated_capital if allocated_capital else 0.0
    max_drawdown = _max_drawdown(equity_curve)
    buys = sum(1 for fill in fills if fill.action == "BUY")
    sells = sum(1 for fill in fills if fill.action == "SELL")

    result = SimulationResult(
        bars=len(bars),
        first_timestamp=bars[0].timestamp_utc,
        last_timestamp=bars[-1].timestamp_utc,
        net_pnl=ending_equity,
        gross_pnl=gross_pnl,
        commissions=commissions,
        return_pct=return_pct,
        max_drawdown=max_drawdown,
        fills=len(fills),
        buys=buys,
        sells=sells,
        final_position=position,
        final_position_value=final_position_value,
        ending_equity=ending_equity,
        allocated_capital=allocated_capital,
        open_buy_step=pending_buy.step if pending_buy else None,
        open_sell_step=pending_sell.step if pending_sell else None,
    )
    return result, fills


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = peak - equity
        worst = max(worst, drawdown)
    return worst
