from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product
from queue import Empty, Queue
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


@dataclass
class MarketEvent:
    type: EventType = field(default=EventType.MARKET, init=False)


@dataclass
class SignalEvent:
    symbol: str
    datetime: pd.Timestamp
    signal_type: str
    strength: float = 1.0
    type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass
class OrderEvent:
    symbol: str
    order_type: str
    quantity: int
    direction: str
    type: EventType = field(default=EventType.ORDER, init=False)


@dataclass
class FillEvent:
    timeindex: pd.Timestamp
    symbol: str
    exchange: str
    quantity: int
    direction: str
    fill_cost: float
    commission: float
    type: EventType = field(default=EventType.FILL, init=False)


class HistoricDataHandler:
    def __init__(self, events: Queue, symbol_data: Dict[str, pd.DataFrame]) -> None:
        if not symbol_data:
            raise ValueError("symbol_data cannot be empty.")

        self.events = events
        self.symbol_list = list(symbol_data.keys())
        self.symbol_data = {
            symbol: self._prepare_frame(frame) for symbol, frame in symbol_data.items()
        }
        self.latest_symbol_data: Dict[str, List[Tuple[pd.Timestamp, pd.Series]]] = {
            symbol: [] for symbol in self.symbol_list
        }
        self._symbol_iters = {
            symbol: iter(self.symbol_data[symbol].iterrows()) for symbol in self.symbol_list
        }
        self.continue_backtest = True

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        data.columns = [str(col).strip().lower() for col in data.columns]

        if not isinstance(data.index, pd.DatetimeIndex):
            datetime_col = next(
                (
                    col
                    for col in ("datetime", "date", "timestamp", "time")
                    if col in data.columns
                ),
                None,
            )
            if datetime_col is None:
                raise ValueError(
                    "Data must have a DatetimeIndex or one of: datetime/date/timestamp/time columns."
                )
            data[datetime_col] = pd.to_datetime(data[datetime_col], errors="coerce")
            data = data.dropna(subset=[datetime_col]).set_index(datetime_col)

        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        if "adj_close" in data.columns and "close" not in data.columns:
            data["close"] = data["adj_close"]
        if "close" not in data.columns:
            raise ValueError("Data must include a close (or adj_close) column.")

        for col in ("open", "high", "low"):
            if col not in data.columns:
                data[col] = data["close"]
        if "volume" not in data.columns:
            data["volume"] = 0.0

        return data[["open", "high", "low", "close", "volume"]].astype(float)

    def update_bars(self) -> None:
        for symbol in self.symbol_list:
            try:
                bar = next(self._symbol_iters[symbol])
            except StopIteration:
                self.continue_backtest = False
                return
            self.latest_symbol_data[symbol].append(bar)

        self.events.put(MarketEvent())

    def get_latest_bars(
        self, symbol: str, num_bars: int = 1
    ) -> List[Tuple[pd.Timestamp, pd.Series]]:
        if symbol not in self.latest_symbol_data:
            raise KeyError(f"Unknown symbol: {symbol}")
        if num_bars <= 0:
            return []
        return self.latest_symbol_data[symbol][-num_bars:]

    def get_latest_bar_datetime(self, symbol: str) -> pd.Timestamp:
        bars = self.get_latest_bars(symbol, 1)
        if not bars:
            raise IndexError(f"No bars available yet for symbol: {symbol}")
        return bars[-1][0]

    def get_latest_bar_value(self, symbol: str, val_type: str) -> float:
        bars = self.get_latest_bars(symbol, 1)
        if not bars:
            raise IndexError(f"No bars available yet for symbol: {symbol}")
        return float(bars[-1][1][val_type.lower()])


class Strategy:
    def calculate_signals(self, event: MarketEvent) -> None:
        raise NotImplementedError


class MovingAverageCrossStrategy(Strategy):
    def __init__(
        self,
        data_handler: HistoricDataHandler,
        events: Queue,
        short_window: int,
        long_window: int,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window.")

        self.data_handler = data_handler
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        self.market_state = {symbol: "OUT" for symbol in self.data_handler.symbol_list}

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.type is not EventType.MARKET:
            return

        for symbol in self.data_handler.symbol_list:
            bars = self.data_handler.get_latest_bars(symbol, self.long_window)
            if len(bars) < self.long_window:
                continue

            closes = pd.Series([bar[1]["close"] for bar in bars], dtype=float)
            short_ema = float(closes.ewm(span=self.short_window, min_periods=self.short_window).mean().iloc[-1])
            long_ema = float(closes.ewm(span=self.long_window, min_periods=self.long_window).mean().iloc[-1])
            dt = bars[-1][0]

            if short_ema > long_ema and self.market_state[symbol] == "OUT":
                self.events.put(
                    SignalEvent(
                        symbol=symbol,
                        datetime=dt,
                        signal_type="LONG",
                        strength=1.0,
                    )
                )
                self.market_state[symbol] = "LONG"

            elif short_ema < long_ema and self.market_state[symbol] == "LONG":
                self.events.put(
                    SignalEvent(
                        symbol=symbol,
                        datetime=dt,
                        signal_type="EXIT",
                        strength=1.0,
                    )
                )
                self.market_state[symbol] = "OUT"


class Portfolio:
    def __init__(
        self,
        data_handler: HistoricDataHandler,
        events: Queue,
        start_date: pd.Timestamp,
        initial_capital: float = 100000.0,
        order_size: int = 100,
    ) -> None:
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = self.data_handler.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.order_size = order_size

        self.current_positions = {symbol: 0 for symbol in self.symbol_list}
        self.all_positions = [
            {"datetime": self.start_date, **{symbol: 0 for symbol in self.symbol_list}}
        ]

        self.current_holdings = {
            **{symbol: 0.0 for symbol in self.symbol_list},
            "cash": self.initial_capital,
            "commission": 0.0,
            "total": self.initial_capital,
        }
        self.all_holdings = [
            {
                "datetime": self.start_date,
                **{symbol: 0.0 for symbol in self.symbol_list},
                "cash": self.initial_capital,
                "commission": 0.0,
                "total": self.initial_capital,
            }
        ]

        self.equity_curve: Optional[pd.DataFrame] = None

    def update_timeindex(self, event: MarketEvent) -> None:
        if event.type is not EventType.MARKET:
            return

        dt = self.data_handler.get_latest_bar_datetime(self.symbol_list[0])

        position_snapshot = {"datetime": dt}
        for symbol in self.symbol_list:
            position_snapshot[symbol] = self.current_positions[symbol]
        self.all_positions.append(position_snapshot)

        holdings_snapshot = {
            "datetime": dt,
            "cash": self.current_holdings["cash"],
            "commission": self.current_holdings["commission"],
            "total": self.current_holdings["cash"],
        }
        for symbol in self.symbol_list:
            market_value = self.current_positions[symbol] * self.data_handler.get_latest_bar_value(
                symbol, "close"
            )
            holdings_snapshot[symbol] = market_value
            holdings_snapshot["total"] += market_value

        self.current_holdings["total"] = holdings_snapshot["total"]
        self.all_holdings.append(holdings_snapshot)

    def generate_order_from_signal(self, signal: SignalEvent) -> Optional[OrderEvent]:
        signal_type = signal.signal_type.upper()
        symbol = signal.symbol
        current_qty = self.current_positions[symbol]

        if signal_type == "LONG" and current_qty == 0:
            return OrderEvent(
                symbol=symbol,
                order_type="MKT",
                quantity=self.order_size,
                direction="BUY",
            )

        if signal_type == "SHORT" and current_qty == 0:
            return OrderEvent(
                symbol=symbol,
                order_type="MKT",
                quantity=self.order_size,
                direction="SELL",
            )

        if signal_type == "EXIT":
            if current_qty > 0:
                return OrderEvent(
                    symbol=symbol,
                    order_type="MKT",
                    quantity=abs(current_qty),
                    direction="SELL",
                )
            if current_qty < 0:
                return OrderEvent(
                    symbol=symbol,
                    order_type="MKT",
                    quantity=abs(current_qty),
                    direction="BUY",
                )

        return None

    def update_signal(self, event: SignalEvent) -> None:
        if event.type is not EventType.SIGNAL:
            return
        order_event = self.generate_order_from_signal(event)
        if order_event is not None:
            self.events.put(order_event)

    def update_fill(self, event: FillEvent) -> None:
        if event.type is not EventType.FILL:
            return

        direction = event.direction.upper()
        trade_value = float(event.fill_cost * event.quantity)
        commission = float(event.commission)

        if direction == "BUY":
            self.current_positions[event.symbol] += event.quantity
            self.current_holdings["cash"] -= trade_value + commission
        elif direction == "SELL":
            self.current_positions[event.symbol] -= event.quantity
            self.current_holdings["cash"] += trade_value - commission
        else:
            raise ValueError(f"Unsupported fill direction: {event.direction}")

        self.current_holdings["commission"] += commission

    def create_equity_curve_dataframe(self) -> pd.DataFrame:
        curve = pd.DataFrame(self.all_holdings)
        curve = curve.drop_duplicates(subset=["datetime"], keep="last")
        curve = curve.set_index("datetime").sort_index()

        curve["returns"] = curve["total"].pct_change().fillna(0.0)
        curve["equity_curve"] = (1.0 + curve["returns"]).cumprod()
        rolling_peak = curve["equity_curve"].cummax()
        curve["drawdown"] = curve["equity_curve"] / rolling_peak - 1.0

        self.equity_curve = curve
        return curve

    def output_summary_stats(self) -> Dict[str, float]:
        if self.equity_curve is None or self.equity_curve.empty:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_duration_bars": 0.0,
                "trade_events": 0.0,
                "ending_capital": self.initial_capital,
            }

        returns = self.equity_curve["returns"]
        std = float(returns.std(ddof=0))
        sharpe = float(np.sqrt(252.0) * returns.mean() / std) if std > 0 else 0.0

        total_return = float(self.equity_curve["equity_curve"].iloc[-1] - 1.0)
        max_drawdown = float(self.equity_curve["drawdown"].min())

        drawdown_duration = 0
        max_drawdown_duration = 0
        for is_drawdown in (self.equity_curve["drawdown"] < 0.0).astype(int):
            if is_drawdown:
                drawdown_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
            else:
                drawdown_duration = 0

        positions_frame = pd.DataFrame(self.all_positions).set_index("datetime").sort_index()
        turnover = positions_frame[self.symbol_list].diff().abs().sum(axis=1).fillna(0.0)
        trade_events = int((turnover > 0.0).sum())

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_bars": float(max_drawdown_duration),
            "trade_events": float(trade_events),
            "ending_capital": float(self.equity_curve["total"].iloc[-1]),
        }


class XGBoostPortfolio(Portfolio):
    def __init__(
        self,
        data_handler: HistoricDataHandler,
        events: Queue,
        start_date: pd.Timestamp,
        initial_capital: float = 100000.0,
        order_size: int = 100,
        db_manager=None,
        kelly_fraction: float = 0.5,
        max_risk_pct: float = 0.2
    ) -> None:
        super().__init__(data_handler, events, start_date, initial_capital, order_size)
        from database import DatabaseManager
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.kelly_fraction = kelly_fraction  # Hệ số Fractional Kelly
        self.max_risk_pct = max_risk_pct      # Giới hạn phân bổ tối đa cho một mã cổ phiếu

    def generate_order_from_signal(self, signal: SignalEvent) -> Optional[OrderEvent]:
        signal_type = signal.signal_type.upper()
        symbol = signal.symbol
        current_qty = self.current_positions[symbol]

        if signal_type == "EXIT":
            if current_qty > 0:
                return OrderEvent(
                    symbol=symbol,
                    order_type="MKT",
                    quantity=abs(current_qty),
                    direction="SELL",
                )
            if current_qty < 0:
                return OrderEvent(
                    symbol=symbol,
                    order_type="MKT",
                    quantity=abs(current_qty),
                    direction="BUY",
                )

        if signal_type == "LONG" and current_qty == 0:
            close_price = self.data_handler.get_latest_bar_value(symbol, "close")
            if close_price <= 0:
                return None

            dt_str = signal.datetime.strftime('%Y-%m-%d %H:%M:%S')
            dt_date_str = signal.datetime.strftime('%Y-%m-%d')

            # Các tham số phân bổ Kelly mặc định
            confidence = 0.60
            expected_return = 0.015

            query = """
            SELECT confidence, decision_metadata
            FROM ensemble_decisions
            WHERE symbol = ? AND (timestamp = ? OR timestamp LIKE ?)
            ORDER BY timestamp DESC LIMIT 1
            """

            try:
                import json
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, (symbol, dt_str, f"{dt_date_str}%"))
                    row = cursor.fetchone()
                    if row:
                        confidence = float(row['confidence']) if row['confidence'] else 0.60
                        meta = json.loads(row['decision_metadata']) if row['decision_metadata'] else {}
                        predictions = meta.get("predictions", {})
                        expected_return = float(predictions.get("reg_5d", {}).get("expected_return", 0.015))
            except Exception:
                pass

            # Ràng buộc expected return an toàn
            expected_return = max(0.005, expected_return)

            # Thiết lập tỷ số Risk-Reward động dựa trên kỳ vọng lợi nhuận
            b = 1.5 + (expected_return * 10)

            # Công thức Kelly: f* = (p * b - (1 - p)) / b
            p = confidence
            f_star = (p * b - (1.0 - p)) / b

            # Áp dụng Fractional Kelly
            f_kelly = self.kelly_fraction * f_star

            # Ràng buộc tỷ lệ vốn phân bổ trong khoảng [2%, max_risk_pct]
            f_kelly = max(0.02, min(self.max_risk_pct, f_kelly))

            # Tính toán lượng tiền mặt cần phân bổ động
            total_equity = self.current_holdings["total"]
            target_value = f_kelly * total_equity

            # Giới hạn không vượt quá lượng tiền mặt thực tế đang có
            total_cash = self.current_holdings["cash"]
            target_value = min(target_value, total_cash * 0.95)

            dynamic_quantity = int(target_value / close_price)

            # Làm tròn về lô 10 cổ phiếu theo quy chuẩn thị trường
            dynamic_quantity = (dynamic_quantity // 10) * 10
            dynamic_quantity = max(10, dynamic_quantity)

            return OrderEvent(
                symbol=symbol,
                order_type="MKT",
                quantity=dynamic_quantity,
                direction="BUY",
            )

        return None


class SimulatedExecutionHandler:
    def __init__(
        self,
        events: Queue,
        data_handler: HistoricDataHandler,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_bps: float = 1.0,
        exchange: str = "SIM",
    ) -> None:
        self.events = events
        self.data_handler = data_handler
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps
        self.exchange = exchange

    def execute_order(self, event: OrderEvent) -> None:
        if event.type is not EventType.ORDER:
            return

        last_price = self.data_handler.get_latest_bar_value(event.symbol, "close")
        side = 1.0 if event.direction.upper() == "BUY" else -1.0
        slipped_price = last_price * (1.0 + side * self.slippage_bps / 10000.0)

        commission = max(self.min_commission, self.commission_per_share * event.quantity)
        fill_event = FillEvent(
            timeindex=self.data_handler.get_latest_bar_datetime(event.symbol),
            symbol=event.symbol,
            exchange=self.exchange,
            quantity=event.quantity,
            direction=event.direction.upper(),
            fill_cost=float(slipped_price),
            commission=float(commission),
        )
        self.events.put(fill_event)


class EventDrivenBacktester:
    def __init__(
        self,
        data_handler: HistoricDataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: SimulatedExecutionHandler,
        events: Queue,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = events

    def run(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        while self.data_handler.continue_backtest:
            self.data_handler.update_bars()

            while True:
                try:
                    event = self.events.get(block=False)
                except Empty:
                    break

                if event.type is EventType.MARKET:
                    self.strategy.calculate_signals(event)
                    self.portfolio.update_timeindex(event)
                elif event.type is EventType.SIGNAL:
                    self.portfolio.update_signal(event)
                elif event.type is EventType.ORDER:
                    self.execution_handler.execute_order(event)
                elif event.type is EventType.FILL:
                    self.portfolio.update_fill(event)

        equity_curve = self.portfolio.create_equity_curve_dataframe()
        summary = self.portfolio.output_summary_stats()
        return summary, equity_curve


def compute_performance_metrics(
    returns: pd.Series, annualization: int = 252
) -> Dict[str, float]:
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
        }

    equity = (1.0 + clean_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    years = max(len(clean_returns) / float(annualization), 1.0 / annualization)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)

    std = float(clean_returns.std(ddof=0))
    sharpe = float(np.sqrt(annualization) * clean_returns.mean() / std) if std > 0 else 0.0

    downside = clean_returns[clean_returns < 0.0]
    downside_std = float(downside.std(ddof=0))
    sortino = (
        float(np.sqrt(annualization) * clean_returns.mean() / downside_std)
        if downside_std > 0
        else 0.0
    )

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    volatility = float(clean_returns.std(ddof=0) * np.sqrt(annualization))
    win_rate = float((clean_returns > 0.0).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "win_rate": win_rate,
    }


def vectorized_ma_backtest(
    close_prices: pd.Series,
    short_window: int,
    long_window: int,
    tc_bps: float = 1.0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    prices = close_prices.astype(float).dropna().sort_index()
    if len(prices) < long_window + 2:
        raise ValueError("Not enough data for given windows.")

    fast_ma = prices.ewm(span=short_window, min_periods=short_window).mean()
    slow_ma = prices.ewm(span=long_window, min_periods=long_window).mean()
    raw_signal = (fast_ma > slow_ma).astype(float)
    position = raw_signal.shift(1).fillna(0.0)

    market_returns = prices.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    transaction_cost = turnover * (tc_bps / 10000.0)
    strategy_returns = position * market_returns - transaction_cost

    metrics = compute_performance_metrics(strategy_returns)
    metrics.update(
        {
            "short_window": float(short_window),
            "long_window": float(long_window),
            "trade_events": float((turnover > 0.0).sum()),
            "final_equity": float((1.0 + strategy_returns).cumprod().iloc[-1]),
        }
    )

    details = pd.DataFrame(
        {
            "price": prices,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "position": position,
            "market_returns": market_returns,
            "strategy_returns": strategy_returns,
        }
    )

    return metrics, details


def vectorized_grid_search(
    close_prices: pd.Series,
    short_windows: Iterable[int],
    long_windows: Iterable[int],
    tc_bps: float = 1.0,
) -> pd.DataFrame:
    results: List[Dict[str, float]] = []

    for short_window, long_window in product(short_windows, long_windows):
        if short_window >= long_window:
            continue
        metrics, _ = vectorized_ma_backtest(
            close_prices=close_prices,
            short_window=short_window,
            long_window=long_window,
            tc_bps=tc_bps,
        )
        results.append(metrics)

    if not results:
        raise ValueError("No valid parameter pair found. Check short/long windows.")

    leaderboard = pd.DataFrame(results)
    leaderboard = leaderboard.sort_values(
        by=["sharpe", "total_return"], ascending=False
    ).reset_index(drop=True)
    return leaderboard


def train_test_split_bars(
    frame: pd.DataFrame, train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.1 < train_ratio < 0.95):
        raise ValueError("train_ratio must be between 0.1 and 0.95")
    split_idx = int(len(frame) * train_ratio)
    if split_idx <= 0 or split_idx >= len(frame):
        raise ValueError("Invalid split. Need more bars.")
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def _run_event_driven_backtest(
    test_data: pd.DataFrame,
    symbol: str,
    short_window: int,
    long_window: int,
    initial_capital: float,
    order_size: int,
    tc_bps: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    events: Queue = Queue()
    data_handler = HistoricDataHandler(events=events, symbol_data={symbol: test_data})
    strategy = MovingAverageCrossStrategy(
        data_handler=data_handler,
        events=events,
        short_window=short_window,
        long_window=long_window,
    )
    portfolio = Portfolio(
        data_handler=data_handler,
        events=events,
        start_date=test_data.index[0],
        initial_capital=initial_capital,
        order_size=order_size,
    )
    execution_handler = SimulatedExecutionHandler(
        events=events,
        data_handler=data_handler,
        commission_per_share=0.005,
        min_commission=1.0,
        slippage_bps=tc_bps,
    )
    event_backtester = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        events=events,
    )
    event_summary, event_equity = event_backtester.run()
    return event_summary, event_equity


def run_hybrid_pipeline(
    price_data: pd.DataFrame,
    symbol: str = "ASSET",
    short_windows: Iterable[int] = (5, 10, 20, 30),
    long_windows: Iterable[int] = (50, 100, 150, 200),
    train_ratio: float = 0.7,
    initial_capital: float = 100000.0,
    order_size: int = 100,
    tc_bps: float = 1.0,
) -> Dict[str, object]:
    if "close" not in price_data.columns:
        raise ValueError("price_data must contain a close column")

    train_data, test_data = train_test_split_bars(price_data, train_ratio=train_ratio)

    leaderboard = vectorized_grid_search(
        close_prices=train_data["close"],
        short_windows=short_windows,
        long_windows=long_windows,
        tc_bps=tc_bps,
    )

    best = leaderboard.iloc[0]
    best_short = int(best["short_window"])
    best_long = int(best["long_window"])

    vector_oos_metrics, vector_oos_frame = vectorized_ma_backtest(
        close_prices=test_data["close"],
        short_window=best_short,
        long_window=best_long,
        tc_bps=tc_bps,
    )

    event_summary, event_equity = _run_event_driven_backtest(
        test_data=test_data,
        symbol=symbol,
        short_window=best_short,
        long_window=best_long,
        initial_capital=initial_capital,
        order_size=order_size,
        tc_bps=tc_bps,
    )

    return {
        "best_params": {"short_window": best_short, "long_window": best_long},
        "vectorized_leaderboard": leaderboard,
        "vectorized_oos_metrics": vector_oos_metrics,
        "vectorized_oos_frame": vector_oos_frame,
        "event_driven_summary": event_summary,
        "event_equity_curve": event_equity,
    }


def generate_walk_forward_splits(
    total_bars: int,
    train_size_bars: int,
    test_size_bars: int,
    step_size_bars: int,
    expanding_window: bool = True,
) -> List[Tuple[int, int, int, int]]:
    if train_size_bars <= 0 or test_size_bars <= 0 or step_size_bars <= 0:
        raise ValueError("train/test/step bars must all be positive integers")
    if total_bars < train_size_bars + test_size_bars:
        raise ValueError("Not enough bars for one walk-forward fold")

    splits: List[Tuple[int, int, int, int]] = []
    offset = 0
    while True:
        train_start = 0 if expanding_window else offset
        train_end = train_size_bars + offset
        test_start = train_end
        test_end = test_start + test_size_bars

        if test_end > total_bars:
            break

        splits.append((train_start, train_end, test_start, test_end))
        offset += step_size_bars

    if not splits:
        raise ValueError("No valid walk-forward fold produced. Check train/test/step bars")

    return splits


def run_walk_forward_optimization(
    price_data: pd.DataFrame,
    symbol: str = "ASSET",
    short_windows: Iterable[int] = (5, 10, 20, 30),
    long_windows: Iterable[int] = (50, 100, 150, 200),
    train_size_bars: int = 504,
    test_size_bars: int = 126,
    step_size_bars: int = 126,
    expanding_window: bool = True,
    initial_capital: float = 100000.0,
    order_size: int = 100,
    tc_bps: float = 1.0,
) -> Dict[str, object]:
    if "close" not in price_data.columns:
        raise ValueError("price_data must contain a close column")

    short_list = [int(item) for item in short_windows]
    long_list = [int(item) for item in long_windows]
    if not short_list or not long_list:
        raise ValueError("short_windows and long_windows cannot be empty")

    max_long = max(long_list)
    min_required_bars = max_long + 2
    if train_size_bars < min_required_bars:
        raise ValueError(
            f"train_size_bars must be at least {min_required_bars} for the selected long windows"
        )
    if test_size_bars < min_required_bars:
        raise ValueError(
            f"test_size_bars must be at least {min_required_bars} for the selected long windows"
        )

    splits = generate_walk_forward_splits(
        total_bars=len(price_data),
        train_size_bars=train_size_bars,
        test_size_bars=test_size_bars,
        step_size_bars=step_size_bars,
        expanding_window=expanding_window,
    )

    fold_rows: List[Dict[str, object]] = []
    vectorized_return_segments: List[pd.Series] = []
    event_return_segments: List[pd.Series] = []

    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(splits, start=1):
        train_data = price_data.iloc[train_start:train_end].copy()
        test_data = price_data.iloc[test_start:test_end].copy()

        leaderboard = vectorized_grid_search(
            close_prices=train_data["close"],
            short_windows=short_list,
            long_windows=long_list,
            tc_bps=tc_bps,
        )
        best = leaderboard.iloc[0]
        best_short = int(best["short_window"])
        best_long = int(best["long_window"])

        vector_metrics, vector_frame = vectorized_ma_backtest(
            close_prices=test_data["close"],
            short_window=best_short,
            long_window=best_long,
            tc_bps=tc_bps,
        )
        event_summary, event_equity = _run_event_driven_backtest(
            test_data=test_data,
            symbol=symbol,
            short_window=best_short,
            long_window=best_long,
            initial_capital=initial_capital,
            order_size=order_size,
            tc_bps=tc_bps,
        )

        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "short_window": best_short,
                "long_window": best_long,
                "vectorized_total_return": float(vector_metrics["total_return"]),
                "vectorized_sharpe": float(vector_metrics["sharpe"]),
                "vectorized_max_drawdown": float(vector_metrics["max_drawdown"]),
                "event_total_return": float(event_summary["total_return"]),
                "event_sharpe": float(event_summary["sharpe"]),
                "event_max_drawdown": float(event_summary["max_drawdown"]),
                "event_trade_events": float(event_summary["trade_events"]),
            }
        )

        fold_vector_returns = vector_frame["strategy_returns"].copy()
        fold_vector_returns.name = "strategy_returns"
        vectorized_return_segments.append(fold_vector_returns)

        fold_event_returns = event_equity["returns"].copy()
        fold_event_returns.name = "returns"
        event_return_segments.append(fold_event_returns)

    fold_results = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)

    vectorized_returns = pd.concat(vectorized_return_segments).sort_index()
    vectorized_returns = vectorized_returns[
        ~vectorized_returns.index.duplicated(keep="last")
    ]

    event_returns = pd.concat(event_return_segments).sort_index()
    event_returns = event_returns[~event_returns.index.duplicated(keep="last")]

    vectorized_aggregate_metrics = compute_performance_metrics(vectorized_returns)
    vectorized_aggregate_metrics["num_folds"] = float(len(splits))

    event_aggregate_metrics = compute_performance_metrics(event_returns)
    event_aggregate_metrics["num_folds"] = float(len(splits))
    event_aggregate_metrics["ending_capital"] = float(
        initial_capital * (1.0 + event_returns).cumprod().iloc[-1]
    )

    vectorized_oos_returns = pd.DataFrame({"strategy_returns": vectorized_returns})
    event_oos_frame = pd.DataFrame(
        {
            "returns": event_returns,
            "equity_curve": (1.0 + event_returns).cumprod(),
        }
    )

    return {
        "fold_results": fold_results,
        "vectorized_aggregate_metrics": vectorized_aggregate_metrics,
        "event_aggregate_metrics": event_aggregate_metrics,
        "vectorized_oos_returns": vectorized_oos_returns,
        "event_oos_frame": event_oos_frame,
    }


def load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame.columns = [str(col).strip().lower() for col in frame.columns]

    datetime_col = next(
        (col for col in ("datetime", "date", "timestamp", "time") if col in frame.columns),
        None,
    )
    if datetime_col is None:
        raise ValueError(
            "CSV must include one datetime column: datetime/date/timestamp/time."
        )

    frame[datetime_col] = pd.to_datetime(frame[datetime_col], errors="coerce")
    frame = frame.dropna(subset=[datetime_col]).set_index(datetime_col).sort_index()

    if "adj_close" in frame.columns and "close" not in frame.columns:
        frame["close"] = frame["adj_close"]
    if "close" not in frame.columns:
        raise ValueError("CSV must include close or adj_close column.")

    for col in ("open", "high", "low"):
        if col not in frame.columns:
            frame[col] = frame["close"]
    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    ohlcv = frame[["open", "high", "low", "close", "volume"]].astype(float)
    return ohlcv


def generate_synthetic_ohlcv(periods: int = 1400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)

    trend_component = np.linspace(0.0001, 0.0003, periods)
    cycle_component = np.sin(np.linspace(0.0, 12.0, periods)) * 0.0008
    noise_component = rng.normal(0.0, 0.01, periods)

    log_returns = trend_component + cycle_component + noise_component
    close = 100.0 * np.exp(np.cumsum(log_returns))

    frame = pd.DataFrame(index=dates)
    frame["close"] = close
    frame["open"] = frame["close"].shift(1).fillna(frame["close"].iloc[0])

    spread = np.abs(rng.normal(0.0, 0.004, periods)) * frame["close"]
    frame["high"] = frame[["open", "close"]].max(axis=1) + spread
    frame["low"] = frame[["open", "close"]].min(axis=1) - spread
    frame["volume"] = rng.integers(100_000, 800_000, size=periods)

    return frame[["open", "high", "low", "close", "volume"]]


def parse_int_list(text: str) -> List[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    parsed = [int(item) for item in values]
    if not parsed:
        raise ValueError("Expected at least one integer value")
    return parsed


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:>28}: {value: .6f}")
        else:
            print(f"{key:>28}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid backtesting pipeline: vectorized parameter scan + event-driven execution."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to OHLCV CSV. If omitted, synthetic data will be generated.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("single", "wfo"),
        default="single",
        help="single = one train/test split, wfo = walk-forward optimization.",
    )
    parser.add_argument("--symbol", type=str, default="ASSET", help="Trading symbol.")
    parser.add_argument(
        "--periods",
        type=int,
        default=1400,
        help="Bars for synthetic data (used only when --csv is not provided).",
    )
    parser.add_argument(
        "--short-windows",
        type=str,
        default="5,10,20,30",
        help="Comma-separated fast MA windows.",
    )
    parser.add_argument(
        "--long-windows",
        type=str,
        default="50,100,150,200",
        help="Comma-separated slow MA windows.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio for vectorized optimization (mode=single only).",
    )
    parser.add_argument(
        "--wfo-train-bars",
        type=int,
        default=504,
        help="Training bars per fold in walk-forward mode.",
    )
    parser.add_argument(
        "--wfo-test-bars",
        type=int,
        default=126,
        help="Testing bars per fold in walk-forward mode.",
    )
    parser.add_argument(
        "--wfo-step-bars",
        type=int,
        default=126,
        help="Step size bars between folds in walk-forward mode.",
    )
    parser.add_argument(
        "--wfo-window-type",
        type=str,
        choices=("expanding", "rolling"),
        default="expanding",
        help="Use expanding or rolling train window in walk-forward mode.",
    )
    parser.add_argument(
        "--tc-bps",
        type=float,
        default=1.0,
        help="Transaction cost in basis points.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for event-driven test.",
    )
    parser.add_argument(
        "--order-size",
        type=int,
        default=100,
        help="Fixed quantity per order for event-driven test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Display top K vectorized parameter sets.",
    )
    parser.add_argument(
        "--export-prefix",
        type=str,
        default=None,
        help="Prefix for CSV exports in both single and walk-forward modes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    short_windows = parse_int_list(args.short_windows)
    long_windows = parse_int_list(args.long_windows)

    if args.csv:
        data = load_ohlcv_csv(args.csv)
    else:
        data = generate_synthetic_ohlcv(periods=args.periods)

    if args.mode == "wfo":
        wfo_results = run_walk_forward_optimization(
            price_data=data,
            symbol=args.symbol,
            short_windows=short_windows,
            long_windows=long_windows,
            train_size_bars=args.wfo_train_bars,
            test_size_bars=args.wfo_test_bars,
            step_size_bars=args.wfo_step_bars,
            expanding_window=args.wfo_window_type == "expanding",
            initial_capital=args.initial_capital,
            order_size=args.order_size,
            tc_bps=args.tc_bps,
        )

        fold_results = wfo_results["fold_results"]
        print("\n=== Walk-Forward Fold Results ===")
        print(
            fold_results.to_string(
                index=False,
                float_format=lambda x: f"{x:0.6f}",
            )
        )

        print_metrics(
            "Walk-Forward Vectorized Aggregate",
            wfo_results["vectorized_aggregate_metrics"],
        )
        print_metrics(
            "Walk-Forward Event-Driven Aggregate",
            wfo_results["event_aggregate_metrics"],
        )

        if args.export_prefix:
            wfo_results["fold_results"].to_csv(
                f"{args.export_prefix}_wfo_folds.csv", index=False
            )
            wfo_results["vectorized_oos_returns"].to_csv(
                f"{args.export_prefix}_wfo_vectorized_returns.csv"
            )
            wfo_results["event_oos_frame"].to_csv(
                f"{args.export_prefix}_wfo_event_oos.csv"
            )
            print(f"\nExported walk-forward outputs with prefix: {args.export_prefix}")

        return

    results = run_hybrid_pipeline(
        price_data=data,
        symbol=args.symbol,
        short_windows=short_windows,
        long_windows=long_windows,
        train_ratio=args.train_ratio,
        initial_capital=args.initial_capital,
        order_size=args.order_size,
        tc_bps=args.tc_bps,
    )

    leaderboard = results["vectorized_leaderboard"]
    print("\n=== Top Vectorized Params (train set) ===")
    print(
        leaderboard.head(args.top_k).to_string(
            index=False, float_format=lambda x: f"{x:0.6f}"
        )
    )

    print("\n=== Selected Params ===")
    print(results["best_params"])

    print_metrics("Vectorized OOS Metrics", results["vectorized_oos_metrics"])
    print_metrics("Event-Driven OOS Metrics", results["event_driven_summary"])

    if args.export_prefix:
        results["vectorized_leaderboard"].to_csv(
            f"{args.export_prefix}_leaderboard.csv", index=False
        )
        results["vectorized_oos_frame"].to_csv(
            f"{args.export_prefix}_vectorized_oos.csv"
        )
        results["event_equity_curve"].to_csv(
            f"{args.export_prefix}_event_equity_curve.csv"
        )
        print(f"\nExported outputs with prefix: {args.export_prefix}")


if __name__ == "__main__":
    main()
