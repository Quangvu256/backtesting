from queue import Queue
import pandas as pd
import logging
from hybrid_backtester import Strategy, MarketEvent, SignalEvent, EventType, HistoricDataHandler
from database import DatabaseManager

logger = logging.getLogger("XGBoostStrategy")

class XGBoostEnsembleStrategy(Strategy):
    def __init__(self, data_handler: HistoricDataHandler, events: Queue, db_manager=None) -> None:
        self.data_handler = data_handler
        self.events = events
        self.db_manager = db_manager if db_manager else DatabaseManager()
        # Khởi tạo trạng thái thị trường cho từng mã: "OUT" hoặc "LONG"
        self.market_state = {symbol: "OUT" for symbol in self.data_handler.symbol_list}

    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Nhận MarketEvent từ sàn giao dịch giả lập.
        Truy vấn DB SQLite để đọc tín hiệu Ensemble biểu quyết lịch sử tương ứng tại ngày hiện tại.
        """
        if event.type is not EventType.MARKET:
            return

        for symbol in self.data_handler.symbol_list:
            bars = self.data_handler.get_latest_bars(symbol, 1)
            if not bars:
                continue
                
            dt = bars[-1][0]
            # Chuẩn hóa timestamp để so khớp với DB SQLite
            # Vì dữ liệu backtest chạy theo ngày (Daily), mốc giờ thường là 00:00:00 hoặc nến ngày
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            dt_date_str = dt.strftime('%Y-%m-%d')
            
            # Truy vấn DB
            query = """
            SELECT signal 
            FROM ensemble_decisions 
            WHERE symbol = ? AND (timestamp = ? OR timestamp LIKE ?)
            ORDER BY timestamp DESC LIMIT 1
            """
            
            signal_type = "HOLD"
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, (symbol, dt_str, f"{dt_date_str}%"))
                    row = cursor.fetchone()
                    if row:
                        signal_type = row['signal']
            except Exception as e:
                logger.error(f"Lỗi khi đọc tín hiệu Ensemble từ DB cho {symbol} tại {dt_str}: {e}")
                continue
            
            # Đưa ra quyết định giao dịch
            # Nếu tín hiệu là BUY/STRONG_BUY và chưa nắm giữ -> MUA (LONG)
            if signal_type in ["STRONG_BUY", "BUY"] and self.market_state[symbol] == "OUT":
                self.events.put(
                    SignalEvent(
                        symbol=symbol,
                        datetime=dt,
                        signal_type="LONG",
                        strength=1.5 if signal_type == "STRONG_BUY" else 1.0
                    )
                )
                self.market_state[symbol] = "LONG"
                logger.debug(f"[BUY SIGNAL] {symbol} tại {dt_date_str} - Tín hiệu: {signal_type}")
                
            # Nếu tín hiệu là SELL/STRONG_SELL và đang nắm giữ -> BÁN (EXIT)
            elif signal_type in ["STRONG_SELL", "SELL"] and self.market_state[symbol] == "LONG":
                self.events.put(
                    SignalEvent(
                        symbol=symbol,
                        datetime=dt,
                        signal_type="EXIT",
                        strength=1.5 if signal_type == "STRONG_SELL" else 1.0
                    )
                )
                self.market_state[symbol] = "OUT"
                logger.debug(f"[EXIT SIGNAL] {symbol} tại {dt_date_str} - Tín hiệu: {signal_type}")
