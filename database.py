import sqlite3
import json
import logging
from datetime import datetime
from config import DATABASE_PATH

logger = logging.getLogger("DatabaseManager")

class DatabaseManager:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        """
        Tạo kết nối tới SQLite và bật chế độ WAL để hỗ trợ đọc ghi song song.
        """
        conn = sqlite3.connect(self.db_path)
        # Bật ghi log trước (Write-Ahead Logging) để tránh block cơ sở dữ liệu khi đọc ghi đồng thời
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        # Trả về kết quả dưới dạng dict nếu cần
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """
        Khởi tạo 7 bảng lưu trữ của hệ thống.
        """
        queries = [
            # 1. Bảng lưu trữ dữ liệu giá lịch sử OHLCV
            """
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                symbol TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, timestamp)
            );
            """,
            # 2. Bảng tin tức tin tức thị trường và kết quả phân tích cảm xúc Gemini
            """
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                title TEXT,
                source TEXT,
                pub_date DATETIME,
                content TEXT,
                url TEXT UNIQUE,
                sentiment_label TEXT,
                sentiment_score REAL,
                sentiment_reason TEXT,
                analyzed_at DATETIME
            );
            """,
            # 3. Bảng các đặc trưng (features) phục vụ train/test và predict
            """
            CREATE TABLE IF NOT EXISTS features (
                symbol TEXT,
                timestamp DATETIME,
                feature_data TEXT, -- Lưu dữ liệu JSON của các đặc trưng kỹ thuật & sentiment
                target_1d_ret REAL,
                target_5d_ret REAL,
                target_20d_ret REAL,
                target_1d_cls INTEGER,
                target_5d_cls INTEGER,
                target_20d_cls INTEGER,
                PRIMARY KEY (symbol, timestamp)
            );
            """,
            # 4. Bảng ghi nhận lịch sử train model và siêu tham số
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                model_id TEXT PRIMARY KEY, -- Định dạng: {symbol}_{horizon}_{mode}
                symbol TEXT,
                horizon INTEGER,
                mode TEXT, -- regression hoặc classification
                trained_at DATETIME,
                best_params TEXT, -- Lưu cấu hình tham số tốt nhất dưới dạng JSON
                train_metrics TEXT, -- JSON
                test_metrics TEXT, -- JSON
                model_path TEXT
            );
            """,
            # 5. Bảng lưu trữ dự báo của từng model đơn lẻ
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp DATETIME,
                horizon INTEGER,
                mode TEXT,
                prediction_value REAL,
                prediction_class TEXT,
                confidence REAL,
                predicted_at DATETIME
            );
            """,
            # 6. Bảng lưu quyết định Ensemble biểu quyết cuối cùng
            """
            CREATE TABLE IF NOT EXISTS ensemble_decisions (
                symbol TEXT,
                timestamp DATETIME,
                signal TEXT, -- STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
                ensemble_score REAL,
                sentiment_impact REAL,
                decision_metadata TEXT, -- JSON chi tiết đóng góp của từng model
                executed INTEGER DEFAULT 0, -- Đã được backtester thực thi hay chưa
                PRIMARY KEY (symbol, timestamp)
            );
            """,
            # 7. Bảng ghi log các lượt gọi API Gemini để tính toán chi phí
            """
            CREATE TABLE IF NOT EXISTS gemini_api_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                status TEXT,
                error_message TEXT
            );
            """
        ]
        
        try:
            with self.get_connection() as conn:
                for query in queries:
                    conn.execute(query)
                conn.commit()
            logger.info("[FACT] Khởi tạo Database và các bảng thành công với chế độ WAL.")
        except Exception as e:
            logger.error(f"[ERROR] Lỗi khi tạo bảng Database: {e}", exc_info=True)

    # --- CÁC HÀM TIỆN ÍCH CHO BẢNG OHLCV ---
    def save_ohlcv(self, df):
        """
        Lưu DataFrame chứa dữ liệu OHLCV vào DB.
        df cần có các cột: symbol, timestamp, open, high, low, close, volume.
        """
        if df.empty:
            return
        
        query = """
        INSERT OR REPLACE INTO ohlcv_data (symbol, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        records = []
        for _, row in df.iterrows():
            # Chuẩn hóa timestamp sang chuỗi định dạng SQLite chuẩn
            ts = row['timestamp']
            if not isinstance(ts, str):
                ts = ts.strftime('%Y-%m-%d %H:%M:%S')
            records.append((
                row['symbol'],
                ts,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            ))
            
        try:
            with self.get_connection() as conn:
                conn.executemany(query, records)
                conn.commit()
            logger.info(f"Đã lưu {len(records)} bản ghi OHLCV thành công.")
        except Exception as e:
            logger.error(f"[ERROR] Lỗi khi lưu dữ liệu OHLCV: {e}", exc_info=True)

    def get_ohlcv(self, symbol, start_date=None, end_date=None):
        """
        Lấy dữ liệu OHLCV của 1 mã cổ phiếu từ DB, trả về danh sách dict hoặc DataFrame
        """
        query = "SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp ASC"
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"[ERROR] Lỗi khi đọc dữ liệu OHLCV: {e}", exc_info=True)
            return []
