import logging
import pandas as pd
from datetime import datetime, timedelta
from vnstock.api.quote import Quote
from config import WATCHLIST
from database import DatabaseManager

logger = logging.getLogger("DataFetcher")

class DataFetcher:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager if db_manager else DatabaseManager()

    def fetch_ohlcv(self, symbol, start_date, end_date, source='VCI'):
        """
        Tải dữ liệu OHLCV của 1 mã cổ phiếu từ thư viện vnstock mới nhất và lưu vào cơ sở dữ liệu.
        start_date, end_date dạng 'YYYY-MM-DD'
        """
        logger.info(f"Đang tải dữ liệu OHLCV cho {symbol} từ {start_date} đến {end_date} bằng nguồn {source}...")
        try:
            # Khởi tạo quote object từ bộ vnstock mới nhất
            q = Quote(symbol=symbol, source=source)
            # Tải lịch sử giá dạng DataFrame
            df = q.history(start=start_date, end=end_date)
            
            if df is None or df.empty:
                logger.warning(f"[UNVERIFIED] Không có dữ liệu trả về cho {symbol} từ nguồn {source}.")
                return pd.DataFrame()

            # Chuẩn hóa dữ liệu trả về
            df = df.copy()
            if 'time' in df.columns:
                df.rename(columns={'time': 'timestamp'}, inplace=True)
            elif 'date' in df.columns:
                df.rename(columns={'date': 'timestamp'}, inplace=True)
                
            # Đảm bảo timestamp ở dạng string YYYY-MM-DD HH:MM:SS hoặc YYYY-MM-DD
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            df['symbol'] = symbol
            
            # Đảm bảo các cột số có kiểu dữ liệu phù hợp
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['timestamp', 'close'], inplace=True)
            
            # Lưu vào Database
            db_df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            self.db_manager.save_ohlcv(db_df)
            
            logger.info(f"[FACT] Tải và lưu thành công {len(db_df)} bản ghi OHLCV của mã {symbol}.")
            return db_df
            
        except Exception as e:
            logger.error(f"[ERROR] Lỗi khi tải dữ liệu OHLCV cho {symbol} từ {source}: {e}", exc_info=True)
            # Thử đổi nguồn dự phòng nếu VCI lỗi (chuyển sang KBS)
            if source == 'VCI':
                logger.info(f"Đang thử tải lại {symbol} với nguồn dự phòng 'KBS'...")
                return self.fetch_ohlcv(symbol, start_date, end_date, source='KBS')
            return pd.DataFrame()

    def update_all_symbols(self, days_back=15):
        """
        Cập nhật dữ liệu OHLCV mới nhất cho toàn bộ mã trong WATCHLIST.
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"Bắt đầu cập nhật dữ liệu OHLCV định kỳ cho watchlist: {WATCHLIST}")
        for symbol in WATCHLIST:
            self.fetch_ohlcv(symbol, start_date, end_date)
        logger.info("Hoàn tất cập nhật dữ liệu OHLCV cho toàn bộ watchlist.")

    def fetch_initial_historical_data(self, years_back=2):
        """
        Tải dữ liệu lịch sử lớn (ví dụ 2 năm) để phục vụ cho việc huấn luyện mô hình ban đầu.
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')
        
        logger.info(f"Bắt đầu tải dữ liệu lịch sử {years_back} năm để phục vụ huấn luyện mô hình...")
        for symbol in WATCHLIST:
            self.fetch_ohlcv(symbol, start_date, end_date)
        logger.info("Hoàn tất tải dữ liệu lịch sử huấn luyện.")
