import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DIRECTION_THRESHOLD, HORIZONS
from database import DatabaseManager

logger = logging.getLogger("FeatureEngineering")

FRAC_DIFF_D = 0.45
FRAC_DIFF_THRESHOLD = 1e-4
TRIPLE_BARRIER_VOL_WINDOW = 20
TRIPLE_BARRIER_WIDTH = 1.5

class FeatureExtractor:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager if db_manager else DatabaseManager()

    def load_raw_data(self, symbol, start_date=None, end_date=None):
        """Đọc dữ liệu giá OHLCV từ DB và chuyển thành DataFrame"""
        rows = self.db_manager.get_ohlcv(symbol, start_date, end_date)
        if not rows:
            logger.warning(f"Không tìm thấy dữ liệu giá cho {symbol} trong DB.")
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def fractional_difference(self, series, diff_order=FRAC_DIFF_D, threshold=FRAC_DIFF_THRESHOLD):
        """
        Fixed-width fractional differentiation.
        [FACT] Chi dung qua khu/hien tai tai moi timestamp, khong dung du lieu tuong lai.
        """
        series = pd.Series(series).astype(float)
        weights = [1.0]
        k = 1
        while True:
            weight = -weights[-1] * (diff_order - k + 1) / k
            if abs(weight) < threshold:
                break
            weights.append(weight)
            k += 1
            if k > len(series):
                break

        weights = np.array(weights[::-1], dtype=float)
        width = len(weights)
        raw = series.to_numpy(dtype=float)
        values = np.full(len(series), np.nan, dtype=float)
        for idx in range(width - 1, len(series)):
            window = raw[idx - width + 1:idx + 1]
            if np.isfinite(window).all():
                values[idx] = np.dot(weights, window)
        return values

    def calculate_technical_indicators(self, df):
        """Tính toán các đặc trưng kỹ thuật từ dữ liệu OHLCV"""
        if len(df) < 30:
            logger.warning("Dữ liệu quá ngắn để tính toán các chỉ báo kỹ thuật (cần tối thiểu 30 phiên).")
            return df
            
        df = df.copy()
        
        # 1. Simple & Exponential Moving Averages
        for window in [5, 10, 20, 30]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
        # Tỷ lệ giá so với các đường trung bình
        df['price_vs_sma20'] = df['close'] / df['sma_20']
        df['price_vs_ema20'] = df['close'] / df['ema_20']
        
        # 2. Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 4. Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # 5. Returns lịch sử (Đặc trưng động lượng)
        for lag in [1, 3, 5, 10, 20]:
            df[f'return_{lag}d'] = df['close'].pct_change(periods=lag)
            
        # 6. Độ biến động (Volatility)
        df['volatility_10d'] = df['return_1d'].rolling(window=10).std()
        df['volatility_20d'] = df['return_1d'].rolling(window=20).std()

        frac_col = f'fracdiff_close_{str(FRAC_DIFF_D).replace(".", "_")}'
        df[frac_col] = self.fractional_difference(
            np.log(df['close'].clip(lower=1e-8)),
            diff_order=FRAC_DIFF_D,
            threshold=FRAC_DIFF_THRESHOLD
        )
        
        # 7. Volume Features
        df['volume_sma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma5'] + 1e-8)
        
        return df

    def integrate_news_sentiment(self, df, symbol):
        """
        Tích hợp điểm số cảm xúc (Sentiment Score) từ DB.
        Lấy trung bình điểm số tin tức trong vòng 24 giờ trước mỗi timestamp của giá.
        """
        df = df.copy()
        df['sentiment_score'] = 0.0
        df['news_count'] = 0
        
        # Đọc tất cả tin tức liên quan đến mã này
        query = """
        SELECT pub_date, sentiment_score 
        FROM news_articles 
        WHERE symbol = ? AND sentiment_score IS NOT NULL
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                news_rows = cursor.fetchall()
                
            if not news_rows:
                return df
                
            news_df = pd.DataFrame([dict(r) for r in news_rows])
            news_df['pub_date'] = pd.to_datetime(news_df['pub_date'])
            
            # Map điểm sentiment vào mỗi mốc thời gian của giá
            for idx, row in df.iterrows():
                price_time = row['timestamp']
                # Lấy tin tức trong vòng 24h trước thời điểm này
                start_window = price_time - timedelta(hours=24)
                
                matched_news = news_df[
                    (news_df['pub_date'] >= start_window) & 
                    (news_df['pub_date'] <= price_time)
                ]
                
                if not matched_news.empty:
                    df.at[idx, 'sentiment_score'] = matched_news['sentiment_score'].mean()
                    df.at[idx, 'news_count'] = len(matched_news)
                    
        except Exception as e:
            logger.error(f"Lỗi khi tích hợp sentiment cho mã {symbol}: {e}")
            
        return df

    def create_targets(self, df):
        """
        Create targets with Triple Barrier labels.
        Regression target stays as vertical-barrier return for forecast compatibility.
        Classification: 0=FLAT, 1=TAKE_PROFIT, 2=STOP_LOSS.
        """
        df = df.copy()
        volatility = df['return_1d'].rolling(window=TRIPLE_BARRIER_VOL_WINDOW).std()

        for horizon in HORIZONS:
            future_close = df['close'].shift(-horizon)
            target_ret = (future_close - df['close']) / df['close']
            df[f'target_{horizon}d_ret'] = target_ret

            cls_col = f'target_{horizon}d_cls'
            df[cls_col] = self.triple_barrier_labels(
                close=df['close'],
                horizon=horizon,
                volatility=volatility
            )
            df.loc[target_ret.isna(), cls_col] = np.nan

        return df

    def triple_barrier_labels(self, close, horizon, volatility):
        """Triple Barrier: 0=vertical/sideway, 1=upper hit first, 2=lower hit first."""
        close = pd.Series(close).astype(float).reset_index(drop=True)
        volatility = pd.Series(volatility).astype(float).reset_index(drop=True)
        labels = np.full(len(close), np.nan)

        for idx in range(len(close)):
            end_idx = idx + int(horizon)
            if end_idx >= len(close) or not np.isfinite(close.iloc[idx]):
                continue

            vol = volatility.iloc[idx]
            barrier = max(float(vol) * TRIPLE_BARRIER_WIDTH, DIRECTION_THRESHOLD) if pd.notna(vol) else DIRECTION_THRESHOLD
            path_returns = (close.iloc[idx + 1:end_idx + 1] / close.iloc[idx]) - 1.0

            label = 0
            for ret in path_returns:
                if ret >= barrier:
                    label = 1
                    break
                if ret <= -barrier:
                    label = 2
                    break
            labels[idx] = label

        return labels
    def build_and_save_features(self, symbol, start_date=None, end_date=None):
        """
        Xây dựng toàn bộ ma trận đặc trưng cho một mã cổ phiếu và lưu vào SQLite DB.
        """
        # 1. Đọc dữ liệu thô
        df = self.load_raw_data(symbol, start_date, end_date)
        if df.empty:
            return False
            
        # 2. Tính các chỉ báo kỹ thuật
        df = self.calculate_technical_indicators(df)
        
        # 3. Tích hợp tin tức cảm xúc
        df = self.integrate_news_sentiment(df, symbol)
        
        # 4. Tạo targets
        df = self.create_targets(df)
        
        # 5. Làm sạch dữ liệu
        # Drop các hàng đầu tiên bị NaN do tính toán rolling indicators (ví dụ: sma_30 cần 30 hàng)
        df_cleaned = df.dropna(subset=['rsi_14', 'bb_upper']).copy()
        
        if df_cleaned.empty:
            logger.warning(f"Không còn dữ liệu cho {symbol} sau khi làm sạch các giá trị NaN chỉ báo.")
            return False
            
        # Xác định danh sách các cột đặc trưng (features)
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20', 'sma_30', 'ema_30',
            'price_vs_sma20', 'price_vs_ema20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_pct', 'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
            'volatility_10d', 'volatility_20d', f'fracdiff_close_{str(FRAC_DIFF_D).replace(".", "_")}',
            'volume_ratio', 'sentiment_score', 'news_count'
        ]
        
        # Lưu vào DB
        query = """
        INSERT OR REPLACE INTO features (
            symbol, timestamp, feature_data,
            target_1d_ret, target_5d_ret, target_20d_ret,
            target_1d_cls, target_5d_cls, target_20d_cls
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        records = []
        for _, row in df_cleaned.iterrows():
            ts = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Gom tất cả các đặc trưng thành dictionary JSON để lưu vào 1 cột động
            features_dict = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in feature_cols if col in df_cleaned.columns}
            
            # Đọc giá trị target (nếu có, để trống nếu ở cuối chuỗi dùng để predict)
            t_1d_ret = float(row['target_1d_ret']) if pd.notna(row['target_1d_ret']) else None
            t_5d_ret = float(row['target_5d_ret']) if pd.notna(row['target_5d_ret']) else None
            t_20d_ret = float(row['target_20d_ret']) if pd.notna(row['target_20d_ret']) else None
            
            t_1d_cls = int(row['target_1d_cls']) if pd.notna(row['target_1d_cls']) else None
            t_5d_cls = int(row['target_5d_cls']) if pd.notna(row['target_5d_cls']) else None
            t_20d_cls = int(row['target_20d_cls']) if pd.notna(row['target_20d_cls']) else None
            
            records.append((
                symbol,
                ts,
                json.dumps(features_dict),
                t_1d_ret, t_5d_ret, t_20d_ret,
                t_1d_cls, t_5d_cls, t_20d_cls
            ))
            
        try:
            with self.db_manager.get_connection() as conn:
                conn.executemany(query, records)
                conn.commit()
            logger.info(f"[FACT] Đã xây dựng và lưu {len(records)} ma trận đặc trưng cho {symbol} vào Database.")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Lỗi khi lưu đặc trưng của {symbol} vào DB: {e}", exc_info=True)
            return False

    def get_features_for_training(self, symbol):
        """
        Đọc các đặc trưng đã lưu trong DB để phục vụ huấn luyện mô hình.
        Trả về X (DataFrame đặc trưng) và y_dict (chứa các nhãn hồi quy và phân loại).
        """
        query = """
        SELECT timestamp, feature_data, 
               target_1d_ret, target_5d_ret, target_20d_ret,
               target_1d_cls, target_5d_cls, target_20d_cls
        FROM features 
        WHERE symbol = ?
        ORDER BY timestamp ASC
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                rows = cursor.fetchall()
                
            if not rows:
                return pd.DataFrame(), {}
                
            timestamps = []
            feature_list = []
            targets = {
                "reg_1": [], "reg_5": [], "reg_20": [],
                "cls_1": [], "cls_5": [], "cls_20": []
            }
            
            for row in rows:
                # Chỉ lấy những bản ghi đã có đầy đủ nhãn mục tiêu (phục vụ train)
                # Ta check target_20d_ret vì nó xa nhất, nếu nó có thì các target 1d, 5d chắc chắn có
                if row['target_20d_ret'] is None:
                    continue
                    
                timestamps.append(row['timestamp'])
                feature_list.append(json.loads(row['feature_data']))
                
                targets["reg_1"].append(row['target_1d_ret'])
                targets["reg_5"].append(row['target_5d_ret'])
                targets["reg_20"].append(row['target_20d_ret'])
                
                targets["cls_1"].append(row['target_1d_cls'])
                targets["cls_5"].append(row['target_5d_cls'])
                targets["cls_20"].append(row['target_20d_cls'])
                
            if not feature_list:
                return pd.DataFrame(), {}
                
            X = pd.DataFrame(feature_list)
            X['timestamp'] = pd.to_datetime(timestamps)
            X.set_index('timestamp', inplace=True)
            
            # Chuyển đổi target thành Series/Array
            y_dict = {k: np.array(v) for k, v in targets.items()}
            return X, y_dict
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu huấn luyện cho {symbol}: {e}")
            return pd.DataFrame(), {}

    def get_latest_features(self, symbol):
        """
        Lấy dòng đặc trưng mới nhất (không cần nhãn target) để phục vụ cho dự đoán thời gian thực.
        """
        query = """
        SELECT timestamp, feature_data 
        FROM features 
        WHERE symbol = ? 
        ORDER BY timestamp DESC LIMIT 1
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                row = cursor.fetchone()
                
            if row:
                feat = json.loads(row['feature_data'])
                df = pd.DataFrame([feat])
                df['timestamp'] = pd.to_datetime(row['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
        except Exception as e:
            logger.error(f"Lỗi khi lấy đặc trưng mới nhất của {symbol}: {e}")
        return pd.DataFrame()
