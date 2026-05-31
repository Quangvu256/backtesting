import logging
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from config import WATCHLIST
from database import DatabaseManager
from data_fetcher import DataFetcher
from news_sentiment import NewsSentimentAnalyzer
from feature_engineering import FeatureExtractor
from ml_pipeline import MultiModelTrainer, EnsembleVoter

logger = logging.getLogger("Scheduler")

class SystemScheduler:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.data_fetcher = DataFetcher(self.db_manager)
        self.news_analyzer = NewsSentimentAnalyzer(self.db_manager)
        self.feature_extractor = FeatureExtractor(self.db_manager)
        self.trainer = MultiModelTrainer(self.db_manager)
        self.voter = EnsembleVoter(self.db_manager)
        
        self.scheduler = BackgroundScheduler()
        self.is_running = False

    def job_update_ohlcv_and_predict(self):
        """
        Job chạy mỗi 5 phút:
        1. Tải dữ liệu OHLCV mới củawatchlist.
        2. Chạy feature engineering tạo đặc trưng mới nhất.
        3. Thực hiện Ensemble voting đưa ra tín hiệu mới nhất.
        """
        logger.info("=== [JOB START] Cập nhật dữ liệu OHLCV và dự đoán định lượng ===")
        try:
            # 1. Thu thập dữ liệu OHLCV 15 ngày qua để đảm bảo đủ dữ liệu tính kỹ thuật
            self.data_fetcher.update_all_symbols(days_back=15)
            
            # 2. Tạo đặc trưng & Dự đoán cho từng mã
            for symbol in WATCHLIST:
                logger.info(f"Đang xử lý đặc trưng cho {symbol}...")
                success = self.feature_extractor.build_and_save_features(symbol)
                
                if success:
                    # Chạy ensemble biểu quyết cho ra tín hiệu giao dịch mới nhất
                    decision = self.voter.make_decision(symbol)
                    if decision:
                        logger.info(f"Dự báo mới nhất cho {symbol}: {decision['signal']} (Score: {decision['final_score']:.4f})")
                else:
                    logger.warning(f"Không thể build đặc trưng cho {symbol} tại thời điểm này.")
                    
            logger.info("=== [JOB END] Hoàn thành cập nhật và dự đoán OHLCV ===")
        except Exception as e:
            logger.error(f"[ERROR] Lỗi trong Job OHLCV & Predict: {e}", exc_info=True)

    def job_fetch_and_analyze_news(self):
        """
        Job chạy mỗi 12 giờ:
        1. Đọc RSS cào tin tức tài chính mới.
        2. Phân tích cảm xúc qua Gemini API.
        """
        logger.info("=== [JOB START] Cập nhật tin tức và phân tích cảm xúc Gemini ===")
        try:
            self.news_analyzer.fetch_and_analyze_news()
            logger.info("=== [JOB END] Hoàn thành cập nhật tin tức ===")
        except Exception as e:
            logger.error(f"[ERROR] Lỗi trong Job Tin tức: {e}", exc_info=True)

    def job_retrain_models(self):
        """
        Job chạy hàng tuần hoặc thủ công để huấn luyện lại toàn bộ 6 mô hình cho mỗi mã cổ phiếu.
        """
        logger.info("=== [JOB START] Huấn luyện lại toàn bộ các mô hình XGBoost ===")
        try:
            # Tải dữ liệu lịch sử lớn trước khi huấn luyện (ví dụ 2 năm)
            self.data_fetcher.fetch_initial_historical_data(years_back=2)
            
            # Xây dựng lại toàn bộ đặc trưng lịch sử
            for symbol in WATCHLIST:
                logger.info(f"Xây dựng lại toàn bộ ma trận đặc trưng lịch sử cho {symbol}...")
                self.feature_extractor.build_and_save_features(symbol)
                
                # Huấn luyện lại model
                success = self.trainer.train_and_evaluate(symbol)
                if success:
                    logger.info(f"[FACT] Đã tái huấn luyện thành công toàn bộ mô hình cho {symbol}.")
                else:
                    logger.warning(f"[WARNING] Huấn luyện model lỗi hoặc không đủ dữ liệu cho {symbol}.")
                    
            logger.info("=== [JOB END] Hoàn thành huấn luyện lại các mô hình ===")
        except Exception as e:
            logger.error(f"[ERROR] Lỗi trong Job Re-train mô hình: {e}", exc_info=True)

    def start(self):
        """Khởi động bộ lập lịch ngầm"""
        if self.is_running:
            logger.warning("Scheduler đang chạy rồi.")
            return
            
        # Thêm các job lập lịch định kỳ
        # 1. Cập nhật giá & dự báo mỗi 5 phút
        self.scheduler.add_job(
            self.job_update_ohlcv_and_predict, 
            'interval', 
            minutes=5, 
            id='fetch_ohlcv_job',
            replace_existing=True
        )
        
        # 2. Cào tin tức & phân tích Gemini mỗi 12 giờ
        self.scheduler.add_job(
            self.job_fetch_and_analyze_news, 
            'interval', 
            hours=12, 
            id='fetch_news_job',
            replace_existing=True
        )
        
        # 3. Huấn luyện lại các mô hình vào chủ nhật lúc 00:00 hàng tuần
        self.scheduler.add_job(
            self.job_retrain_models, 
            'cron', 
            day_of_week='sun', 
            hour=0, 
            minute=0, 
            id='retrain_models_job',
            replace_existing=True
        )
        
        self.scheduler.start()
        self.is_running = True
        logger.info("[FACT] Trình lập lịch BackgroundScheduler đã bắt đầu chạy ngầm.")
        
        # Thực thi đồng bộ ngay lập tức các job khi khởi động để đảm bảo hệ thống có dữ liệu và dự đoán ban đầu
        logger.info("Đang chạy đồng bộ các job lập lịch ban đầu để khởi tạo dữ liệu...")
        self.job_fetch_and_analyze_news()
        self.job_update_ohlcv_and_predict()

    def shutdown(self):
        """Dừng bộ lập lịch"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Đã dừng hoạt động của Scheduler.")
