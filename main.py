import os
import argparse
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from queue import Queue

from config import WATCHLIST, DATABASE_PATH, DEVICE
from database import DatabaseManager
from data_fetcher import DataFetcher
from news_sentiment import NewsSentimentAnalyzer
from feature_engineering import FeatureExtractor
from ml_pipeline import MultiModelTrainer, EnsembleVoter
from scheduler import SystemScheduler
from strategy import XGBoostEnsembleStrategy
from forecasting import build_future_predictions

# Import các module từ hybrid_backtester
from hybrid_backtester import (
    HistoricDataHandler,
    Portfolio,
    XGBoostPortfolio,
    SimulatedExecutionHandler,
    EventDrivenBacktester,
    print_metrics
)

logger = logging.getLogger("MainController")

def run_historical_predictions(symbol, db_manager):
    """
    Chạy dự đoán Ensemble lịch sử cho tất cả các điểm dữ liệu của symbol.
    Điều này giúp tạo sẵn dữ liệu trong bảng ensemble_decisions để phục vụ chạy backtest lịch sử.
    """
    logger.info(f"Bắt đầu chạy dự báo Ensemble lịch sử cho {symbol}...")
    extractor = FeatureExtractor(db_manager)
    voter = EnsembleVoter(db_manager)
    
    # 1. Đọc tất cả các ngày đặc trưng đã lưu
    query = "SELECT timestamp FROM features WHERE symbol = ? ORDER BY timestamp ASC"
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (symbol,))
            rows = cursor.fetchall()
    except Exception as e:
        logger.error(f"Lỗi khi đọc danh sách timestamp đặc trưng: {e}")
        return
        
    if not rows:
        logger.warning(f"Không có dữ liệu đặc trưng cho {symbol}. Vui lòng chạy feature engineering trước.")
        return
        
    timestamps = [r['timestamp'] for r in rows]
    logger.info(f"Phát hiện {len(timestamps)} điểm dữ liệu cần chạy dự đoán lịch sử.")
    
    # Gom danh sách model để kiểm tra
    query_models = "SELECT COUNT(model_id) as m_count FROM model_runs WHERE symbol = ?"
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query_models, (symbol,))
        m_count = cursor.fetchone()['m_count']
        
    if m_count < 6:
        logger.error(f"[ERROR] Chưa đủ 6 model được train cho {symbol} (hiện có {m_count}). Vui lòng chạy train trước.")
        return
        
    # Duyệt qua từng mốc thời gian lịch sử để tạo dự đoán Ensemble
    # Hàm make_decision mặc định lấy dòng đặc trưng mới nhất, chúng ta tạm thời thay đổi nhẹ logic
    # để lấy đặc trưng tại thời điểm t
    success_count = 0
    
    # Lấy toàn bộ đặc trưng để dự đoán hiệu quả hơn
    query_all_feat = "SELECT timestamp, feature_data FROM features WHERE symbol = ? ORDER BY timestamp ASC"
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query_all_feat, (symbol,))
        feat_rows = cursor.fetchall()
        
    for r in feat_rows:
        ts_str = r['timestamp']
        # Để EnsembleVoter sử dụng đúng thời điểm t, ta hack nhẹ get_latest_features 
        # hoặc tự gọi dự đoán. Để đơn giản và chính xác, chúng ta giả lập:
        # Mocking extractor.get_latest_features để trả về đặc trưng tại ts_str
        original_get_latest = extractor.get_latest_features
        
        def mock_get_latest(sym):
            import json
            feat = json.loads(r['feature_data'])
            df = pd.DataFrame([feat])
            df['timestamp'] = pd.to_datetime(ts_str)
            df.set_index('timestamp', inplace=True)
            return df
            
        voter.feature_extractor.get_latest_features = mock_get_latest
        
        # Gọi làm quyết định Ensemble
        decision = voter.make_decision(symbol)
        if decision:
            success_count += 1
            
        # Khôi phục hàm gốc
        voter.feature_extractor.get_latest_features = original_get_latest
        
    logger.info(f"[FACT] Hoàn tất dự đoán lịch sử cho {symbol}. Đã lưu {success_count} quyết định Ensemble.")

def run_backtest_ensemble(symbol, db_manager, initial_capital=100000.0, order_size=100, portfolio_mode="static"):
    """
    Thực hiện Backtest Event-Driven cho chiến lược XGBoost Ensemble.
    """
    logger.info(f"Đang chuẩn bị chạy Backtest Event-Driven cho {symbol}...")
    
    # 1. Đọc dữ liệu OHLCV từ DB
    ohlcv_rows = db_manager.get_ohlcv(symbol)
    if not ohlcv_rows:
        logger.error(f"[ERROR] Không có dữ liệu giá OHLCV cho {symbol} trong DB. Vui lòng chạy tải dữ liệu trước.")
        return
        
    df = pd.DataFrame(ohlcv_rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # 2. Kiểm tra xem đã có quyết định Ensemble trong DB chưa
    query_decisions = "SELECT COUNT(*) as d_count FROM ensemble_decisions WHERE symbol = ?"
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query_decisions, (symbol,))
        d_count = cursor.fetchone()['d_count']
        
    if d_count == 0:
        logger.warning(f"[WARNING] Không tìm thấy quyết định Ensemble lịch sử nào cho {symbol} trong DB. Đang tự động chạy dự đoán lịch sử trước...")
        run_historical_predictions(symbol, db_manager)
        
    # 3. Khởi tạo Event Loop Backtest
    events = Queue()
    data_handler = HistoricDataHandler(events=events, symbol_data={symbol: df})
    strategy = XGBoostEnsembleStrategy(data_handler=data_handler, events=events, db_manager=db_manager)
    
    if portfolio_mode == "dynamic":
        logger.info("[FACT] Sử dụng chế độ quản lý vốn động (Fractional Kelly).")
        portfolio = XGBoostPortfolio(
            data_handler=data_handler,
            events=events,
            start_date=df.index[0],
            initial_capital=initial_capital,
            order_size=order_size,
            db_manager=db_manager,
            kelly_fraction=0.5,
            max_risk_pct=0.2
        )
    else:
        logger.info("[FACT] Sử dụng chế độ quản lý vốn tĩnh (Cố định order_size).")
        portfolio = Portfolio(
            data_handler=data_handler,
            events=events,
            start_date=df.index[0],
            initial_capital=initial_capital,
            order_size=order_size
        )
        
    execution_handler = SimulatedExecutionHandler(
        events=events,
        data_handler=data_handler,
        commission_per_share=0.005,
        min_commission=1.0,
        slippage_bps=1.0
    )
    backtester = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        events=events
    )
    
    logger.info("=== [BACKTEST START] Đang thực thi mô phỏng sự kiện ===")
    summary, equity = backtester.run()
    
    print_metrics(f"XGBoost Ensemble Backtest - {symbol}", summary)
    
    # Lưu kết quả equity curve ra file
    equity_file = os.path.join("data", f"{symbol}_ensemble_equity.csv")
    equity.to_csv(equity_file)
    logger.info(f"[FACT] Đã xuất biểu đồ tài sản (Equity Curve) ra file: {equity_file}")

def main():
    parser = argparse.ArgumentParser(description="XGBoost & Gemini Hybrid Quant Trading System CLI")
    parser.add_argument("--init-db", action="store_true", help="Khởi tạo cơ sở dữ liệu SQLite và các bảng.")
    parser.add_argument("--fetch-hist", type=int, nargs="?", const=2, help="Tải dữ liệu OHLCV lịch sử (mặc định 2 năm).")
    parser.add_argument("--fetch-news", action="store_true", help="Cào tin tức tài chính và phân tích sentiment qua Gemini API.")
    parser.add_argument("--train", action="store_true", help="Huấn luyện lại toàn bộ 6 mô hình cho watchlist.")
    parser.add_argument("--predict-hist", action="store_true", help="Chạy dự báo lịch sử cho watchlist để sinh tín hiệu backtest.")
    parser.add_argument("--predict-today", action="store_true", help="Tính toán đặc trưng mới nhất và đưa ra tín hiệu cho hôm nay.")
    parser.add_argument("--forecast", type=str, metavar="SYMBOL", help="In future forecast 1D/5D/20D cho mot ma.")
    parser.add_argument("--run-scheduler", action="store_true", help="Khởi chạy Scheduler ngầm chạy liên tục (OHLCV 5p, News 12h).")
    parser.add_argument("--run-backtest", type=str, metavar="SYMBOL", help="Chạy backtest event-driven chiến lược XGBoost Ensemble cho 1 mã.")
    parser.add_argument("--portfolio-mode", type=str, choices=["static", "dynamic"], default="static", help="Chế độ quản lý vốn: static (cố định) hoặc dynamic (Fractional Kelly).")
    
    args = parser.parse_args()
    
    db_manager = DatabaseManager()
    data_fetcher = DataFetcher(db_manager)
    news_analyzer = NewsSentimentAnalyzer(db_manager)
    feature_extractor = FeatureExtractor(db_manager)
    trainer = MultiModelTrainer(db_manager)
    voter = EnsembleVoter(db_manager)
    
    # 1. Khởi tạo DB
    if args.init_db:
        db_manager.init_db()
        print("[FACT] Đã khởi tạo cơ sở dữ liệu thành công.")
        
    # 2. Tải dữ liệu lịch sử
    if args.fetch_hist is not None:
        years = args.fetch_hist
        print(f"Bắt đầu tải dữ liệu OHLCV lịch sử {years} năm cho watchlist {WATCHLIST}...")
        data_fetcher.fetch_initial_historical_data(years_back=years)
        
    # 3. Phân tích tin tức
    if args.fetch_news:
        print("Đang cào tin tức tài chính và gọi Gemini API phân tích...")
        news_analyzer.fetch_and_analyze_news()
        
    # 4. Huấn luyện mô hình
    if args.train:
        print("Bắt đầu trích xuất đặc trưng và huấn luyện mô hình XGBoost...")
        for symbol in WATCHLIST:
            # Build đặc trưng trước
            feature_extractor.build_and_save_features(symbol)
            # Train mô hình
            success = trainer.train_and_evaluate(symbol)
            if success:
                print(f"[FACT] Đã huấn luyện thành công 6 model cho {symbol}.")
            else:
                print(f"[WARNING] Huấn luyện model thất bại cho {symbol}.")
                
    # 5. Dự đoán lịch sử
    if args.predict_hist:
        print("Bắt đầu chạy dự đoán Ensemble lịch sử để phục vụ Backtest...")
        for symbol in WATCHLIST:
            run_historical_predictions(symbol, db_manager)
            
    # 6. Dự báo hôm nay
    if args.predict_today:
        print("Đang tải giá mới nhất, cập nhật đặc trưng và chạy dự đoán hôm nay...")
        data_fetcher.update_all_symbols(days_back=15)
        for symbol in WATCHLIST:
            feature_extractor.build_and_save_features(symbol)
            decision = voter.make_decision(symbol)
            if decision:
                print(f"\nTÍN HIỆU HÔM NAY - {symbol}:")
                print(f"  Thời gian: {decision['timestamp']}")
                print(f"  Tín hiệu : {decision['signal']}")
                print(f"  Điểm số  : {decision['final_score']:.4f}")
                print(f"  Sentiment Impact: {decision['sentiment_impact']:.4f}")
                
    # 7. Khởi chạy Scheduler ngầm
    if args.forecast:
        symbol = args.forecast.upper()
        forecast = build_future_predictions(symbol, db_manager)
        if not forecast:
            print(f"[UNKNOWN] Chua co future forecast cho {symbol}. Hay chay --predict-today truoc.")
        else:
            print(f"\nFUTURE FORECAST - {symbol}")
            print(f"  Base price : {forecast['base_price']:.2f} @ {forecast['price_timestamp']}")
            print(f"  Signal     : {forecast['signal']} | Score: {forecast['final_score']:.4f}")
            for item in forecast["forecast"]:
                projected = item["projected_price"]
                expected = item["expected_return"]
                confidence = item["classification_confidence"]
                projected_txt = f"{projected:.2f}" if projected is not None else "N/A"
                expected_txt = f"{expected * 100:+.2f}%" if expected is not None else "N/A"
                confidence_txt = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
                print(
                    f"  {item['horizon']:>2}D -> {item['target_timestamp'][:10]} | "
                    f"price={projected_txt} | ret={expected_txt} | "
                    f"dir={item['direction']} | cls_conf={confidence_txt}"
                )

    if args.run_scheduler:
        print("Đang khởi động Background Scheduler...")
        scheduler = SystemScheduler(db_manager)
        scheduler.start()
        print("Scheduler đang chạy ngầm. Bấm Ctrl+C để tắt.")
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()
            print("Đã dừng hệ thống Scheduler.")
            
    # 8. Chạy Backtest
    if args.run_backtest:
        symbol = args.run_backtest.upper()
        if symbol not in WATCHLIST:
            print(f"[WARNING] Mã {symbol} không nằm trong Watchlist mặc định. Vẫn sẽ thực hiện backtest.")
        run_backtest_ensemble(symbol, db_manager, portfolio_mode=args.portfolio_mode)

if __name__ == "__main__":
    main()
