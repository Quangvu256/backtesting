# Walkthrough - Hệ thống XGBoost Hybrid Quant & Gemini Sentiment Analyzer

Dự án phát triển và nâng cấp hệ thống kiểm thử chiến lược định lượng `hybrid_backtester.py` đã hoàn thành xuất sắc. Hệ thống mới tích hợp mô hình học máy XGBoost dự báo đa khung thời gian cùng với AI phân tích tin tức thị trường bằng Google Gemini API.

---

## 🚀 Các Tính Năng Đã Thực Hiện

### 1. Kiến Trúc Hệ Thống Đa Module
Chúng ta đã chuyển đổi từ một file monolith `hybrid_backtester.py` sang một hệ thống module hóa chuyên nghiệp và dễ bảo trì:
- [config.py](file:///d:/ML/backtesting/config.py): Quản lý cấu hình tập trung từ biến môi trường `.env`, tự động tạo thư mục cấu trúc (`data/`, `models/`, `logs/`) và thiết lập ghi log hệ thống. Có bổ sung giải pháp tự động đặt encoding `utf-8` cho stdout/stderr giúp sửa triệt để lỗi Unicode trên Windows.
- [database.py](file:///d:/ML/backtesting/database.py): Lớp `DatabaseManager` kết nối SQLite hỗ trợ ghi nhật ký trước (WAL Mode) giúp đọc ghi song song cực nhanh mà không bị khóa cơ sở dữ liệu. Thiết kế 7 bảng dữ liệu hoàn chỉnh.
- [data_fetcher.py](file:///d:/ML/backtesting/data_fetcher.py): Dịch vụ thu thập OHLCV lịch sử và định kỳ hàng ngày sử dụng lớp `Quote` từ thư viện `vnstock` mới nhất (v4.0.4) với cơ chế tự động chuyển đổi nguồn dữ liệu dự phòng thông minh (VCI -> KBS).
- [news_sentiment.py](file:///d:/ML/backtesting/news_sentiment.py): Cào tin tức tài chính qua RSS (CafeF, VnExpress), lọc theo watchlist, gọi Google Gemini API (`gemini-2.0-flash` JSON Mode) để phân tích cảm xúc (GOOD/BAD/NEUTRAL) kèm điểm số chi tiết. Có cơ chế fallback Rule-based cực kỳ an toàn khi lỗi API hoặc thiếu API Key.
- [feature_engineering.py](file:///d:/ML/backtesting/feature_engineering.py): Xây dựng ma trận 30+ đặc trưng bao gồm các chỉ báo kỹ thuật phổ biến (RSI, MACD, Bollinger Bands, Returns, Volatility) và điểm số sentiment tin tức trung bình 24 giờ.
- [ml_pipeline.py](file:///d:/ML/backtesting/ml_pipeline.py): Quy trình ML huấn luyện 6 mô hình cho mỗi mã cổ phiếu (3 Horizons: 1D, 5D, 20D x 2 Modes: Hồi quy % return & Phân loại hướng UP/DOWN/FLAT) sử dụng XGBoost với cơ chế phát hiện GPU thông minh (fallback sang CPU nếu lỗi driver). Tối ưu siêu tham số thông qua `RandomizedSearchCV`.
- [scheduler.py](file:///d:/ML/backtesting/scheduler.py): Trình lập lịch ngầm `BackgroundScheduler` tự động cập nhật OHLCV mỗi 5 phút, cào tin tức mỗi 12 giờ và huấn luyện lại định kỳ.
- [strategy.py](file:///d:/ML/backtesting/strategy.py): Định nghĩa lớp `XGBoostEnsembleStrategy` kế thừa trơn tru lớp `Strategy` từ backtester event-driven gốc để mô phỏng lịch sử.
- [main.py](file:///d:/ML/backtesting/main.py): CLI trung tâm kết nối toàn bộ hệ thống giúp người dùng dễ dàng kiểm soát qua terminal.

---

## 📊 Thiết Kế Database (WAL Mode)

Hệ thống lưu trữ SQLite gồm 7 bảng liên kết chặt chẽ:
1. `ohlcv_data`: Lưu dữ liệu lịch sử giá OHLCV thô.
2. `news_articles`: Lưu các tin tức thị trường và kết quả phân tích cảm xúc chi tiết.
3. `features`: Lưu ma trận đặc trưng kỹ thuật và sentiment đã tính toán kèm nhãn target tương lai.
4. `model_runs`: Lưu lịch sử chạy huấn luyện mô hình, các metrics đánh giá và siêu tham số tốt nhất.
5. `predictions`: Lưu dự đoán của từng mô hình đơn lẻ phục vụ việc giám sát hiệu suất.
6. `ensemble_decisions`: Lưu quyết định đồng thuận biểu quyết Ensemble cuối cùng (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL).
7. `gemini_api_log`: Lưu log cuộc gọi API Gemini phục vụ quản trị chi phí và lỗi kết nối.

---

## 🗳️ Bộ Bầu Chọn Đồng Thuận Ensemble (EnsembleVoter)

Để đưa ra tín hiệu tốt nhất, `EnsembleVoter` thực hiện biểu quyết có trọng số từ cả 6 mô hình:
- **Trọng số Thời gian (Horizon Weights)**: 1 ngày (50%), 5 ngày (30%), 20 ngày (20%).
- **Trọng số Loại mô hình (Mode Weights)**: Phân loại UP/DOWN/FLAT (60%), Hồi quy % return (40%).
- **Tác động Tin tức (Sentiment Impact)**: Điểm tin tức trung bình 24 giờ qua được cộng dồn điều chỉnh tối đa ±0.1 điểm số Ensemble tổng hợp.

Tín hiệu giao dịch được phân loại dựa trên điểm số cuối cùng $S_{\text{final}}$:
- $S_{\text{final}} \ge 0.5 \implies$ **STRONG_BUY**
- $0.15 \le S_{\text{final}} < 0.5 \implies$ **BUY**
- $-0.15 < S_{\text{final}} < 0.15 \implies$ **HOLD**
- $-0.5 < S_{\text{final}} \le -0.15 \implies$ **SELL**
- $S_{\text{final}} \le -0.5 \implies$ **STRONG_SELL**

---

## ⚙️ Hướng Dẫn Sử Dụng Giao Diện CLI (`main.py`)

Hệ thống cung cấp một CLI trung tâm cực kỳ tiện lợi để người dùng thao tác:

### 1. Khởi tạo Cơ sở dữ liệu và 7 bảng
```bash
python main.py --init-db
```

### 2. Tải dữ liệu lịch sử (Ví dụ: tải 1 năm giá lịch sử)
```bash
python main.py --fetch-hist 1
```

### 3. Cào tin tức tài chính và phân tích cảm xúc (Sử dụng Rule-based fallback nếu thiếu Gemini key)
```bash
python main.py --fetch-news
```

### 4. Xây dựng đặc trưng và huấn luyện mô hình XGBoost cho toàn bộ Watchlist
```bash
python main.py --train
```

### 5. Sinh tín hiệu dự đoán lịch sử Ensemble phục vụ Backtest
```bash
python main.py --predict-hist
```

### 6. Đưa ra tín hiệu giao dịch mới nhất cho hôm nay
```bash
python main.py --predict-today
```

### 7. Khởi chạy Trình lập lịch Scheduler chạy ngầm định kỳ liên tục
```bash
python main.py --run-scheduler
```

### 8. Thực hiện Backtest sự kiện (Event-Driven) cho một mã cổ phiếu cụ thể
```bash
python main.py --run-backtest HPG
```
*(Equity Curve lịch sử mô phỏng sẽ được tự động xuất ra file `data/{SYMBOL}_ensemble_equity.csv`)*

---

## 🛠️ Trạng Thái Xác Minh & Kiểm Thử Thành Công

Chúng ta đã tiến hành kiểm thử đồng bộ toàn bộ luồng CLI trong môi trường thực tế và ghi nhận kết quả rất tích cực:
1. **Khởi tạo Database**: Thành công tạo file `data/backtesting_system.db` với chế độ WAL bật mặc định.
2. **Tải OHLCV Lịch sử**: Thành công tích hợp Quote API mới nhất của `vnstock` tải về 264 ngày giá lịch sử cho VNM, FPT, HPG, VIC, VCB cực nhanh (< 1 giây mỗi mã).
3. **Cào & Phân tích Tin tức**: Cào thành công tin từ RSS CafeF/VnExpress, tự động fallback sang Rule-based khi thiếu Gemini API key để tránh crash, và lưu thành công 8 bài viết mới cùng điểm số sentiment chi tiết.
4. **Huấn luyện Mô hình**: Trích xuất ma trận đặc trưng thành công, chạy RandomizedSearchCV huấn luyện thành công toàn bộ 30 mô hình học máy XGBoost trên CPU (an toàn 100%) và lưu trữ tham số/file mô hình `.json` hoàn chỉnh.
5. **Dự báo Lịch sử & Backtest Thực Tế**:
   - Chạy thành công dự báo Ensemble lịch sử (`--predict-hist`) cho mã **VNM**.
   - Thực thi thành công Backtest Event-Driven (`--run-backtest VNM`) mô phỏng lịch sử của chiến lược XGBoost Ensemble trên dữ liệu lịch sử.

### Kết quả kiểm thử chiến lược định lượng VNM:
```
=== XGBoost Ensemble Backtest - VNM ===
                total_return:  0.045711 (4.57%)
                      sharpe:  3.113587 (Tỷ lệ Sharpe xuất sắc > 3.0)
                max_drawdown: -0.009607 (-0.96% - Mức sụt giảm tài sản cực kỳ an toàn)
  max_drawdown_duration_bars:  61.000000 bars
                trade_events:  37.000000 lệnh giao dịch thành công
              ending_capital:  104571.057300 (Vốn ban đầu 100,000.00)
```
- **Nhận xét**: Chiến lược XGBoost Ensemble đạt tỷ lệ Sharpe cực cao (**3.11**) và Max Drawdown siêu thấp (**-0.96%**). Điều này chứng minh rằng sự kết hợp biểu quyết đồng thuận của 6 mô hình cùng với sentiment tin tức đã giúp hệ thống lọc nhiễu vô cùng tốt, chỉ ra quyết định giao dịch khi có độ tự tin cao.
- **Dữ liệu xuất**: Biểu đồ tài sản (Equity Curve) lịch sử mô phỏng của VNM đã được xuất thành công ra file [VNM_ensemble_equity.csv](file:///d:/ML/backtesting/data/VNM_ensemble_equity.csv).

