# TÀI LIỆU HYBRID BACKTESTER

## 1. Overview
| Thuộc tính | Chi tiết |
|------------|----------|
| Tên hệ thống | Hybrid Backtester (Event-Driven + Vectorized + Walk-Forward) |
| File xử lý chính | \hybrid_backtester.py\ |
| Ngôn ngữ | Python 3 |
| Thư viện | pandas, numpy |
| Mục tiêu | Tối ưu tham số nhanh, mô phỏng thực thi sát thực tế, và đánh giá OOS bền vững |

## 2. ASCII Architecture
\\	ext
+---------------------------------------------------------------+
|                      DATA SOURCE                              |
|   CSV OHLCV hoặc synthetic generator                          |
+-----------------------------+---------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|                PREPROCESS / NORMALIZE OHLCV                   |
+-----------------------------+---------------------------------+
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
+---------------------------+          +---------------------------+
|     VECTORIZED ENGINE     |          |   EVENT-DRIVEN ENGINE     |
| - Fast MA signal          |          | - Queue MARKET/SIGNAL/... |
| - Grid search parameters  |          | - Portfolio accounting    |
| - Fast metrics evaluation |          | - Commission & slippage   |
+-------------+-------------+          +-------------+-------------+
              |                                      ^
              +---------- best params ---------------+
                              |
                              v
+---------------------------------------------------------------+
|            SINGLE SPLIT hoặc WALK-FORWARD OOS                 |
+-----------------------------+---------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|   METRICS / LEADERBOARD / CSV EXPORT                          |
+---------------------------------------------------------------+
\
## 3. Layer Analysis
| WHY (Tại sao cần?) | WHERE (Nằm ở đâu?) | HOW (Hoạt động thế nào?) | WHEN (Khi nào dùng?) |
|--------------------|--------------------|--------------------------|----------------------|
| Khắc phục tốc độ chậm của Event-Driven khi quét tham số | \ectorized_grid_search\, \ectorized_ma_backtest\ | Tính toán ma trận trên pandas Series/DataFrame | Khi cần quét tham số nhanh (Prototyping) |
| Khắc phục điểm yếu thiếu tính thực tế của Vectorized | \EventDrivenBacktester\, \SimulatedExecutionHandler\ | Mô phỏng chu trình MARKET -> SIGNAL -> ORDER -> FILL | Khi cần đánh giá mức giảm lợi nhuận do phí/slippage |
| Đảm bảo tính bền vững, chống Overfitting (WFO) | \generate_walk_forward_splits\, un_walk_forward_optimization\ | Chia window rolling/expanding, test trên OOS | Khi chuẩn bị đưa mô hình vào Paper/Live Trading |
| Chuẩn hoá I/O, tạo báo cáo nhất quán cho Data/Portfolio | Hàm \main\, khối Metrics, \Portfolio\ | Ghi nhận PnL, tính toán metrics (Sharpe, Drawdown) | Cuối mỗi vòng lặp hoặc sau khi finish backtest |

## 4. Use Case
| Kịch bản | Mục đích | Dòng lệnh | Đầu ra chính |
|----------|----------|-----------|--------------|
| Khởi tạo nhanh (Single) | Test logic và ý tưởng MA | \python hybrid_backtester.py --mode single\ | Bảng Leaderboard, OOS Metrics cho Vectorized & Event-Driven |
| Chạy chế độ WFO (Expanding) | Tránh Overfitting, học thêm lịch sử | \python hybrid_backtester.py --mode wfo --wfo-window-type expanding\ | Folds Metrics, OOS Returns qua các chu kỳ |
| Chạy chế độ WFO (Rolling) | Tránh Overfitting, theo dõi Regime mới | \python hybrid_backtester.py --mode wfo --wfo-window-type rolling\ | Tương tự Expanding nhưng di chuyển khung Train |
| Xuất tệp tin (Export) | Lưu lại báo cáo để External Review | \python hybrid_backtester.py --mode single --export-prefix kq\ | Các file CSV bắt đầu bằng \kq_\ |

## 5. File Table
| Tên File | Chức năng (Đóng vai trò gì?) | Sự phụ thuộc (Gọi đến đâu?) |
|----------|------------------------------|-----------------------------|
| \hybrid_backtester.py\ | Chứa toàn bộ logic Event-driven, Vectorized, WFO và CLI | library: pandas, numpy |
| equirements.txt\ | Khai báo các thư viện phụ thuộc | Không có |
| \TAI_LIEU_5W1H_HYBRID_BACKTESTER.md\ | Tài liệu đặc tả hệ thống | Không có |

## 6. Business Rules
| Rule ID | Tên quy tắc | Nội dung kiểm soát |
|---------|-------------|--------------------|
| BR-01 | Ràng buộc Window | \short_window\ bắt buộc phải nhỏ hơn \long_window\ |
| BR-02 | Chống Look-ahead | Tín hiệu giao dịch (position) phải được dịch (shift) 1 bar |
| BR-03 | Chi phí giao dịch | Phải trừ chi phí (transaction cost) theo turnover và tỷ lệ bps |
| BR-04 | Dữ liệu đầu vào | File CSV bắt buộc phải có cột \close\ hoặc \dj_close\ |
| BR-05 | Điều kiện WFO | Chiều dài \	rain_size\ và \	est_size\ phải lớn hơn \long_window\ + 2 |
| BR-06 | Kết thúc event loop | Khi nhận được tín hiệu StopIteration, biến \continue_backtest\ = False |
