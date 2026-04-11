# 📊 Project Report — Hybrid Backtester

> **Phiên bản:** 1.0  
> **Ngày tạo:** 2026-04-11  
> **Ngôn ngữ:** Python 3.10+  
> **File chính:** `hybrid_backtester.py` (1137 dòng)  
> **Dependencies:** `numpy>=1.24.0`, `pandas>=2.0.0`

---

## 1. Tổng quan Dự án

| Thuộc tính | Chi tiết |
|---|---|
| **Tên dự án** | Hybrid Backtester — Hệ thống kiểm thử chiến lược giao dịch lai (Vectorized + Event-Driven) |
| **Mục tiêu** | Tối ưu siêu tham số (parameter optimization) trên tập train bằng vectorized engine tốc độ cao, sau đó xác nhận (validate) trên tập test bằng event-driven engine mô phỏng sát thực tế |
| **Chiến lược mặc định** | Moving Average Crossover (EMA ngắn cắt EMA dài) |
| **Chế độ vận hành** | 2 chế độ: `single` (train/test split cố định) và `wfo` (Walk-Forward Optimization) |
| **Giao diện** | CLI (Command-Line Interface) qua `argparse` |
| **Triết lý thiết kế** | Monolith single-file, zero external API dependency, reproducible (synthetic data seeded) |

---

## 2. Sơ đồ Kiến trúc Tổng thể (ASCII Art)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CLI ENTRY POINT (main)                              │
│                     build_parser() → parse args → dispatch                     │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         ┌──────────────────┐          ┌─────────────────────┐
         │  mode = "single" │          │   mode = "wfo"      │
         │  run_hybrid_     │          │  run_walk_forward_  │
         │  pipeline()      │          │  optimization()     │
         └────────┬─────────┘          └──────────┬──────────┘
                  │                               │
                  ▼                               ▼
  ┌───────────────────────────────┐  ┌──────────────────────────────────┐
  │  PHASE 1: VECTORIZED ENGINE   │  │  FOR EACH FOLD:                  │
  │  ┌──────────────────────────┐ │  │   ┌────────────────────────────┐ │
  │  │ train_test_split_bars()  │ │  │   │ generate_walk_forward_    │ │
  │  │ vectorized_grid_search() │ │  │   │ splits()                  │ │
  │  │ → Best (short, long)     │ │  │   │ vectorized_grid_search()  │ │
  │  └──────────────────────────┘ │  │   │ → Best params per fold    │ │
  │                               │  │   └────────────────────────────┘ │
  │  PHASE 2: EVENT-DRIVEN ENGINE │  │   ┌────────────────────────────┐ │
  │  ┌──────────────────────────┐ │  │   │ vectorized_ma_backtest()  │ │
  │  │ HistoricDataHandler      │ │  │   │ _run_event_driven_        │ │
  │  │ MovingAverageCross       │ │  │   │ backtest()                │ │
  │  │ Portfolio                │ │  │   └────────────────────────────┘ │
  │  │ SimulatedExecution       │ │  │                                  │
  │  │ EventDrivenBacktester    │ │  │  AGGREGATE: concat returns →     │
  │  └──────────────────────────┘ │  │  compute_performance_metrics()   │
  └───────────────────────────────┘  └──────────────────────────────────┘
                  │                               │
                  ▼                               ▼
         ┌──────────────────────────────────────────────┐
         │              OUTPUT LAYER                     │
         │  print_metrics() + optional CSV export        │
         │  (--export-prefix)                            │
         └──────────────────────────────────────────────┘
```

---

## 3. Sơ đồ Luồng Sự kiện Event-Driven (ASCII Art)

```
    ┌──────────────────┐
    │  DataHandler      │
    │  update_bars()    │──── lấy bar tiếp theo từ iterator
    └────────┬─────────┘
             │ put(MarketEvent)
             ▼
    ┌──────────────────┐
    │   EVENT QUEUE     │◄──────────────────────────────────┐
    │   (FIFO Queue)    │                                   │
    └────────┬─────────┘                                   │
             │ get(block=False)                             │
             ▼                                              │
    ┌──────────────────────────────────────────┐            │
    │  EventDrivenBacktester.run() — Dispatcher│            │
    │  ┌─────────────────────────────────────┐ │            │
    │  │ MARKET → Strategy.calculate_signals │ │            │
    │  │          Portfolio.update_timeindex  │ │            │
    │  │                                     │ │            │
    │  │ SIGNAL → Portfolio.update_signal    │─┼───► put(OrderEvent)
    │  │                                     │ │            │
    │  │ ORDER  → Execution.execute_order   │─┼───► put(FillEvent)
    │  │                                     │ │            │
    │  │ FILL   → Portfolio.update_fill      │ │            │
    │  └─────────────────────────────────────┘ │            │
    └──────────────────────────────────────────┘            │
             │                                              │
             │ Khi Strategy phát hiện crossover:            │
             │ put(SignalEvent) ────────────────────────────┘
```

**Thứ tự xử lý trong 1 bar:**

```
DataHandler.update_bars()
    → [MarketEvent vào Queue]
        → Strategy.calculate_signals()   → có thể put(SignalEvent)
        → Portfolio.update_timeindex()
            → [SignalEvent vào Queue]
                → Portfolio.generate_order_from_signal() → put(OrderEvent)
                    → [OrderEvent vào Queue]
                        → ExecutionHandler.execute_order() → put(FillEvent)
                            → [FillEvent vào Queue]
                                → Portfolio.update_fill()
```

---

## 4. Phân tích Tầng (Layer Analysis)

### 4.1 Tầng 1 — Event & Data Layer

#### 4.1.1 `EventType` (Enum)

| 5W1H | Chi tiết |
|---|---|
| **What** | Enum định nghĩa 4 loại sự kiện trong hệ thống: `MARKET`, `SIGNAL`, `ORDER`, `FILL` |
| **Why** | Cung cấp type-safety cho hệ thống message passing; mỗi handler chỉ xử lý đúng loại event của mình |
| **Where** | `hybrid_backtester.py` dòng 14–18 |
| **When** | Được sử dụng mỗi khi một event được tạo hoặc kiểm tra type trong dispatcher loop |
| **Who** | Tất cả các component tạo/nhận event: DataHandler, Strategy, Portfolio, ExecutionHandler |
| **How** | Sử dụng `enum.auto()` để gán giá trị tự động. Mỗi dataclass event có `type` field mặc định tương ứng |

**Bảng chi tiết 4 Event Types:**

| Event Type | Dataclass | Producer | Consumer | Payload chính |
|---|---|---|---|---|
| `MARKET` | `MarketEvent` | `HistoricDataHandler.update_bars()` | `Strategy`, `Portfolio` | Không có payload (chỉ là trigger) |
| `SIGNAL` | `SignalEvent` | `Strategy.calculate_signals()` | `Portfolio.update_signal()` | `symbol`, `datetime`, `signal_type`, `strength` |
| `ORDER` | `OrderEvent` | `Portfolio.generate_order_from_signal()` | `ExecutionHandler.execute_order()` | `symbol`, `order_type`, `quantity`, `direction` |
| `FILL` | `FillEvent` | `ExecutionHandler.execute_order()` | `Portfolio.update_fill()` | `timeindex`, `symbol`, `exchange`, `quantity`, `direction`, `fill_cost`, `commission` |

---

#### 4.1.2 `HistoricDataHandler`

| 5W1H | Chi tiết |
|---|---|
| **What** | Component quản lý dữ liệu OHLCV lịch sử, phát ra dữ liệu theo từng bar một (drip-feed) để mô phỏng real-time |
| **Why** | Ngăn chặn look-ahead bias — chiến lược chỉ nhìn thấy dữ liệu đã xảy ra, không thấy tương lai |
| **Where** | `hybrid_backtester.py` dòng 56–141 |
| **When** | Được khởi tạo đầu mỗi event-driven backtest. `update_bars()` được gọi mỗi iteration của main loop |
| **Who** | Được tạo bởi `_run_event_driven_backtest()`. Được sử dụng bởi Strategy, Portfolio, ExecutionHandler |
| **How** | Lưu `symbol_data` dạng dict `{symbol: DataFrame}`. Dùng Python iterator (`iterrows()`) để drip-feed từng bar. Mỗi bar mới được append vào `latest_symbol_data`. Khi hết data → set `continue_backtest = False` |

**Bảng phương thức:**

| Phương thức | Input | Output | Mô tả |
|---|---|---|---|
| `_prepare_frame()` | `pd.DataFrame` | `pd.DataFrame` (OHLCV chuẩn hóa) | Chuẩn hóa cột, xử lý DatetimeIndex, fill missing OHLCV, validate `close` column |
| `update_bars()` | — | `MarketEvent` → Queue | Lấy bar tiếp theo từ mỗi symbol iterator, push `MarketEvent`. Hết data → dừng backtest |
| `get_latest_bars()` | `symbol`, `num_bars` | `List[Tuple[Timestamp, Series]]` | Trả về N bar gần nhất đã được phát. Dùng slicing `[-num_bars:]` |
| `get_latest_bar_datetime()` | `symbol` | `pd.Timestamp` | Timestamp của bar mới nhất |
| `get_latest_bar_value()` | `symbol`, `val_type` | `float` | Giá trị OHLCV cụ thể của bar mới nhất |

**Bảng xử lý `_prepare_frame()` — Quá trình chuẩn hóa DataFrame:**

| Bước | Logic | Fallback |
|---|---|---|
| 1. Chuẩn hóa tên cột | `str(col).strip().lower()` | — |
| 2. Xử lý index | Tìm cột datetime/date/timestamp/time → `set_index()` | Raise `ValueError` nếu không tìm thấy |
| 3. Cột `close` | Ưu tiên `close`, nếu không có → dùng `adj_close` | Raise `ValueError` nếu cả hai đều thiếu |
| 4. Cột `open/high/low` | Kiểm tra tồn tại | Copy từ `close` nếu thiếu |
| 5. Cột `volume` | Kiểm tra tồn tại | Mặc định `0.0` |
| 6. Sắp xếp | `sort_index()` theo thời gian | — |

---

### 4.2 Tầng 2 — Strategy & Portfolio Layer

#### 4.2.1 `Strategy` (Abstract Base)

| 5W1H | Chi tiết |
|---|---|
| **What** | Abstract base class định nghĩa interface cho mọi strategy |
| **Why** | Cho phép mở rộng hệ thống bằng cách thêm strategy mới mà không sửa code cũ (Open/Closed Principle) |
| **Where** | `hybrid_backtester.py` dòng 144–146 |
| **When** | Không bao giờ được khởi tạo trực tiếp. Là contract cho các subclass |
| **Who** | `EventDrivenBacktester` gọi `calculate_signals()` trên instance con |
| **How** | Chỉ có 1 abstract method `calculate_signals(event)` → raise `NotImplementedError` |

---

#### 4.2.2 `MovingAverageCrossStrategy`

| 5W1H | Chi tiết |
|---|---|
| **What** | Strategy cụ thể: phát tín hiệu LONG/EXIT dựa trên giao cắt (crossover) giữa EMA ngắn và EMA dài |
| **Why** | Là chiến lược trend-following kinh điển, dễ hiểu, phổ biến trong backtesting research |
| **Where** | `hybrid_backtester.py` dòng 149–200 |
| **When** | Được kích hoạt mỗi khi nhận `MarketEvent`. Chỉ phát signal khi có đủ `long_window` bar |
| **Who** | `EventDrivenBacktester` gọi `calculate_signals()`. Output signal được `Portfolio` tiêu thụ |
| **How** | Tính EMA(short) và EMA(long) trên close prices bằng `ewm(span=...)`. So sánh → chuyển state machine `OUT ↔ LONG`. Chi tiết bên dưới |

**Bảng State Machine:**

| Trạng thái hiện tại | Điều kiện | Tín hiệu phát ra | Trạng thái mới |
|---|---|---|---|
| `OUT` | `EMA_short > EMA_long` | `LONG` (strength=1.0) | `LONG` |
| `LONG` | `EMA_short < EMA_long` | `EXIT` (strength=1.0) | `OUT` |
| `OUT` | `EMA_short < EMA_long` | _(không phát)_ | `OUT` |
| `LONG` | `EMA_short > EMA_long` | _(không phát)_ | `LONG` |

**Thuật toán `calculate_signals()` — Step-by-step:**

| Bước | Hành động |
|---|---|
| 1 | Kiểm tra `event.type == MARKET`. Nếu không → return ngay |
| 2 | Duyệt từng `symbol` trong `symbol_list` |
| 3 | Lấy `long_window` bar gần nhất. Nếu chưa đủ → skip |
| 4 | Trích chuỗi close prices → tính `short_ema = ewm(span=short_window).mean().iloc[-1]` |
| 5 | Tính `long_ema = ewm(span=long_window).mean().iloc[-1]` |
| 6 | So sánh EMA + trạng thái hiện tại → quyết định phát `SignalEvent` hay không |

---

#### 4.2.3 `Portfolio`

| 5W1H | Chi tiết |
|---|---|
| **What** | Trung tâm quản lý vốn: theo dõi positions (số lượng cổ phiếu), holdings (giá trị tiền), sinh order, xử lý fill, tính equity curve |
| **Why** | Là lớp trung gian giữa Strategy (ra tín hiệu) và Execution (thực thi). Đảm bảo đúng về mặt kế toán (accounting correctness) |
| **Where** | `hybrid_backtester.py` dòng 203–385 |
| **When** | Được gọi ở 3 event types: `MARKET` (update_timeindex), `SIGNAL` (update_signal → generate order), `FILL` (update_fill) |
| **Who** | `EventDrivenBacktester` là caller chính. Strategy cung cấp input (signal), ExecutionHandler cung cấp fill |
| **How** | Duy trì 2 track song song: `positions` (số lượng shares) và `holdings` (giá trị dollar). Snapshot mỗi bar. Cuối backtest → build equity curve DataFrame |

**Bảng Cấu trúc Dữ liệu Nội bộ:**

| Thuộc tính | Kiểu | Mô tả |
|---|---|---|
| `current_positions` | `Dict[str, int]` | Số cổ phiếu đang nắm giữ hiện tại, theo từng symbol |
| `all_positions` | `List[Dict]` | Lịch sử snapshot position tại mỗi bar (có `datetime` key) |
| `current_holdings` | `Dict[str, float]` | Giá trị tiền hiện tại: mỗi symbol + `cash` + `commission` + `total` |
| `all_holdings` | `List[Dict]` | Lịch sử snapshot holdings tại mỗi bar |
| `equity_curve` | `Optional[pd.DataFrame]` | DataFrame cuối cùng chứa `returns`, `equity_curve`, `drawdown` |

**Bảng phương thức Portfolio:**

| Phương thức | Trigger Event | Logic chính |
|---|---|---|
| `update_timeindex()` | `MARKET` | Snapshot positions + tính market value holdings = Σ(position × close_price) + cash |
| `update_signal()` | `SIGNAL` | Gọi `generate_order_from_signal()` → nếu có order → put vào Queue |
| `generate_order_from_signal()` | — (internal) | Mapping: LONG→BUY(order_size), SHORT→SELL(order_size), EXIT→đóng position hiện tại |
| `update_fill()` | `FILL` | BUY: +qty, cash−=(cost×qty+comm). SELL: −qty, cash+=(cost×qty−comm). Cộng dồn commission |
| `create_equity_curve_dataframe()` | Cuối backtest | Tạo DataFrame từ `all_holdings` → tính `returns`, `equity_curve`, `drawdown` |
| `output_summary_stats()` | Cuối backtest | Tính: total_return, sharpe(√252), max_drawdown, max_drawdown_duration, trade_events, ending_capital |

**Logic `generate_order_from_signal()` — Quyết định Order:**

| Signal Type | Position hiện tại | Order được tạo |
|---|---|---|
| `LONG` | `== 0` | BUY `order_size` shares (MKT) |
| `LONG` | `!= 0` | `None` (đã trong vị thế) |
| `SHORT` | `== 0` | SELL `order_size` shares (MKT) |
| `SHORT` | `!= 0` | `None` (đã trong vị thế) |
| `EXIT` | `> 0` | SELL `abs(qty)` shares (MKT) |
| `EXIT` | `< 0` | BUY `abs(qty)` shares (MKT) |
| `EXIT` | `== 0` | `None` (không có gì để đóng) |

---

#### 4.2.4 `SimulatedExecutionHandler`

| 5W1H | Chi tiết |
|---|---|
| **What** | Giả lập thực thi lệnh giao dịch: áp dụng slippage và commission lên mỗi order |
| **Why** | Mô phỏng chi phí thực tế của giao dịch (transaction cost modeling), tránh kết quả backtest quá lạc quan |
| **Where** | `hybrid_backtester.py` dòng 388–423 |
| **When** | Được gọi khi có `OrderEvent` trong Queue |
| **Who** | `EventDrivenBacktester` gọi `execute_order()`. Output (`FillEvent`) được `Portfolio` tiêu thụ |
| **How** | Lấy `close` price → áp slippage (BUY: +bps, SELL: −bps) → tính commission → push `FillEvent` |

**Bảng tham số mô phỏng:**

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `commission_per_share` | `0.005` | Phí hoa hồng trên mỗi cổ phiếu ($0.005/share) |
| `min_commission` | `1.0` | Phí tối thiểu mỗi lệnh ($1.00) |
| `slippage_bps` | `1.0` | Trượt giá tính theo basis points (1 bps = 0.01%) |
| `exchange` | `"SIM"` | Tên sàn giả lập |

**Công thức tính Slippage:**

```
slipped_price = last_close × (1 + side × slippage_bps / 10000)

    side = +1 (BUY)  → giá bị đẩy lên (mua đắt hơn)
    side = -1 (SELL) → giá bị đẩy xuống (bán rẻ hơn)
```

**Công thức tính Commission:**

```
commission = max(min_commission, commission_per_share × quantity)
```

---

#### 4.2.5 `EventDrivenBacktester` (Orchestrator)

| 5W1H | Chi tiết |
|---|---|
| **What** | Orchestrator chạy vòng lặp chính (main event loop) của event-driven backtest |
| **Why** | Ghép nối 4 component (DataHandler, Strategy, Portfolio, Execution) thành pipeline hoàn chỉnh |
| **Where** | `hybrid_backtester.py` dòng 426–463 |
| **When** | Được gọi bởi `_run_event_driven_backtest()` mỗi khi cần validate trên tập test |
| **Who** | `run_hybrid_pipeline()` và `run_walk_forward_optimization()` (gián tiếp qua `_run_event_driven_backtest`) |
| **How** | 2 vòng lặp lồng nhau: outer = drip-feed bar, inner = drain event queue. Dispatch dựa trên `event.type` |

**Pseudocode vòng lặp chính:**

```
WHILE data_handler.continue_backtest:
    data_handler.update_bars()           ← push MarketEvent
    
    WHILE queue NOT empty:
        event = queue.get()
        
        SWITCH event.type:
            MARKET → strategy.calculate_signals(event)
                     portfolio.update_timeindex(event)
            SIGNAL → portfolio.update_signal(event)    ← có thể push OrderEvent
            ORDER  → execution.execute_order(event)    ← push FillEvent  
            FILL   → portfolio.update_fill(event)

RETURN portfolio.output_summary_stats(), portfolio.create_equity_curve_dataframe()
```

---

### 4.3 Tầng 3 — Pipeline & Orchestration Layer

#### 4.3.1 `compute_performance_metrics()`

| 5W1H | Chi tiết |
|---|---|
| **What** | Hàm tiện ích tính 7 chỉ số hiệu suất từ chuỗi returns |
| **Why** | Cung cấp bộ metrics chuẩn hóa cho cả vectorized lẫn event-driven engine |
| **Where** | `hybrid_backtester.py` dòng 466–512 |
| **When** | Sau mỗi backtest (vectorized hoặc aggregate walk-forward) |
| **Who** | `vectorized_ma_backtest()`, `run_walk_forward_optimization()` |
| **How** | Nhận `pd.Series` returns → tính toán thuần pandas/numpy. Annualization factor mặc định = 252 (trading days) |

**Bảng 7 Metrics được tính:**

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| `total_return` | `equity[-1] - 1.0` | Tổng lợi nhuận tích lũy (%) |
| `cagr` | `equity[-1]^(1/years) - 1` | Compound Annual Growth Rate |
| `sharpe` | `√252 × mean(returns) / std(returns)` | Risk-adjusted return (annualized) |
| `sortino` | `√252 × mean(returns) / std(negative_returns)` | Sharpe chỉ tính downside risk |
| `max_drawdown` | `min(equity / cummax - 1)` | Sụt giảm tối đa từ đỉnh |
| `volatility` | `std(returns) × √252` | Biến động niên hóa |
| `win_rate` | `count(returns > 0) / total` | Tỷ lệ bar có lợi nhuận dương |

---

#### 4.3.2 `vectorized_ma_backtest()`

| 5W1H | Chi tiết |
|---|---|
| **What** | Backtest chiến lược MA crossover bằng vectorized operations (toàn bộ tính toán trên array, không loop bar-by-bar) |
| **Why** | Tốc độ cực nhanh (100–1000x so với event-driven), phù hợp cho grid search nhiều tham số |
| **Where** | `hybrid_backtester.py` dòng 515–559 |
| **When** | Được gọi trong grid search (train) và validation (test OOS) |
| **Who** | `vectorized_grid_search()`, `run_hybrid_pipeline()`, `run_walk_forward_optimization()` |
| **How** | `ewm(span=...).mean()` → binary signal → `shift(1)` → strategy_returns = position × market_returns − transaction_cost |

**Luồng xử lý Vectorized:**

```
close_prices
    │
    ├──► fast_ma = ewm(span=short_window).mean()
    ├──► slow_ma = ewm(span=long_window).mean()
    │
    ▼
raw_signal = (fast_ma > slow_ma).astype(float)    [1.0 = LONG, 0.0 = OUT]
    │
    ▼
position = raw_signal.shift(1).fillna(0)           [tránh look-ahead bias]
    │
    ├──► market_returns = prices.pct_change()
    ├──► turnover = |Δposition|
    ├──► transaction_cost = turnover × (tc_bps / 10000)
    │
    ▼
strategy_returns = position × market_returns − transaction_cost
    │
    ▼
compute_performance_metrics(strategy_returns)
```

---

#### 4.3.3 `vectorized_grid_search()`

| 5W1H | Chi tiết |
|---|---|
| **What** | Brute-force tìm kiếm tổ hợp `(short_window, long_window)` tốt nhất trên tập train |
| **Why** | Tối ưu hóa siêu tham số — tìm cặp EMA cho Sharpe ratio cao nhất |
| **Where** | `hybrid_backtester.py` dòng 562–588 |
| **When** | Phase 1 của cả 2 mode (single và wfo) |
| **Who** | `run_hybrid_pipeline()`, `run_walk_forward_optimization()` |
| **How** | `itertools.product(short_windows, long_windows)` → bỏ qua `short >= long` → gọi `vectorized_ma_backtest()` cho mỗi cặp → sort theo Sharpe ↓, total_return ↓ |

**Bảng tham số mặc định Grid Search:**

| Tham số | Giá trị mặc định | Số lượng |
|---|---|---|
| `short_windows` | `[5, 10, 20, 30]` | 4 |
| `long_windows` | `[50, 100, 150, 200]` | 4 |
| **Tổng tổ hợp hợp lệ** | Tất cả cặp `short < long` | **16 cặp** |
| Tiêu chí chọn | `sort_values(["sharpe", "total_return"], ascending=False)` | Top 1 |

---

#### 4.3.4 `run_hybrid_pipeline()` (Mode: `single`)

| 5W1H | Chi tiết |
|---|---|
| **What** | Pipeline chính cho mode single-split: chia train/test → grid search → validate bằng cả 2 engine |
| **Why** | Quy trình chuẩn giúp tránh overfitting: optimize trên train, validate trên test |
| **Where** | `hybrid_backtester.py` dòng 644–694 |
| **When** | Khi user chạy với `--mode single` (mặc định) |
| **Who** | `main()` gọi trực tiếp |
| **How** | Xem pipeline flow bên dưới |

**Pipeline Flow:**

| Bước | Hàm được gọi | Input | Output |
|---|---|---|---|
| 1 | `train_test_split_bars()` | `price_data`, `train_ratio=0.7` | `train_data`, `test_data` |
| 2 | `vectorized_grid_search()` | `train_data["close"]`, windows | `leaderboard` (DataFrame sorted by Sharpe) |
| 3 | Extract best | `leaderboard.iloc[0]` | `best_short`, `best_long` |
| 4 | `vectorized_ma_backtest()` | `test_data["close"]`, best params | `vector_oos_metrics`, `vector_oos_frame` |
| 5 | `_run_event_driven_backtest()` | `test_data`, best params, capital | `event_summary`, `event_equity` |

**Output Dictionary:**

| Key | Type | Mô tả |
|---|---|---|
| `best_params` | `Dict[str, int]` | `{"short_window": X, "long_window": Y}` |
| `vectorized_leaderboard` | `pd.DataFrame` | Bảng xếp hạng tất cả tổ hợp tham số (train set) |
| `vectorized_oos_metrics` | `Dict[str, float]` | 7 metrics + extra trên test set (vectorized) |
| `vectorized_oos_frame` | `pd.DataFrame` | Chi tiết: price, MA, position, returns |
| `event_driven_summary` | `Dict[str, float]` | Metrics từ event-driven engine (test set) |
| `event_equity_curve` | `pd.DataFrame` | Equity curve từ event-driven engine |

---

#### 4.3.5 `generate_walk_forward_splits()`

| 5W1H | Chi tiết |
|---|---|
| **What** | Sinh danh sách các fold (train_start, train_end, test_start, test_end) cho walk-forward optimization |
| **Why** | Walk-forward tránh overfitting tốt hơn single split: re-optimize tham số theo thời gian, mô phỏng thực tế hơn |
| **Where** | `hybrid_backtester.py` dòng 697–726 |
| **When** | Đầu quá trình walk-forward optimization |
| **Who** | `run_walk_forward_optimization()` |
| **How** | Sliding window với 2 chế độ: `expanding` (train luôn bắt đầu từ bar 0) hoặc `rolling` (train cố định kích thước) |

**Minh họa 2 chế độ Window:**

```
EXPANDING (expanding_window=True):
═══════════════════════════════════════════════
Fold 1: [████ TRAIN ████][▓▓ TEST ▓▓]
Fold 2: [██████ TRAIN ██████][▓▓ TEST ▓▓]
Fold 3: [████████ TRAIN ████████][▓▓ TEST ▓▓]

ROLLING (expanding_window=False):
═══════════════════════════════════════════════
Fold 1: [████ TRAIN ████][▓▓ TEST ▓▓]
Fold 2:       [████ TRAIN ████][▓▓ TEST ▓▓]
Fold 3:             [████ TRAIN ████][▓▓ TEST ▓▓]
```

**Bảng tham số mặc định WFO:**

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `train_size_bars` | 504 | ≈ 2 năm trading days |
| `test_size_bars` | 126 | ≈ 6 tháng trading days |
| `step_size_bars` | 126 | Bước nhảy giữa các fold = 6 tháng |
| `expanding_window` | `True` | Train window mở rộng dần |

---

#### 4.3.6 `run_walk_forward_optimization()` (Mode: `wfo`)

| 5W1H | Chi tiết |
|---|---|
| **What** | Pipeline walk-forward: lặp qua nhiều fold, mỗi fold re-optimize trên train → validate trên test, cuối cùng aggregate |
| **Why** | Gold standard cho backtesting: tránh curve-fitting, mô phỏng production workflow (re-calibrate chiến lược mỗi N tháng) |
| **Where** | `hybrid_backtester.py` dòng 729–863 |
| **When** | Khi user chạy với `--mode wfo` |
| **Who** | `main()` gọi trực tiếp |
| **How** | Xem flow bên dưới |

**Walk-Forward Pipeline Flow:**

| Bước | Hành động | Chi tiết |
|---|---|---|
| 1 | Validate inputs | Kiểm tra `train_size >= max_long + 2`, `test_size >= max_long + 2` |
| 2 | `generate_walk_forward_splits()` | Sinh danh sách fold indices |
| 3 | **FOR** mỗi fold | Slice `train_data`, `test_data` theo indices |
| 3a | → `vectorized_grid_search()` trên train | Tìm best `(short, long)` cho fold này |
| 3b | → `vectorized_ma_backtest()` trên test | Tính vectorized OOS metrics |
| 3c | → `_run_event_driven_backtest()` trên test | Tính event-driven OOS metrics |
| 3d | → Collect returns | Append `strategy_returns` và `event_returns` |
| 4 | Aggregate | `pd.concat()` all return segments → `compute_performance_metrics()` |

**Output Dictionary:**

| Key | Type | Mô tả |
|---|---|---|
| `fold_results` | `pd.DataFrame` | Chi tiết mỗi fold: dates, best params, metrics cả 2 engine |
| `vectorized_aggregate_metrics` | `Dict[str, float]` | Metrics tổng hợp vectorized (tất cả fold concat) |
| `event_aggregate_metrics` | `Dict[str, float]` | Metrics tổng hợp event-driven + `ending_capital` |
| `vectorized_oos_returns` | `pd.DataFrame` | Chuỗi returns vectorized concat |
| `event_oos_frame` | `pd.DataFrame` | Chuỗi returns + equity curve event-driven concat |

---

### 4.4 Tầng Phụ trợ — Utility Functions

#### 4.4.1 `load_ohlcv_csv()`

| 5W1H | Chi tiết |
|---|---|
| **What** | Đọc file CSV chứa dữ liệu OHLCV, chuẩn hóa format giống `_prepare_frame()` |
| **Why** | Entry point cho dữ liệu thật từ nguồn ngoài (Yahoo Finance, broker export...) |
| **Where** | `hybrid_backtester.py` dòng 866–894 |
| **When** | Khi user cung cấp `--csv path/to/data.csv` |
| **Who** | `main()` |
| **How** | `pd.read_csv()` → normalize columns → find datetime col → set_index → validate & fill missing cols |

---

#### 4.4.2 `generate_synthetic_ohlcv()`

| 5W1H | Chi tiết |
|---|---|
| **What** | Sinh dữ liệu OHLCV giả lập có trend + cycle + noise, deterministic (seeded) |
| **Why** | Cho phép test/demo mà không cần dữ liệu thật. Mặc định khi không có `--csv` |
| **Where** | `hybrid_backtester.py` dòng 897–917 |
| **When** | Khi `--csv` không được cung cấp (mặc định) |
| **Who** | `main()` |
| **How** | `log_returns = trend(linear) + cycle(sin) + noise(normal)` → `close = 100 × exp(cumsum(log_returns))` → sinh OHLCV |

**Bảng cấu trúc Synthetic Data:**

| Thành phần | Công thức | Mô tả |
|---|---|---|
| Trend | `linspace(0.0001, 0.0003, periods)` | Drift tăng dần nhẹ |
| Cycle | `sin(linspace(0, 12, periods)) × 0.0008` | Chu kỳ sin mô phỏng market cycle |
| Noise | `normal(0, 0.01, periods)` | Random noise mô phỏng biến động |
| Close | `100 × exp(cumsum(log_returns))` | Geometric Brownian Motion (GBM) đơn giản hóa |
| Open | `close.shift(1)` | Close của bar trước |
| High | `max(open, close) + |normal| × close × 0.004` | Spread ngẫu nhiên phía trên |
| Low | `min(open, close) − |normal| × close × 0.004` | Spread ngẫu nhiên phía dưới |
| Volume | `randint(100000, 800000)` | Volume ngẫu nhiên |

---

#### 4.4.3 `train_test_split_bars()`

| 5W1H | Chi tiết |
|---|---|
| **What** | Chia DataFrame thành 2 phần train/test theo tỷ lệ, giữ thứ tự thời gian |
| **Why** | Time-series split — không shuffle (đúng cách cho financial data) |
| **Where** | `hybrid_backtester.py` dòng 591–599 |
| **When** | Đầu `run_hybrid_pipeline()` mode single |
| **Who** | `run_hybrid_pipeline()` |
| **How** | `split_idx = int(len × ratio)` → `iloc[:split_idx]`, `iloc[split_idx:]` |

---

## 5. Use Cases

### 5.1 Use Case 1: Single-Split Backtest (Mặc định)

| Bước | Actor | Hành động | System Response |
|---|---|---|---|
| 1 | User | `python hybrid_backtester.py` | Load synthetic data (1400 bars) |
| 2 | System | `train_test_split_bars(ratio=0.7)` | 980 bars train, 420 bars test |
| 3 | System | `vectorized_grid_search()` trên train | Leaderboard 16 tổ hợp, sort by Sharpe |
| 4 | System | Pick `best_params` từ top-1 | `{"short_window": X, "long_window": Y}` |
| 5 | System | `vectorized_ma_backtest()` trên test | OOS metrics (vectorized) |
| 6 | System | `_run_event_driven_backtest()` trên test | OOS metrics (event-driven, có slippage+commission) |
| 7 | System | `print_metrics()` | In ra console |
| 8 | User (opt) | Thêm `--export-prefix results` | Xuất `results_leaderboard.csv`, `results_vectorized_oos.csv`, `results_event_equity_curve.csv` |

### 5.2 Use Case 2: Walk-Forward Optimization

| Bước | Actor | Hành động | System Response |
|---|---|---|---|
| 1 | User | `python hybrid_backtester.py --mode wfo` | Load data (1400 bars) |
| 2 | System | `generate_walk_forward_splits()` | N fold với expanding window |
| 3 | System | FOR mỗi fold: grid search → backtest (2 engine) | Collect fold-level results |
| 4 | System | Concat returns → `compute_performance_metrics()` | Aggregate metrics |
| 5 | System | Print fold table + aggregate metrics | Console output |
| 6 | User (opt) | Thêm `--export-prefix wfo` | Xuất 3 CSV files |

### 5.3 Use Case 3: Dữ liệu thật từ CSV

| Bước | Actor | Hành động | System Response |
|---|---|---|---|
| 1 | User | `python hybrid_backtester.py --csv data.csv --symbol AAPL` | `load_ohlcv_csv()` |
| 2 | System | Validate CSV: datetime col, close col, fill missing | Chuẩn hóa OHLCV DataFrame |
| 3 | System | Chạy pipeline (single hoặc wfo) | Tùy `--mode` |

---

## 6. Bảng Tổng hợp CLI Arguments

| Argument | Type | Mặc định | Mô tả | Mode |
|---|---|---|---|---|
| `--csv` | `str` | `None` | Đường dẫn CSV OHLCV. Nếu bỏ qua → dùng synthetic | Cả hai |
| `--mode` | `str` | `"single"` | `single` hoặc `wfo` | — |
| `--symbol` | `str` | `"ASSET"` | Mã chứng khoán | Cả hai |
| `--periods` | `int` | `1400` | Số bar cho synthetic data | Cả hai |
| `--short-windows` | `str` | `"5,10,20,30"` | Danh sách fast MA windows | Cả hai |
| `--long-windows` | `str` | `"50,100,150,200"` | Danh sách slow MA windows | Cả hai |
| `--train-ratio` | `float` | `0.7` | Tỷ lệ train/test | `single` |
| `--wfo-train-bars` | `int` | `504` | Bars train mỗi fold | `wfo` |
| `--wfo-test-bars` | `int` | `126` | Bars test mỗi fold | `wfo` |
| `--wfo-step-bars` | `int` | `126` | Bước nhảy giữa fold | `wfo` |
| `--wfo-window-type` | `str` | `"expanding"` | `expanding` hoặc `rolling` | `wfo` |
| `--tc-bps` | `float` | `1.0` | Chi phí giao dịch (basis points) | Cả hai |
| `--initial-capital` | `float` | `100000.0` | Vốn khởi đầu ($) | Cả hai |
| `--order-size` | `int` | `100` | Số cổ phiếu mỗi lệnh | Cả hai |
| `--top-k` | `int` | `10` | Hiển thị top K kết quả | `single` |
| `--export-prefix` | `str` | `None` | Tiền tố tên file CSV xuất | Cả hai |

---

## 7. Bảng Tổng hợp File

| File | Dòng | Vai trò | Dependencies |
|---|---|---|---|
| `hybrid_backtester.py` | 1137 | Toàn bộ source code (monolith) | `numpy`, `pandas`, `argparse`, `dataclasses`, `enum`, `itertools`, `queue`, `typing` |
| `requirements.txt` | 2 | Khai báo dependencies Python | — |

**Bảng Chi tiết Class/Function trong `hybrid_backtester.py`:**

| # | Tên | Loại | Dòng | Layer | Vai trò tóm tắt |
|---|---|---|---|---|---|
| 1 | `EventType` | Enum | 14–18 | Event | 4 loại event: MARKET, SIGNAL, ORDER, FILL |
| 2 | `MarketEvent` | Dataclass | 22–23 | Event | Trigger khi có bar mới |
| 3 | `SignalEvent` | Dataclass | 27–32 | Event | Tín hiệu giao dịch (LONG/SHORT/EXIT) |
| 4 | `OrderEvent` | Dataclass | 36–41 | Event | Lệnh giao dịch (BUY/SELL, MKT) |
| 5 | `FillEvent` | Dataclass | 45–53 | Event | Xác nhận thực thi (price, commission) |
| 6 | `HistoricDataHandler` | Class | 56–141 | Data | Drip-feed OHLCV, ngăn look-ahead bias |
| 7 | `Strategy` | Class (ABC) | 144–146 | Strategy | Abstract base cho chiến lược |
| 8 | `MovingAverageCrossStrategy` | Class | 149–200 | Strategy | EMA crossover, state machine OUT↔LONG |
| 9 | `Portfolio` | Class | 203–385 | Portfolio | Quản lý position/holdings/equity curve |
| 10 | `SimulatedExecutionHandler` | Class | 388–423 | Execution | Slippage + commission simulation |
| 11 | `EventDrivenBacktester` | Class | 426–463 | Orchestrator | Main event loop dispatcher |
| 12 | `compute_performance_metrics()` | Function | 466–512 | Utility | 7 chỉ số hiệu suất chuẩn |
| 13 | `vectorized_ma_backtest()` | Function | 515–559 | Vectorized | Backtest MA nhanh (array ops) |
| 14 | `vectorized_grid_search()` | Function | 562–588 | Vectorized | Brute-force tối ưu tham số |
| 15 | `train_test_split_bars()` | Function | 591–599 | Utility | Time-series train/test split |
| 16 | `_run_event_driven_backtest()` | Function | 602–641 | Orchestrator | Factory + runner cho event-driven test |
| 17 | `run_hybrid_pipeline()` | Function | 644–694 | Pipeline | Mode single: grid search → validate |
| 18 | `generate_walk_forward_splits()` | Function | 697–726 | Utility | Sinh fold indices cho WFO |
| 19 | `run_walk_forward_optimization()` | Function | 729–863 | Pipeline | Mode wfo: multi-fold + aggregate |
| 20 | `load_ohlcv_csv()` | Function | 866–894 | I/O | Đọc + chuẩn hóa CSV |
| 21 | `generate_synthetic_ohlcv()` | Function | 897–917 | I/O | Sinh dữ liệu giả lập (seeded) |
| 22 | `parse_int_list()` | Function | 920–925 | Utility | Parse "5,10,20" → [5,10,20] |
| 23 | `print_metrics()` | Function | 928–934 | I/O | In metrics ra console |
| 24 | `build_parser()` | Function | 937–1036 | CLI | Xây dựng argparse parser |
| 25 | `main()` | Function | 1039–1133 | Entry | Entry point, dispatch theo mode |

---

## 8. Business Rules

### 8.1 Ràng buộc Validation

| Rule ID | Mô tả | Nơi enforce | Exception |
|---|---|---|---|
| BR-01 | `short_window < long_window` | `MovingAverageCrossStrategy.__init__`, `vectorized_ma_backtest()`, `vectorized_grid_search()` | `ValueError` |
| BR-02 | Data phải có cột `close` (hoặc `adj_close`) | `_prepare_frame()`, `load_ohlcv_csv()` | `ValueError` |
| BR-03 | Data phải có DatetimeIndex hoặc cột datetime/date/timestamp/time | `_prepare_frame()`, `load_ohlcv_csv()` | `ValueError` |
| BR-04 | `symbol_data` không được rỗng | `HistoricDataHandler.__init__` | `ValueError` |
| BR-05 | `train_ratio` phải trong khoảng `(0.1, 0.95)` | `train_test_split_bars()` | `ValueError` |
| BR-06 | `train/test/step bars` phải là số nguyên dương | `generate_walk_forward_splits()` | `ValueError` |
| BR-07 | `total_bars >= train_size + test_size` | `generate_walk_forward_splits()` | `ValueError` |
| BR-08 | `train_size_bars >= max(long_windows) + 2` | `run_walk_forward_optimization()` | `ValueError` |
| BR-09 | `test_size_bars >= max(long_windows) + 2` | `run_walk_forward_optimization()` | `ValueError` |
| BR-10 | Data đủ bars: `len >= long_window + 2` | `vectorized_ma_backtest()` | `ValueError` |

### 8.2 Quy tắc Giao dịch

| Rule ID | Mô tả | Logic |
|---|---|---|
| TR-01 | Chỉ mở LONG khi position = 0 | `generate_order_from_signal()`: LONG signal + qty=0 → BUY |
| TR-02 | Chỉ mở SHORT khi position = 0 | `generate_order_from_signal()`: SHORT signal + qty=0 → SELL |
| TR-03 | EXIT đóng toàn bộ position hiện tại | `generate_order_from_signal()`: EXIT → SELL/BUY |abs(qty)| |
| TR-04 | Mọi order đều là Market Order (MKT) | Hardcoded `order_type="MKT"` |
| TR-05 | Order size cố định | `order_size` parameter, mặc định 100 shares |
| TR-06 | Vectorized shift(1) tránh look-ahead bias | `position = raw_signal.shift(1)`: trade hôm sau mới có effect |

### 8.3 Quy tắc Chi phí

| Rule ID | Mô tả | Công thức |
|---|---|---|
| TC-01 | Commission tối thiểu $1.00 | `max(1.0, 0.005 × qty)` |
| TC-02 | Slippage đối xứng | BUY: `price × (1 + bps/10000)`, SELL: `price × (1 − bps/10000)` |
| TC-03 | Vectorized TC = turnover × bps | `|Δposition| × tc_bps / 10000` |

### 8.4 So sánh 2 Engine

| Tiêu chí | Vectorized Engine | Event-Driven Engine |
|---|---|---|
| **Tốc độ** | Rất nhanh (numpy/pandas array ops) | Chậm hơn (loop bar-by-bar) |
| **Mục đích** | Grid search tối ưu tham số | Validate kết quả sát thực tế |
| **Slippage** | Ước lượng đơn giản (turnover × bps) | Mô phỏng per-order (BUY đắt, SELL rẻ) |
| **Commission** | Gộp vào transaction cost | Per-share + minimum commission |
| **Position sizing** | Binary (0 hoặc 1 — fully in/out) | Fixed quantity (e.g., 100 shares) |
| **Cash tracking** | Không | Có (initial_capital, cash balance) |
| **Equity curve** | Cumulative product of returns | Dollar-based portfolio valuation |
| **Vai trò trong pipeline** | Phase 1: Tìm tham số | Phase 2: Xác nhận tham số |

---

## 9. Dependency Graph

```
main()
 ├── build_parser()
 ├── parse_int_list()
 ├── load_ohlcv_csv()  ──OR──  generate_synthetic_ohlcv()
 │
 ├── [mode=single] run_hybrid_pipeline()
 │    ├── train_test_split_bars()
 │    ├── vectorized_grid_search()
 │    │    └── vectorized_ma_backtest()
 │    │         └── compute_performance_metrics()
 │    ├── vectorized_ma_backtest()            [OOS]
 │    └── _run_event_driven_backtest()        [OOS]
 │         ├── HistoricDataHandler
 │         ├── MovingAverageCrossStrategy
 │         ├── Portfolio
 │         ├── SimulatedExecutionHandler
 │         └── EventDrivenBacktester.run()
 │
 ├── [mode=wfo] run_walk_forward_optimization()
 │    ├── generate_walk_forward_splits()
 │    ├── FOR each fold:
 │    │    ├── vectorized_grid_search()
 │    │    ├── vectorized_ma_backtest()       [OOS]
 │    │    └── _run_event_driven_backtest()   [OOS]
 │    └── compute_performance_metrics()       [aggregate]
 │
 └── print_metrics()
```

---

> **Ghi chú cuối:** Hệ thống được thiết kế theo monolith single-file, phù hợp cho nghiên cứu cá nhân và rapid prototyping. Để scale lên production, cần tách thành modules riêng (events, data, strategy, portfolio, execution, pipeline) và thêm unit tests, logging, multi-asset support, và async execution.
