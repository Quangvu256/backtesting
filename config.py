import os
import sys
import io
import logging

# Thiết lập UTF-8 encoding cho stdout/stderr để tránh crash Unicode tiếng Việt trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables từ file .env
load_dotenv()

# Các đường dẫn thư mục cơ sở
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Đảm bảo các thư mục tồn tại
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Đường dẫn Database
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATA_DIR / "backtesting_system.db"))

# Danh sách mã cổ phiếu theo dõi (Watchlist)
WATCHLIST_STR = os.getenv("WATCHLIST", "VNM,FPT,HPG,VIC,VCB")
WATCHLIST = [symbol.strip().upper() for symbol in WATCHLIST_STR.split(",") if symbol.strip()]

# Cấu hình phần cứng (GPU hoặc CPU)
DEVICE = os.getenv("DEVICE", "GPU").upper()

# Google Gemini API Key (legacy, kept for backward compatibility)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# LMStudio Local Server Config (OpenAI-compatible API)
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://100.121.12.126:1235/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "")  # Để trống = dùng model đang load trên LMStudio
LLM_BACKEND = os.getenv("LLM_BACKEND", "lmstudio")  # 'lmstudio' hoặc 'gemini'

# Cấu hình học máy (Machine Learning Configs)
HORIZONS = [1, 5, 20]  # Các khung thời gian dự báo: 1 ngày, 5 ngày, 20 ngày
ENSEMBLE_WEIGHTS = {
    1: 0.50,   # Trọng số dự báo 1 ngày
    5: 0.30,   # Trọng số dự báo 5 ngày
    20: 0.20   # Trọng số dự báo 20 ngày
}
MODE_WEIGHTS = {
    "classification": 0.60, # Trọng số của mô hình phân loại (UP/DOWN/FLAT)
    "regression": 0.40      # Trọng số của mô hình hồi quy (% return)
}
DIRECTION_THRESHOLD = 0.005  # Ngưỡng ±0.5% để phân loại xu hướng UP/DOWN/FLAT

# Khởi tạo Logging cho toàn hệ thống
log_file = LOGS_DIR / "system.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SystemConfig")
logger.info(f"[FACT] Hệ thống đã tải cấu hình thành công.")
logger.info(f"Watchlist: {WATCHLIST}")
logger.info(f"Database Path: {DATABASE_PATH}")
logger.info(f"Target Device: {DEVICE}")
