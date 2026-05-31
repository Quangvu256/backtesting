import os
import json
import logging
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from scipy.stats import loguniform, randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight
from config import MODELS_DIR, HORIZONS, ENSEMBLE_WEIGHTS, MODE_WEIGHTS, DIRECTION_THRESHOLD, DEVICE
from database import DatabaseManager
from feature_engineering import FeatureExtractor

logger = logging.getLogger("MLPipeline")

try:
    import optuna
except ImportError:
    optuna = None

XGB_SEARCH_ITERATIONS = 24
XGB_RANDOM_STATE = 42
XGB_EARLY_STOPPING_ROUNDS = 50
FINAL_EVAL_FRACTION = 0.15
EMBARGO_BARS = 5

XGB_CLASSIFICATION_PARAM_DIST = {
    'max_depth': randint(2, 6),
    'learning_rate': loguniform(1e-3, 2e-1),
    'n_estimators': randint(400, 2001),
    'subsample': uniform(0.65, 0.30),
    'colsample_bytree': uniform(0.65, 0.30),
    'min_child_weight': loguniform(1.0, 20.0),
    'gamma': loguniform(1e-8, 0.5),
    'reg_alpha': loguniform(1e-8, 1.0),
    'reg_lambda': loguniform(0.5, 20.0)
}

XGB_REGRESSION_PARAM_DIST = {
    'max_depth': randint(2, 6),
    'learning_rate': loguniform(1e-3, 2e-1),
    'n_estimators': randint(400, 2001),
    'subsample': uniform(0.65, 0.30),
    'colsample_bytree': uniform(0.65, 0.30),
    'min_child_weight': loguniform(1.0, 20.0),
    'gamma': loguniform(1e-8, 0.5),
    'reg_alpha': loguniform(1e-8, 1.0),
    'reg_lambda': loguniform(0.5, 20.0)
}

INTEGER_XGB_PARAMS = {"max_depth", "n_estimators"}


class PurgedEmbargoKFold:
    """
    Purged & embargoed splitter for overlapping financial labels.
    Samples are assumed ordered by time. Label interval for row i is [i, i+horizon].
    """
    def __init__(self, n_splits=4, horizon=5, embargo=EMBARGO_BARS):
        self.n_splits = int(n_splits)
        self.horizon = int(horizon)
        self.embargo = int(embargo)

    def split(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_start = current
            test_stop = current + fold_size
            current = test_stop
            if fold_size == 0:
                continue

            test_idx = indices[test_start:test_stop]
            test_interval_start = test_start
            test_interval_end = min(n_samples - 1, test_stop - 1 + self.horizon)
            embargo_end = min(n_samples, test_stop + self.embargo)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_stop] = False

            label_starts = indices
            label_ends = np.minimum(indices + self.horizon, n_samples - 1)
            overlaps = (label_starts <= test_interval_end) & (label_ends >= test_interval_start)
            train_mask[overlaps] = False
            train_mask[test_stop:embargo_end] = False

            train_idx = indices[train_mask]
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx

def detect_xgboost_device():
    """
    Tự động phát hiện thiết bị tốt nhất để huấn luyện XGBoost (GPU vs CPU).
    Nếu config chỉ định rõ CPU, bỏ qua kiểm tra GPU để tránh bị treo.
    """
    if DEVICE == "CPU":
        logger.info("[FACT] Cấu hình yêu cầu sử dụng CPU. Sẽ sử dụng device='cpu'.")
        return "cpu"
        
    X_toy = np.random.randn(100, 5)
    y_toy = np.random.randint(0, 2, 100)
    
    # 1. Thử thiết bị 'cuda' (NVIDIA GPU)
    try:
        model = xgb.XGBClassifier(n_estimators=2, device='cuda', tree_method='hist')
        model.fit(X_toy, y_toy)
        logger.info("[FACT] Tự động phát hiện: GPU NVIDIA CUDA hoạt động tốt. Sẽ sử dụng device='cuda'.")
        return 'cuda'
    except Exception:
        pass
        
    # 2. Thử thiết bị 'gpu' chung (hỗ trợ các nền tảng khác)
    try:
        model = xgb.XGBClassifier(n_estimators=2, device='gpu', tree_method='hist')
        model.fit(X_toy, y_toy)
        logger.info("[FACT] Tự động phát hiện: GPU OpenCL/Generic hoạt động tốt. Sẽ sử dụng device='gpu'.")
        return 'gpu'
    except Exception:
        pass
        
    logger.info("[FACT] Không tìm thấy thiết bị GPU tương thích hoặc thiếu driver. Fallback sử dụng CPU.")
    return 'cpu'

class MultiModelTrainer:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.feature_extractor = FeatureExtractor(self.db_manager)
        self.device = detect_xgboost_device()

    def _preprocessor(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    def _sanitize_params(self, params):
        clean = {}
        for key, value in params.items():
            if key in INTEGER_XGB_PARAMS:
                clean[key] = int(value)
            else:
                clean[key] = float(value)
        return clean

    def _build_model(self, mode, params, early_stopping=False):
        common = {
            **self._sanitize_params(params),
            "device": self.device,
            "tree_method": "hist",
            "random_state": XGB_RANDOM_STATE,
            "n_jobs": 1 if self.device in {"cuda", "gpu"} else -1,
            "verbosity": 0
        }
        if early_stopping:
            common["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS

        if mode == "classification":
            return xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                **common
            )

        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **common
        )

    def _classification_sample_weight(self, y):
        base = compute_sample_weight(class_weight="balanced", y=y)
        class_risk_weight = np.ones_like(base, dtype=float)
        class_risk_weight[np.asarray(y).astype(int) == 0] *= 0.6
        class_risk_weight[np.asarray(y).astype(int) == 1] *= 1.2
        class_risk_weight[np.asarray(y).astype(int) == 2] *= 1.4
        return base * class_risk_weight

    def _suggest_xgb_params(self, trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        }

    def _fit_model_with_fold_pipeline(self, mode, params, X_fit, y_fit, X_eval=None, y_eval=None):
        preprocessor = self._preprocessor()
        X_fit_processed = preprocessor.fit_transform(X_fit)
        fit_kwargs = {"verbose": False}

        if X_eval is not None and y_eval is not None and len(X_eval) > 0:
            X_eval_processed = preprocessor.transform(X_eval)
            fit_kwargs["eval_set"] = [(X_eval_processed, y_eval)]

        if mode == "classification":
            fit_kwargs["sample_weight"] = self._classification_sample_weight(y_fit)

        model = self._build_model(mode, params, early_stopping=X_eval is not None and y_eval is not None)
        model.fit(X_fit_processed, y_fit, **fit_kwargs)
        return preprocessor, model

    def _score_cv_fold(self, mode, model, preprocessor, X_valid, y_valid):
        X_valid_processed = preprocessor.transform(X_valid)
        y_pred = model.predict(X_valid_processed)
        if mode == "classification":
            return self._profit_aware_classification_score(y_valid, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
        directional = float(np.mean(np.sign(y_valid) == np.sign(y_pred)))
        return float(directional - rmse)

    def _profit_aware_classification_score(self, y_true, y_pred):
        """
        Utility score for Triple Barrier labels.
        0=FLAT, 1=TAKE_PROFIT/long, 2=STOP_LOSS/risk-off.
        Penalize long-when-stop-loss more heavily than missing a flat day.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        utility = np.zeros_like(y_true, dtype=float)

        utility[(y_true == y_pred) & (y_true == 1)] = 1.5
        utility[(y_true == y_pred) & (y_true == 2)] = 1.2
        utility[(y_true == y_pred) & (y_true == 0)] = 0.2
        utility[(y_true == 2) & (y_pred == 1)] = -3.0
        utility[(y_true == 1) & (y_pred == 2)] = -2.0
        utility[(y_true == 0) & (y_pred != 0)] = -0.4
        utility[(y_true != 0) & (y_pred == 0)] = -0.2

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return float(macro_f1 + np.mean(utility))

    def _optuna_search(self, mode, X_train, y_train, splitter, model_id):
        if optuna is None:
            logger.warning("[UNVERIFIED] Optuna is not installed. Falling back to randomized sampler.")
            return None, None

        sampler = optuna.samplers.TPESampler(seed=XGB_RANDOM_STATE)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            params = self._suggest_xgb_params(trial)
            fold_scores = []
            for fold_train_idx, fold_valid_idx in splitter.split(X_train):
                X_fold_train = X_train.iloc[fold_train_idx]
                X_fold_valid = X_train.iloc[fold_valid_idx]
                y_fold_train = y_train[fold_train_idx]
                y_fold_valid = y_train[fold_valid_idx]
                preprocessor, model = self._fit_model_with_fold_pipeline(
                    mode,
                    params,
                    X_fold_train,
                    y_fold_train,
                    X_fold_valid,
                    y_fold_valid
                )
                fold_scores.append(self._score_cv_fold(mode, model, preprocessor, X_fold_valid, y_fold_valid))
            return float(np.mean(fold_scores)) if fold_scores else -np.inf

        study.optimize(objective, n_trials=XGB_SEARCH_ITERATIONS, show_progress_bar=False)
        logger.info(f"[FACT] Optuna best for {model_id}: score={study.best_value:.6f}; params={study.best_params}")
        return self._sanitize_params(study.best_params), float(study.best_value)

    def _randomized_time_series_search(self, mode, param_dist, X_train, y_train, splitter):
        best_params = None
        best_score = -np.inf
        sampled_params = list(ParameterSampler(
            param_dist,
            n_iter=XGB_SEARCH_ITERATIONS,
            random_state=XGB_RANDOM_STATE
        ))

        for idx, params in enumerate(sampled_params, start=1):
            params = self._sanitize_params(params)
            fold_scores = []
            for fold_train_idx, fold_valid_idx in splitter.split(X_train):
                X_fold_train = X_train.iloc[fold_train_idx]
                X_fold_valid = X_train.iloc[fold_valid_idx]
                y_fold_train = y_train[fold_train_idx]
                y_fold_valid = y_train[fold_valid_idx]

                try:
                    preprocessor, model = self._fit_model_with_fold_pipeline(
                        mode,
                        params,
                        X_fold_train,
                        y_fold_train,
                        X_fold_valid,
                        y_fold_valid
                    )
                    fold_scores.append(self._score_cv_fold(
                        mode,
                        model,
                        preprocessor,
                        X_fold_valid,
                        y_fold_valid
                    ))
                except Exception as e:
                    logger.warning(f"[UNVERIFIED] CV fold failed for params={params}: {e}")
                    fold_scores.append(-np.inf)

            mean_score = float(np.mean(fold_scores))
            logger.info(f"[FACT] Search trial {idx}/{len(sampled_params)} mode={mode}: score={mean_score:.6f}; params={params}")
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        if best_params is None:
            raise RuntimeError("No valid hyperparameter candidate found during randomized search.")
        return best_params, best_score

    def _fit_final_pipeline(self, mode, best_params, X_train, y_train):
        eval_size = max(1, int(len(X_train) * FINAL_EVAL_FRACTION))
        if len(X_train) - eval_size < 30:
            eval_size = max(1, min(len(X_train) // 5, len(X_train) - 2))

        if eval_size > 0 and len(X_train) - eval_size >= 2:
            X_fit, X_eval = X_train.iloc[:-eval_size], X_train.iloc[-eval_size:]
            y_fit, y_eval = y_train[:-eval_size], y_train[-eval_size:]
            _, es_model = self._fit_model_with_fold_pipeline(mode, best_params, X_fit, y_fit, X_eval, y_eval)
            best_iteration = getattr(es_model, "best_iteration", None)
            if best_iteration is not None:
                best_params = {**best_params, "n_estimators": int(best_iteration) + 1}
                logger.info(f"[FACT] Early stopping selected n_estimators={best_params['n_estimators']} for final refit.")

        final_model = self._build_model(mode, best_params, early_stopping=False)
        final_pipeline = Pipeline([
            ("preprocessor", self._preprocessor()),
            ("xgb", final_model)
        ])
        fit_kwargs = {}
        if mode == "classification":
            fit_kwargs["xgb__sample_weight"] = self._classification_sample_weight(y_train)
        final_pipeline.fit(X_train, y_train, **fit_kwargs)
        return final_pipeline, best_params

    def train_and_evaluate(self, symbol):
        """
        Huấn luyện đầy đủ 6 mô hình cho một mã cổ phiếu:
        - 3 horizons (1D, 5D, 20D)
        - 2 modes (regression & classification)
        """
        logger.info(f"Bắt đầu quy trình huấn luyện đa mô hình cho {symbol}...")
        
        # 1. Lấy ma trận đặc trưng từ DB
        X, y_dict = self.feature_extractor.get_features_for_training(symbol)
        if X.empty or not y_dict:
            logger.error(f"[ERROR] Không có đủ dữ liệu đặc trưng đã gán nhãn cho {symbol} để huấn luyện.")
            return False
            
        # Chia train/test theo trục thời gian (Time-Series Split) để tránh data leakage
        n_samples = len(X)
        split_idx = int(n_samples * 0.8) # 80% train, 20% test
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        
        logger.info(f"Tổng số mẫu: {n_samples}. Tập Train: {len(X_train)} (từ {X_train.index[0]} đến {X_train.index[-1]}). Tập Test: {len(X_test)} (từ {X_test.index[0]} đến {X_test.index[-1]}).")
        
        # Purged & embargoed CV de giam leakage do overlapping outcomes.
        cv_splits = min(4, max(2, len(X_train) // 120))
        logger.info(f"[FACT] Hyperparameter search: PurgedEmbargoKFold={cv_splits}, n_iter={XGB_SEARCH_ITERATIONS}, embargo={EMBARGO_BARS}, early_stopping_rounds={XGB_EARLY_STOPPING_ROUNDS}.")
        
        # Danh sách lưu kết quả thành công
        trained_models_count = 0
        
        # Huấn luyện lần lượt 6 mô hình
        for horizon in HORIZONS:
            for mode in ['classification', 'regression']:
                model_id = f"{symbol}_{horizon}D_{mode}"
                logger.info(f"Đang tối ưu siêu tham số và huấn luyện model: {model_id}...")
                
                # Xác định nhãn target tương ứng
                target_key = f"cls_{horizon}" if mode == 'classification' else f"reg_{horizon}"
                y = y_dict[target_key]
                if mode == 'classification':
                    y = y.astype(int)
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                param_dist = XGB_CLASSIFICATION_PARAM_DIST if mode == 'classification' else XGB_REGRESSION_PARAM_DIST
                
                try:
                    splitter = PurgedEmbargoKFold(n_splits=cv_splits, horizon=horizon, embargo=EMBARGO_BARS)
                    best_params, best_cv_score = self._optuna_search(
                        mode=mode,
                        X_train=X_train,
                        y_train=y_train,
                        splitter=splitter,
                        model_id=model_id
                    )
                    if best_params is None:
                        best_params, best_cv_score = self._randomized_time_series_search(
                            mode=mode,
                            param_dist=param_dist,
                            X_train=X_train,
                            y_train=y_train,
                            splitter=splitter
                        )
                    best_model, best_params = self._fit_final_pipeline(mode, best_params, X_train, y_train)
                    logger.info(f"[FACT] Best CV score for {model_id}: {best_cv_score:.6f}; params={best_params}")
                    
                    # Đánh giá trên tập test
                    y_pred = best_model.predict(X_test)
                    
                    train_metrics = {}
                    test_metrics = {}
                    
                    if mode == 'classification':
                        train_pred = best_model.predict(X_train)
                        train_metrics = {
                            "accuracy": float(accuracy_score(y_train, train_pred)),
                            "f1_macro": float(f1_score(y_train, train_pred, average='macro', zero_division=0))
                        }
                        test_metrics = {
                            "accuracy": float(accuracy_score(y_test, y_pred)),
                            "f1_macro": float(f1_score(y_test, y_pred, average='macro', zero_division=0))
                        }
                        logger.info(f"Model {model_id} - Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_macro']:.4f}")
                    else:
                        train_pred = best_model.predict(X_train)
                        train_mse = mean_squared_error(y_train, train_pred)
                        test_mse = mean_squared_error(y_test, y_pred)
                        train_metrics = {
                            "mse": float(train_mse),
                            "rmse": float(np.sqrt(train_mse)),
                            "r2": float(r2_score(y_train, train_pred))
                        }
                        
                        # Tính toán độ chính xác hướng dự đoán (Directional Accuracy)
                        # Dự đoán đúng hướng nếu tích của y_test và y_pred dương, hoặc cả hai đều bằng 0
                        dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred))
                        
                        test_metrics = {
                            "mse": float(test_mse),
                            "rmse": float(np.sqrt(test_mse)),
                            "r2": float(r2_score(y_test, y_pred)),
                            "directional_accuracy": float(dir_acc)
                        }
                        logger.info(f"Model {model_id} - Test MSE: {test_metrics['mse']:.6f}, R2: {test_metrics['r2']:.4f}, DirAcc: {test_metrics['directional_accuracy']:.4f}")
                        
                    # Lưu model vào file dưới dạng JSON của XGBoost
                    model_filename = f"{model_id}.joblib"
                    model_path = os.path.join(MODELS_DIR, model_filename)
                    joblib.dump(best_model, model_path)
                    
                    # Lưu thông tin chạy model vào DB
                    self.save_model_run_to_db(
                        model_id=model_id,
                        symbol=symbol,
                        horizon=horizon,
                        mode=mode,
                        best_params=best_params,
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                        model_path=model_path
                    )
                    trained_models_count += 1
                    
                except Exception as e:
                    logger.error(f"[ERROR] Lỗi khi huấn luyện model {model_id}: {e}", exc_info=True)
                    
        logger.info(f"[FACT] Hoàn tất quy trình huấn luyện cho {symbol}. Đã huấn luyện thành công {trained_models_count}/6 mô hình.")
        return trained_models_count == 6

    def save_model_run_to_db(self, model_id, symbol, horizon, mode, best_params, train_metrics, test_metrics, model_path):
        """Lưu lịch sử model run và các metrics đánh giá vào DB SQLite"""
        query = """
        INSERT OR REPLACE INTO model_runs (
            model_id, symbol, horizon, mode, trained_at, best_params, train_metrics, test_metrics, model_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(query, (
                    model_id,
                    symbol,
                    int(horizon),
                    mode,
                    now,
                    json.dumps(best_params),
                    json.dumps(train_metrics),
                    json.dumps(test_metrics),
                    model_path
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Lỗi khi ghi thông tin chạy mô hình {model_id} vào DB: {e}")

class EnsembleVoter:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.feature_extractor = FeatureExtractor(self.db_manager)

    def load_model(self, model_path, mode):
        """Tai model moi dang sklearn Pipeline/joblib; van doc duoc XGBoost JSON cu."""
        if model_path.endswith(".joblib"):
            return joblib.load(model_path)

        if mode == 'classification':
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model

    def make_decision(self, symbol):
        """
        Đọc đặc trưng mới nhất của mã cổ phiếu, chạy dự báo trên toàn bộ 6 mô hình,
        tổ hợp điểm biểu quyết (Ensemble Voting) kết hợp tin tức để đưa ra quyết định giao dịch cuối cùng.
        """
        logger.info(f"Bắt đầu quá trình biểu quyết Ensemble cho mã {symbol}...")
        
        # 1. Lấy dữ liệu đặc trưng mới nhất
        latest_feat = self.feature_extractor.get_latest_features(symbol)
        if latest_feat.empty:
            logger.warning(f"Không có dữ liệu đặc trưng mới nhất cho {symbol} để dự đoán.")
            return None
            
        timestamp = latest_feat.index[0]
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # 2. Truy vấn danh sách mô hình đã lưu trong DB
        query = "SELECT horizon, mode, model_path FROM model_runs WHERE symbol = ?"
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                model_rows = cursor.fetchall()
        except Exception as e:
            logger.error(f"Lỗi khi đọc danh sách model của {symbol}: {e}")
            return None
            
        if len(model_rows) < 6:
            logger.warning(f"[UNVERIFIED] Chưa đủ 6 mô hình cho mã {symbol} trong DB. Cần huấn luyện trước.")
            return None
            
        # Lưu các dự đoán riêng lẻ để tổ hợp
        predictions_meta = {}
        weighted_score_cls = 0.0
        weighted_score_reg = 0.0
        
        for row in model_rows:
            horizon = int(row['horizon'])
            mode = row['mode']
            model_path = row['model_path']
            
            if not os.path.exists(model_path):
                logger.error(f"Không tìm thấy file mô hình tại {model_path}")
                return None
                
            try:
                # Load và predict
                model = self.load_model(model_path, mode)
                
                # Chuẩn bị dữ liệu đầu vào (bỏ cột index timestamp)
                X_pred = latest_feat.copy()
                
                if mode == 'classification':
                    # Lấy xác suất của các lớp (0: FLAT, 1: UP, 2: DOWN)
                    probs = model.predict_proba(X_pred)[0]
                    p_flat, p_up, p_down = float(probs[0]), float(probs[1]), float(probs[2])
                    
                    # Điểm xu hướng của khung thời gian horizon h: (Prob(UP) - Prob(DOWN))
                    horizon_score = p_up - p_down
                    
                    # Cộng dồn điểm có trọng số thời gian (1D: 50%, 5D: 30%, 20D: 20%)
                    weight = ENSEMBLE_WEIGHTS[horizon]
                    weighted_score_cls += weight * horizon_score
                    
                    predictions_meta[f"cls_{horizon}d"] = {
                        "p_up": p_up, "p_down": p_down, "p_flat": p_flat, "score": horizon_score
                    }
                    
                    # Lưu dự đoán đơn lẻ vào DB predictions
                    self.save_single_prediction(symbol, timestamp_str, horizon, mode, horizon_score, "UP" if p_up > max(p_down, p_flat) else "DOWN" if p_down > max(p_up, p_flat) else "FLAT", max(p_up, p_down, p_flat))
                    
                else:
                    # Lấy % return dự kiến
                    pred_ret = float(model.predict(X_pred)[0])
                    
                    # Chuẩn hóa return qua hàm tanh để đưa về khoảng [-1, 1] tương đương điểm
                    # Giả định lợi suất kỳ vọng chuẩn hóa so với ngưỡng
                    horizon_score = np.tanh(pred_ret / DIRECTION_THRESHOLD)
                    
                    weight = ENSEMBLE_WEIGHTS[horizon]
                    weighted_score_reg += weight * horizon_score
                    
                    predictions_meta[f"reg_{horizon}d"] = {
                        "expected_return": pred_ret, "score": horizon_score
                    }
                    
                    # Lưu dự đoán đơn lẻ vào DB predictions
                    self.save_single_prediction(symbol, timestamp_str, horizon, mode, pred_ret, "UP" if pred_ret > DIRECTION_THRESHOLD else "DOWN" if pred_ret < -DIRECTION_THRESHOLD else "FLAT", 1.0)
                    
            except Exception as e:
                logger.error(f"Lỗi khi chạy dự báo với model {symbol}_{horizon}D_{mode}: {e}")
                return None
                
        # 3. Tính điểm biểu quyết Ensemble tổng hợp
        # Kết hợp điểm phân loại (60%) và điểm hồi quy (40%)
        ensemble_score = (MODE_WEIGHTS['classification'] * weighted_score_cls) + (MODE_WEIGHTS['regression'] * weighted_score_reg)
        
        # 4. Tích hợp điểm số Cảm xúc tin tức tác động
        # Đọc điểm tin tức trung bình trong 24 giờ qua của mã cổ phiếu
        sentiment_score = float(latest_feat['sentiment_score'].iloc[0]) if 'sentiment_score' in latest_feat.columns else 0.0
        
        # Tác động tin tức điều chỉnh điểm số tối đa ±0.1
        sentiment_impact = 0.1 * sentiment_score
        final_score = ensemble_score + sentiment_impact
        
        # 5. Phân loại thành tín hiệu giao dịch cuối cùng
        if final_score >= 0.5:
            signal = "STRONG_BUY"
        elif final_score >= 0.15:
            signal = "BUY"
        elif final_score > -0.15:
            signal = "HOLD"
        elif final_score > -0.5:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"
            
        logger.info(f"[FACT] Quyết định Ensemble cho {symbol} tại {timestamp_str}: Tín hiệu={signal}, Điểm={final_score:.4f} (Ensemble={ensemble_score:.4f}, Sentiment Impact={sentiment_impact:.4f})")
        
        # 6. Lưu quyết định vào DB
        decision_metadata = {
            "predictions": predictions_meta,
            "weighted_score_cls": weighted_score_cls,
            "weighted_score_reg": weighted_score_reg,
            "sentiment_score": sentiment_score
        }
        
        self.save_ensemble_decision(
            symbol=symbol,
            timestamp=timestamp_str,
            signal=signal,
            ensemble_score=ensemble_score,
            sentiment_impact=sentiment_impact,
            metadata=decision_metadata
        )
        
        return {
            "symbol": symbol,
            "timestamp": timestamp_str,
            "signal": signal,
            "final_score": final_score,
            "ensemble_score": ensemble_score,
            "sentiment_impact": sentiment_impact
        }

    def save_single_prediction(self, symbol, timestamp, horizon, mode, value, pred_class, confidence):
        """Lưu dự đoán của từng mô hình đơn lẻ vào DB"""
        query = """
        INSERT OR REPLACE INTO predictions (symbol, timestamp, horizon, mode, prediction_value, prediction_class, confidence, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(query, (symbol, timestamp, horizon, mode, float(value), pred_class, float(confidence), now))
                conn.commit()
        except Exception as e:
            logger.error(f"Lỗi khi lưu dự báo đơn lẻ vào DB: {e}")

    def save_ensemble_decision(self, symbol, timestamp, signal, ensemble_score, sentiment_impact, metadata):
        """Lưu quyết định biểu quyết ensemble cuối cùng vào DB"""
        query = """
        INSERT OR REPLACE INTO ensemble_decisions (symbol, timestamp, signal, ensemble_score, sentiment_impact, decision_metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(query, (
                    symbol,
                    timestamp,
                    signal,
                    float(ensemble_score),
                    float(sentiment_impact),
                    json.dumps(metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Lỗi khi lưu quyết định Ensemble vào DB: {e}")
