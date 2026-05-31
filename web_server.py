"""
Flask API Server cho Stock Dashboard.
Đọc dữ liệu trực tiếp từ SQLite DB hiện có (7 bảng) và phục vụ REST API + static files.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from database import DatabaseManager
from config import WATCHLIST, DATABASE_PATH
from forecasting import build_future_predictions

logger = logging.getLogger("WebServer")

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

db = DatabaseManager()

# ─── STATIC FILES ───────────────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

# ─── API: WATCHLIST OVERVIEW ────────────────────────────────────────────────────

@app.route('/api/watchlist')
def api_watchlist():
    """Trả danh sách mã + giá mới nhất + tín hiệu Ensemble mới nhất."""
    result = []
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            for symbol in WATCHLIST:
                item = {"symbol": symbol, "price": None, "change": None, "change_pct": None, "signal": "N/A", "score": 0}

                # Giá 2 phiên gần nhất
                cursor.execute(
                    "SELECT close FROM ohlcv_data WHERE symbol = ? ORDER BY timestamp DESC LIMIT 2",
                    (symbol,)
                )
                rows = cursor.fetchall()
                if rows:
                    item["price"] = rows[0]["close"]
                    if len(rows) > 1:
                        prev = rows[1]["close"]
                        item["change"] = round(item["price"] - prev, 2)
                        item["change_pct"] = round((item["change"] / prev) * 100, 2) if prev else 0

                # Tín hiệu Ensemble mới nhất
                cursor.execute(
                    "SELECT signal, ensemble_score FROM ensemble_decisions WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                    (symbol,)
                )
                sig_row = cursor.fetchone()
                if sig_row:
                    item["signal"] = sig_row["signal"]
                    item["score"] = round(sig_row["ensemble_score"], 4)

                result.append(item)
    except Exception as e:
        logger.error(f"Error in api_watchlist: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

# ─── API: OHLCV DATA ───────────────────────────────────────────────────────────

@app.route('/api/stock/<symbol>/ohlcv')
def api_ohlcv(symbol):
    """Trả dữ liệu OHLCV cho biểu đồ nến. Query param: days (default 90)."""
    symbol = symbol.upper()
    days = request.args.get('days', 90, type=int)
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT timestamp, open, high, low, close, volume
                   FROM ohlcv_data WHERE symbol = ? AND timestamp >= ?
                   ORDER BY timestamp ASC""",
                (symbol, start_date)
            )
            rows = cursor.fetchall()
            data = []
            for r in rows:
                ts = r["timestamp"]
                # Lightweight Charts cần Unix timestamp (seconds)
                try:
                    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    dt = datetime.strptime(ts, '%Y-%m-%d')
                data.append({
                    "time": int(dt.timestamp()),
                    "open": r["open"],
                    "high": r["high"],
                    "low": r["low"],
                    "close": r["close"],
                    "volume": r["volume"]
                })
            return jsonify(data)
    except Exception as e:
        logger.error(f"Error in api_ohlcv: {e}")
        return jsonify({"error": str(e)}), 500

# ─── API: ENSEMBLE SIGNALS ──────────────────────────────────────────────────────

@app.route('/api/stock/<symbol>/signals')
def api_signals(symbol):
    """Trả lịch sử tín hiệu Ensemble."""
    symbol = symbol.upper()
    limit = request.args.get('limit', 200, type=int)

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT timestamp, signal, ensemble_score, sentiment_impact, decision_metadata
                   FROM ensemble_decisions WHERE symbol = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (symbol, limit)
            )
            rows = cursor.fetchall()
            data = []
            for r in rows:
                ts = r["timestamp"]
                try:
                    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    dt = datetime.strptime(ts, '%Y-%m-%d')
                meta = {}
                if r["decision_metadata"]:
                    try:
                        meta = json.loads(r["decision_metadata"])
                    except json.JSONDecodeError:
                        pass
                data.append({
                    "time": int(dt.timestamp()),
                    "timestamp": ts,
                    "signal": r["signal"],
                    "score": round(r["ensemble_score"], 4),
                    "sentiment_impact": round(r["sentiment_impact"], 4) if r["sentiment_impact"] else 0,
                    "metadata": meta
                })
            return jsonify(data)
    except Exception as e:
        logger.error(f"Error in api_signals: {e}")
        return jsonify({"error": str(e)}), 500

# ─── API: TECHNICAL INDICATORS ──────────────────────────────────────────────────

@app.route('/api/stock/<symbol>/indicators')
def api_indicators(symbol):
    """Trả features kỹ thuật mới nhất (RSI, MACD, BB, returns, volatility)."""
    symbol = symbol.upper()

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT timestamp, feature_data
                   FROM features WHERE symbol = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol,)
            )
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": "No features data"}), 404

            features = json.loads(row["feature_data"])
            features["timestamp"] = row["timestamp"]
            return jsonify(features)
    except Exception as e:
        logger.error(f"Error in api_indicators: {e}")
        return jsonify({"error": str(e)}), 500

# ─── API: NEWS & SENTIMENT ──────────────────────────────────────────────────────

@app.route('/api/stock/<symbol>/news')
def api_news(symbol):
    """Trả tin tức + sentiment score gần nhất."""
    symbol = symbol.upper()
    limit = request.args.get('limit', 20, type=int)

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT title, source, pub_date, sentiment_label, sentiment_score, sentiment_reason, url
                   FROM news_articles WHERE symbol = ?
                   ORDER BY pub_date DESC LIMIT ?""",
                (symbol, limit)
            )
            rows = cursor.fetchall()
            data = [dict(r) for r in rows]
            return jsonify(data)
    except Exception as e:
        logger.error(f"Error in api_news: {e}")
        return jsonify({"error": str(e)}), 500

# ─── API: MODEL PREDICTIONS ─────────────────────────────────────────────────────

@app.route('/api/stock/<symbol>/predictions')
def api_predictions(symbol):
    """Trả chi tiết dự đoán mới nhất của 6 model."""
    symbol = symbol.upper()

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            # Lấy ensemble decision mới nhất chứa metadata chi tiết
            cursor.execute(
                """SELECT timestamp, signal, ensemble_score, sentiment_impact, decision_metadata
                   FROM ensemble_decisions WHERE symbol = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol,)
            )
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": "No predictions available"}), 404

            result = {
                "timestamp": row["timestamp"],
                "signal": row["signal"],
                "ensemble_score": round(row["ensemble_score"], 4),
                "sentiment_impact": round(row["sentiment_impact"], 4) if row["sentiment_impact"] else 0,
                "models": {}
            }

            if row["decision_metadata"]:
                try:
                    meta = json.loads(row["decision_metadata"])
                    result["models"] = meta.get("predictions", {})
                    result["weighted_score_cls"] = meta.get("weighted_score_cls", 0)
                    result["weighted_score_reg"] = meta.get("weighted_score_reg", 0)
                    result["sentiment_score"] = meta.get("sentiment_score", 0)
                except json.JSONDecodeError:
                    pass

            return jsonify(result)
    except Exception as e:
        logger.error(f"Error in api_predictions: {e}")
        return jsonify({"error": str(e)}), 500

# ─── API: BACKTEST EQUITY CURVE ──────────────────────────────────────────────────

@app.route('/api/stock/<symbol>/future-predictions')
def api_future_predictions(symbol):
    """Return future projected prices for the latest 1D/5D/20D model horizons."""
    symbol = symbol.upper()
    forecast = build_future_predictions(symbol, db)
    if not forecast:
        return jsonify({"error": "No future predictions available", "available": False}), 404
    return jsonify(forecast)

@app.route('/api/stock/<symbol>/backtest')
def api_backtest(symbol):
    """Trả equity curve từ CSV nếu có."""
    symbol = symbol.upper()
    csv_path = os.path.join("data", f"{symbol}_ensemble_equity.csv")

    if not os.path.exists(csv_path):
        return jsonify({"error": "No backtest data available", "available": False}), 404

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Xác định cột timestamp và equity
        time_col = df.columns[0]
        data = []
        for _, row in df.iterrows():
            ts = row[time_col]
            try:
                dt = datetime.strptime(str(ts), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    dt = datetime.strptime(str(ts), '%Y-%m-%d')
                except ValueError:
                    continue

            entry = {"time": int(dt.timestamp())}
            # Thêm tất cả cột số
            for col in df.columns[1:]:
                try:
                    entry[col] = float(row[col])
                except (ValueError, TypeError):
                    pass
            data.append(entry)

        return jsonify({"available": True, "data": data, "columns": list(df.columns[1:])})
    except Exception as e:
        logger.error(f"Error in api_backtest: {e}")
        return jsonify({"error": str(e)}), 500

# ─── API: MODEL STATUS ──────────────────────────────────────────────────────────

@app.route('/api/models/status')
def api_models_status():
    """Trả trạng thái huấn luyện tất cả models."""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT model_id, symbol, horizon, mode, trained_at, train_metrics, test_metrics
                   FROM model_runs ORDER BY symbol, horizon, mode"""
            )
            rows = cursor.fetchall()
            data = []
            for r in rows:
                item = dict(r)
                # Parse JSON metrics
                for key in ["train_metrics", "test_metrics"]:
                    if item.get(key):
                        try:
                            item[key] = json.loads(item[key])
                        except json.JSONDecodeError:
                            pass
                data.append(item)
            return jsonify(data)
    except Exception as e:
        logger.error(f"Error in api_models_status: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  📊 Stock Dashboard Server")
    print(f"  🌐 http://localhost:5000")
    print(f"  📁 Database: {DATABASE_PATH}")
    print(f"  📋 Watchlist: {WATCHLIST}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
