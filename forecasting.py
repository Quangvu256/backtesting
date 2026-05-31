import json
import logging
from datetime import datetime, timedelta

from config import DIRECTION_THRESHOLD, HORIZONS
from database import DatabaseManager

logger = logging.getLogger("Forecasting")


def _parse_timestamp(value):
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value}")


def _add_business_days(start_dt, days):
    target = start_dt
    remaining = int(days)
    while remaining > 0:
        target += timedelta(days=1)
        if target.weekday() < 5:
            remaining -= 1
    return target


def _direction_from_return(expected_return):
    if expected_return is None:
        return "UNKNOWN"
    if expected_return > DIRECTION_THRESHOLD:
        return "UP"
    if expected_return < -DIRECTION_THRESHOLD:
        return "DOWN"
    return "FLAT"


def _direction_from_probs(cls_pred):
    if not cls_pred:
        return None, None

    probs = {
        "UP": float(cls_pred.get("p_up", 0.0)),
        "DOWN": float(cls_pred.get("p_down", 0.0)),
        "FLAT": float(cls_pred.get("p_flat", 0.0)),
    }
    direction = max(probs, key=probs.get)
    return direction, probs[direction]


def build_future_predictions(symbol, db_manager=None):
    """
    Build a forward-looking forecast from the latest ensemble decision.

    The existing models already predict 1D/5D/20D returns/classes. This helper
    turns those outputs into target dates and projected prices without creating
    synthetic OHLCV rows or retraining models.
    """
    symbol = symbol.upper()
    db = db_manager if db_manager else DatabaseManager()

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT timestamp, close
                   FROM ohlcv_data
                   WHERE symbol = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol,),
            )
            price_row = cursor.fetchone()

            cursor.execute(
                """SELECT timestamp, signal, ensemble_score, sentiment_impact, decision_metadata
                   FROM ensemble_decisions
                   WHERE symbol = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol,),
            )
            decision_row = cursor.fetchone()

            fallback_rows = {}
            if decision_row:
                cursor.execute(
                    """SELECT horizon, mode, prediction_value, prediction_class, confidence
                       FROM predictions
                       WHERE symbol = ? AND timestamp = ?
                       ORDER BY horizon, mode, predicted_at DESC""",
                    (symbol, decision_row["timestamp"]),
                )
                for row in cursor.fetchall():
                    key = (int(row["horizon"]), row["mode"])
                    if key not in fallback_rows:
                        fallback_rows[key] = row
    except Exception as e:
        logger.error(f"[ERROR] Failed to build future predictions for {symbol}: {e}", exc_info=True)
        return None

    if not price_row or not decision_row:
        return None

    base_price = float(price_row["close"])
    price_timestamp = price_row["timestamp"]
    decision_timestamp = decision_row["timestamp"]
    base_dt = _parse_timestamp(price_timestamp)
    decision_dt = _parse_timestamp(decision_timestamp)

    metadata = {}
    if decision_row["decision_metadata"]:
        try:
            metadata = json.loads(decision_row["decision_metadata"])
        except json.JSONDecodeError:
            metadata = {}

    model_predictions = metadata.get("predictions", {})
    sentiment_impact = float(decision_row["sentiment_impact"] or 0.0)
    ensemble_score = float(decision_row["ensemble_score"] or 0.0)
    final_score = ensemble_score + sentiment_impact

    forecast = []
    for horizon in HORIZONS:
        reg_pred = model_predictions.get(f"reg_{horizon}d") or model_predictions.get(f"reg_{horizon}")
        cls_pred = model_predictions.get(f"cls_{horizon}d") or model_predictions.get(f"cls_{horizon}")

        expected_return = None
        reg_score = None
        if reg_pred:
            expected_return = float(reg_pred.get("expected_return", 0.0))
            reg_score = float(reg_pred.get("score", 0.0))
        else:
            fallback_reg = fallback_rows.get((int(horizon), "regression"))
            if fallback_reg:
                expected_return = float(fallback_reg["prediction_value"])

        cls_direction, cls_confidence = _direction_from_probs(cls_pred)
        cls_score = float(cls_pred.get("score", 0.0)) if cls_pred else None
        if cls_direction is None:
            fallback_cls = fallback_rows.get((int(horizon), "classification"))
            if fallback_cls:
                cls_direction = fallback_cls["prediction_class"]
                cls_confidence = float(fallback_cls["confidence"] or 0.0)

        return_direction = _direction_from_return(expected_return)
        direction = return_direction if return_direction != "UNKNOWN" else (cls_direction or "UNKNOWN")
        projected_price = None
        if expected_return is not None:
            projected_price = base_price * (1.0 + expected_return)

        target_dt = _add_business_days(base_dt, horizon)
        forecast.append(
            {
                "horizon": int(horizon),
                "target_timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "target_time": int(target_dt.timestamp()),
                "base_price": base_price,
                "projected_price": projected_price,
                "expected_return": expected_return,
                "direction": direction,
                "classification_direction": cls_direction,
                "classification_confidence": cls_confidence,
                "regression_score": reg_score,
                "classification_score": cls_score,
            }
        )

    return {
        "symbol": symbol,
        "available": True,
        "price_timestamp": price_timestamp,
        "decision_timestamp": decision_timestamp,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_price": base_price,
        "signal": decision_row["signal"],
        "ensemble_score": ensemble_score,
        "sentiment_impact": sentiment_impact,
        "final_score": final_score,
        "data_lag_days": max((decision_dt.date() - base_dt.date()).days, 0),
        "forecast": forecast,
    }
