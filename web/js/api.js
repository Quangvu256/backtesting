/**
 * API Client — Fetch wrapper cho Stock Dashboard
 */
const API = {
    BASE: '',

    async _fetch(url) {
        try {
            const res = await fetch(this.BASE + url);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (e) {
            console.error(`[API] ${url}:`, e.message);
            return null;
        }
    },

    /** Danh sách watchlist + giá + tín hiệu */
    getWatchlist() {
        return this._fetch('/api/watchlist');
    },

    /** Dữ liệu OHLCV cho chart */
    getOHLCV(symbol, days = 90) {
        return this._fetch(`/api/stock/${symbol}/ohlcv?days=${days}`);
    },

    /** Lịch sử tín hiệu Ensemble */
    getSignals(symbol, limit = 200) {
        return this._fetch(`/api/stock/${symbol}/signals?limit=${limit}`);
    },

    /** Indicators kỹ thuật mới nhất */
    getIndicators(symbol) {
        return this._fetch(`/api/stock/${symbol}/indicators`);
    },

    /** Tin tức + sentiment */
    getNews(symbol, limit = 20) {
        return this._fetch(`/api/stock/${symbol}/news?limit=${limit}`);
    },

    /** Chi tiết dự đoán 6 model */
    getPredictions(symbol) {
        return this._fetch(`/api/stock/${symbol}/predictions`);
    },

    /** Future projected prices from latest model horizons */
    getFuturePredictions(symbol) {
        return this._fetch(`/api/stock/${symbol}/future-predictions`);
    },

    /** Backtest equity curve */
    getBacktest(symbol) {
        return this._fetch(`/api/stock/${symbol}/backtest`);
    },

    /** Model status tổng quan */
    getModelsStatus() {
        return this._fetch('/api/models/status');
    }
};
