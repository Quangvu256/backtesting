/**
 * App Controller — Main orchestrator cho Stock Dashboard
 */
const App = {
    currentSymbol: null,
    currentDays: 90,
    watchlistData: [],

    async init() {
        console.log('[App] Initializing XGBoost Quant Dashboard...');

        // Load watchlist
        this.watchlistData = await API.getWatchlist();
        if (!this.watchlistData || this.watchlistData.length === 0) {
            console.error('[App] Failed to load watchlist');
            return;
        }

        // Default: first symbol
        this.currentSymbol = this.watchlistData[0].symbol;

        // Render watchlist tabs & cards
        this._renderWatchlist();

        // Init chart
        Charts.initMainChart('main-chart');

        // Load data for current symbol
        await this.selectSymbol(this.currentSymbol);

        // Load model health
        this._loadModelHealth();

        // Bind events
        this._bindEvents();

        console.log('[App] Dashboard ready.');
    },

    /** Select a symbol and load all its data */
    async selectSymbol(symbol) {
        this.currentSymbol = symbol;

        // Update hero from cached watchlist data
        const symbolData = this.watchlistData.find(w => w.symbol === symbol) || {};
        Components.updateHero(symbolData);

        // Update active states in tabs & cards
        this._updateActiveStates(symbol);

        // Load all data in parallel
        const [ohlcv, signals, indicators, news, predictions, futurePredictions, backtest] = await Promise.all([
            API.getOHLCV(symbol, this.currentDays),
            API.getSignals(symbol),
            API.getIndicators(symbol),
            API.getNews(symbol),
            API.getPredictions(symbol),
            API.getFuturePredictions(symbol),
            API.getBacktest(symbol),
        ]);

        // Render chart
        if (ohlcv && ohlcv.length > 0) {
            Charts.setOHLCVData(ohlcv);
            // Signal markers
            if (signals) {
                const ohlcvTimes = ohlcv.map(d => d.time);
                Charts.setSignalMarkers(signals, ohlcvTimes);
            }
        }
        Charts.setFutureProjection(futurePredictions);

        // Render panels
        Components.renderTechnicalPanel(indicators);
        Components.renderSignalHistory(signals);
        Components.renderNewsFeed(news);
        Components.renderModelBreakdown(predictions);
        Components.renderFuturePredictions(futurePredictions);
        Components.renderBacktestPanel(backtest);
    },

    /** Change timeframe and reload chart */
    async changeTimeframe(days) {
        this.currentDays = days;

        // Update active button
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.days) === days);
        });

        // Reload OHLCV + signals
        const [ohlcv, signals, futurePredictions] = await Promise.all([
            API.getOHLCV(this.currentSymbol, days),
            API.getSignals(this.currentSymbol),
            API.getFuturePredictions(this.currentSymbol),
        ]);

        if (ohlcv && ohlcv.length > 0) {
            Charts.setOHLCVData(ohlcv);
            if (signals) {
                Charts.setSignalMarkers(signals, ohlcv.map(d => d.time));
            }
        }
        Charts.setFutureProjection(futurePredictions);
    },

    /** Refresh all data */
    async refresh() {
        const btn = document.getElementById('refresh-btn');
        btn.classList.add('spinning');

        this.watchlistData = await API.getWatchlist();
        this._renderWatchlist();
        await this.selectSymbol(this.currentSymbol);
        this._loadModelHealth();

        btn.classList.remove('spinning');
    },

    // ─── PRIVATE HELPERS ────────────────────────────────────────

    _renderWatchlist() {
        Components.renderWatchlistTabs(this.watchlistData, this.currentSymbol, (sym) => this.selectSymbol(sym));
        Components.renderWatchlistCards(this.watchlistData, this.currentSymbol, (sym) => this.selectSymbol(sym));
    },

    _updateActiveStates(symbol) {
        // Tabs
        document.querySelectorAll('.wl-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.symbol === symbol);
        });
        // Cards
        document.querySelectorAll('.wl-card').forEach(card => {
            card.classList.toggle('active', card.dataset.symbol === symbol);
        });
    },

    async _loadModelHealth() {
        const modelsStatus = await API.getModelsStatus();
        Components.renderModelHealth(modelsStatus);
    },

    _bindEvents() {
        // Timeframe buttons
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.changeTimeframe(parseInt(btn.dataset.days));
            });
        });

        // Refresh button
        document.getElementById('refresh-btn')?.addEventListener('click', () => {
            this.refresh();
        });

        // Panel tabs
        document.querySelectorAll('.panel-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Toggle active tab
                document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Show/hide panels
                const panelName = tab.dataset.panel;
                document.querySelectorAll('.panel-content').forEach(p => {
                    p.style.display = 'none';
                });
                const target = document.getElementById(`panel-${panelName}`);
                if (target) target.style.display = 'block';

                // Re-render backtest chart if needed (resize issue)
                if (panelName === 'backtest' && Charts.backtestChart) {
                    Charts.backtestChart.timeScale().fitContent();
                }
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // R = refresh
            if (e.key === 'r' && !e.ctrlKey && !e.metaKey && document.activeElement === document.body) {
                this.refresh();
            }
            // 1-5 = select symbol
            const num = parseInt(e.key);
            if (num >= 1 && num <= this.watchlistData.length && document.activeElement === document.body) {
                this.selectSymbol(this.watchlistData[num - 1].symbol);
            }
        });
    }
};

// ─── BOOT ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => App.init());
