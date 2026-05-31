/**
 * Charts Module — TradingView Lightweight Charts wrapper
 */
const Charts = {
    mainChart: null,
    candleSeries: null,
    volumeSeries: null,
    smaSeries: null,
    futureSeries: null,
    backtestChart: null,
    backtestLineSeries: null,

    /** Khởi tạo main candlestick chart */
    initMainChart(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Cleanup previous chart
        if (this.mainChart) {
            this.mainChart.remove();
            this.mainChart = null;
        }

        this.mainChart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: container.clientHeight || 400,
            layout: {
                background: { type: 'solid', color: 'transparent' },
                textColor: '#94a3b8',
                fontSize: 11,
                fontFamily: "'Inter', sans-serif",
            },
            grid: {
                vertLines: { color: 'rgba(255,255,255,0.03)' },
                horzLines: { color: 'rgba(255,255,255,0.03)' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {
                    color: 'rgba(59, 130, 246, 0.3)',
                    labelBackgroundColor: '#3b82f6',
                },
                horzLine: {
                    color: 'rgba(59, 130, 246, 0.3)',
                    labelBackgroundColor: '#3b82f6',
                },
            },
            rightPriceScale: {
                borderColor: 'rgba(255,255,255,0.06)',
                scaleMargins: { top: 0.1, bottom: 0.25 },
            },
            timeScale: {
                borderColor: 'rgba(255,255,255,0.06)',
                timeVisible: false,
                dayVisible: true,
            },
            handleScroll: { vertTouchDrag: false },
        });

        // Candlestick series
        this.candleSeries = this.mainChart.addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#ef4444',
            borderDownColor: '#ef4444',
            borderUpColor: '#10b981',
            wickDownColor: '#ef4444',
            wickUpColor: '#10b981',
        });

        // Volume series (histogram overlay)
        this.volumeSeries = this.mainChart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
        });
        this.mainChart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.85, bottom: 0 },
        });

        // SMA overlay
        this.smaSeries = this.mainChart.addLineSeries({
            color: 'rgba(59, 130, 246, 0.5)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Solid,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });

        this.futureSeries = this.mainChart.addLineSeries({
            color: '#f59e0b',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: true,
            crosshairMarkerVisible: true,
        });

        // Responsive resize
        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (this.mainChart) {
                    this.mainChart.applyOptions({ width, height });
                }
            }
        });
        resizeObserver.observe(container);
    },

    /** Set OHLCV data trên chart */
    setOHLCVData(data) {
        if (!this.candleSeries || !data || data.length === 0) return;

        this.candleSeries.setData(data);

        // Volume bars
        const volumeData = data.map(d => ({
            time: d.time,
            value: d.volume,
            color: d.close >= d.open
                ? 'rgba(16, 185, 129, 0.15)'
                : 'rgba(239, 68, 68, 0.15)',
        }));
        this.volumeSeries.setData(volumeData);

        // SMA 20
        const smaData = this._calcSMA(data.map(d => ({ time: d.time, value: d.close })), 20);
        this.smaSeries.setData(smaData);

        this.mainChart.timeScale().fitContent();
    },

    /** Render future forecast path on the main chart */
    setFutureProjection(forecastData) {
        if (!this.futureSeries) return;

        if (!forecastData || !forecastData.forecast || forecastData.forecast.length === 0) {
            this.futureSeries.setData([]);
            return;
        }

        const series = [];
        const baseTime = this._parseTimeToUnix(forecastData.price_timestamp);
        if (baseTime && forecastData.base_price != null) {
            series.push({ time: baseTime, value: forecastData.base_price });
        }

        for (const item of forecastData.forecast) {
            if (item.target_time && item.projected_price != null) {
                series.push({ time: item.target_time, value: item.projected_price });
            }
        }

        series.sort((a, b) => a.time - b.time);
        this.futureSeries.setData(series);
    },

    /** Add signal markers trên chart */
    setSignalMarkers(signals, ohlcvTimes) {
        if (!this.candleSeries || !signals) return;

        const timeSet = new Set(ohlcvTimes);
        const markers = [];

        for (const sig of signals) {
            if (!timeSet.has(sig.time)) continue;

            if (sig.signal === 'STRONG_BUY' || sig.signal === 'BUY') {
                markers.push({
                    time: sig.time,
                    position: 'belowBar',
                    color: sig.signal === 'STRONG_BUY' ? '#34d399' : '#10b981',
                    shape: 'arrowUp',
                    text: sig.signal === 'STRONG_BUY' ? 'S-BUY' : 'BUY',
                });
            } else if (sig.signal === 'STRONG_SELL' || sig.signal === 'SELL') {
                markers.push({
                    time: sig.time,
                    position: 'aboveBar',
                    color: sig.signal === 'STRONG_SELL' ? '#f87171' : '#ef4444',
                    shape: 'arrowDown',
                    text: sig.signal === 'STRONG_SELL' ? 'S-SELL' : 'SELL',
                });
            }
        }

        // Sort by time ascending (required by Lightweight Charts)
        markers.sort((a, b) => a.time - b.time);
        this.candleSeries.setMarkers(markers);
    },

    /** Khởi tạo và render backtest equity chart */
    renderBacktestChart(containerId, data, columns) {
        const container = document.getElementById(containerId);
        if (!container || !data || data.length === 0) return;

        if (this.backtestChart) {
            this.backtestChart.remove();
            this.backtestChart = null;
        }

        this.backtestChart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 280,
            layout: {
                background: { type: 'solid', color: 'transparent' },
                textColor: '#94a3b8',
                fontSize: 11,
                fontFamily: "'Inter', sans-serif",
            },
            grid: {
                vertLines: { color: 'rgba(255,255,255,0.03)' },
                horzLines: { color: 'rgba(255,255,255,0.03)' },
            },
            rightPriceScale: {
                borderColor: 'rgba(255,255,255,0.06)',
            },
            timeScale: {
                borderColor: 'rgba(255,255,255,0.06)',
            },
        });

        // Find equity-like column
        const equityCol = columns.find(c =>
            c.toLowerCase().includes('equity') ||
            c.toLowerCase().includes('total') ||
            c.toLowerCase().includes('capital')
        ) || columns[0];

        if (equityCol) {
            this.backtestLineSeries = this.backtestChart.addAreaSeries({
                topColor: 'rgba(59, 130, 246, 0.3)',
                bottomColor: 'rgba(59, 130, 246, 0.02)',
                lineColor: '#3b82f6',
                lineWidth: 2,
            });
            const lineData = data
                .filter(d => d[equityCol] != null)
                .map(d => ({ time: d.time, value: d[equityCol] }));
            this.backtestLineSeries.setData(lineData);
        }

        this.backtestChart.timeScale().fitContent();

        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                if (this.backtestChart) {
                    this.backtestChart.applyOptions({ width: entry.contentRect.width });
                }
            }
        });
        resizeObserver.observe(container);
    },

    /** Simple Moving Average calculator */
    _calcSMA(data, period) {
        const result = [];
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += data[i - j].value;
            }
            result.push({ time: data[i].time, value: sum / period });
        }
        return result;
    },

    _parseTimeToUnix(value) {
        if (!value) return null;
        const normalized = String(value).includes('T')
            ? String(value)
            : String(value).replace(' ', 'T');
        const ts = Date.parse(normalized);
        return Number.isFinite(ts) ? Math.floor(ts / 1000) : null;
    }
};
