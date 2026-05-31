/**
 * UI Components — Render functions cho dashboard panels
 */
const Components = {

    // ─── SIGNAL BADGE HTML ──────────────────────────────────────
    signalBadge(signal) {
        const cls = (signal || 'N/A').replace(/\s+/g, '_');
        return `<span class="signal-badge ${cls}">${signal || 'N/A'}</span>`;
    },

    signalMiniTag(signal) {
        const colors = {
            'STRONG_BUY': { bg: 'rgba(16,185,129,0.15)', color: '#34d399' },
            'BUY': { bg: 'rgba(16,185,129,0.10)', color: '#10b981' },
            'HOLD': { bg: 'rgba(100,116,139,0.10)', color: '#94a3b8' },
            'SELL': { bg: 'rgba(239,68,68,0.10)', color: '#ef4444' },
            'STRONG_SELL': { bg: 'rgba(239,68,68,0.15)', color: '#f87171' },
        };
        const s = colors[signal] || colors['HOLD'];
        return `<span class="wl-card-signal" style="background:${s.bg};color:${s.color}">${signal || 'N/A'}</span>`;
    },

    changeClass(val) {
        if (val > 0) return 'up';
        if (val < 0) return 'down';
        return '';
    },

    formatPrice(p) {
        if (p == null) return '—';
        return new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 }).format(p);
    },

    formatChange(change, pct) {
        if (change == null) return '—';
        const sign = change >= 0 ? '+' : '';
        return `${sign}${change.toFixed(2)} (${sign}${(pct || 0).toFixed(2)}%)`;
    },

    formatScore(s) {
        if (s == null) return '—';
        return (s >= 0 ? '+' : '') + s.toFixed(4);
    },

    // ─── HERO CARD ──────────────────────────────────────────────
    updateHero(data) {
        document.getElementById('hero-symbol').textContent = data.symbol || '---';
        document.getElementById('hero-price').textContent = this.formatPrice(data.price);

        const changeEl = document.getElementById('hero-change');
        changeEl.textContent = this.formatChange(data.change, data.change_pct);
        changeEl.className = 'hero-change ' + this.changeClass(data.change);

        const badgeEl = document.getElementById('signal-badge');
        badgeEl.className = 'signal-badge ' + (data.signal || '').replace(/\s+/g, '_');
        badgeEl.textContent = data.signal || 'N/A';

        document.getElementById('signal-score').textContent = `Score: ${this.formatScore(data.score)}`;
    },

    // ─── WATCHLIST CARDS ────────────────────────────────────────
    renderWatchlistCards(watchlistData, activeSymbol, onSelect) {
        const container = document.getElementById('watchlist-cards');
        if (!container) return;

        container.innerHTML = watchlistData.map(item => {
            const isActive = item.symbol === activeSymbol;
            const chgClass = this.changeClass(item.change);
            return `
                <div class="wl-card ${isActive ? 'active' : ''}" data-symbol="${item.symbol}">
                    <div class="wl-card-left">
                        <div class="wl-card-symbol">${item.symbol}</div>
                        ${this.signalMiniTag(item.signal)}
                    </div>
                    <div class="wl-card-right">
                        <div class="wl-card-price">${this.formatPrice(item.price)}</div>
                        <div class="wl-card-change ${chgClass}" style="color: var(--${chgClass === 'up' ? 'profit-green' : chgClass === 'down' ? 'loss-red' : 'text-muted'})">
                            ${this.formatChange(item.change, item.change_pct)}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Bind click events
        container.querySelectorAll('.wl-card').forEach(card => {
            card.addEventListener('click', () => {
                onSelect(card.dataset.symbol);
            });
        });
    },

    // ─── WATCHLIST TABS (Topbar) ────────────────────────────────
    renderWatchlistTabs(watchlistData, activeSymbol, onSelect) {
        const nav = document.getElementById('watchlist-tabs');
        if (!nav) return;

        nav.innerHTML = watchlistData.map(item => {
            const isActive = item.symbol === activeSymbol;
            return `<button class="wl-tab ${isActive ? 'active' : ''}" data-symbol="${item.symbol}">${item.symbol}</button>`;
        }).join('');

        nav.querySelectorAll('.wl-tab').forEach(btn => {
            btn.addEventListener('click', () => {
                onSelect(btn.dataset.symbol);
            });
        });
    },

    // ─── TECHNICAL PANEL ────────────────────────────────────────
    renderTechnicalPanel(indicators) {
        const el = document.getElementById('technical-content');
        if (!el) return;

        if (!indicators) {
            el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📉</div><div class="empty-state-text">Không có dữ liệu kỹ thuật</div></div>';
            return;
        }

        const rsi = indicators.rsi_14 || 0;
        const rsiColor = rsi > 70 ? 'var(--loss-red)' : rsi < 30 ? 'var(--profit-green)' : 'var(--accent-blue)';
        const rsiLabel = rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral';

        const macd = indicators.macd || 0;
        const macdSignal = indicators.macd_signal || 0;
        const macdHist = indicators.macd_hist || 0;
        const macdColor = macdHist > 0 ? 'var(--profit-green)' : 'var(--loss-red)';

        const bbPct = indicators.bb_pct || 0;
        const bbLabel = bbPct > 0.8 ? 'Near Upper' : bbPct < 0.2 ? 'Near Lower' : 'Middle';

        el.innerHTML = `
            <div class="tech-grid">
                <div class="tech-card">
                    <div class="tech-label">RSI (14)</div>
                    <div class="tech-value" style="color:${rsiColor}">${rsi.toFixed(1)}</div>
                    <div class="tech-sub">${rsiLabel}</div>
                    <div class="gauge-bar">
                        <div class="gauge-fill" style="width:${Math.min(rsi, 100)}%;background:${rsiColor}"></div>
                    </div>
                </div>

                <div class="tech-card">
                    <div class="tech-label">MACD</div>
                    <div class="tech-value" style="color:${macdColor}">${macd.toFixed(2)}</div>
                    <div class="tech-sub">Signal: ${macdSignal.toFixed(2)} | Hist: ${macdHist.toFixed(2)}</div>
                </div>

                <div class="tech-card">
                    <div class="tech-label">Bollinger Band %</div>
                    <div class="tech-value">${(bbPct * 100).toFixed(1)}%</div>
                    <div class="tech-sub">${bbLabel}</div>
                    <div class="gauge-bar">
                        <div class="gauge-fill" style="width:${Math.min(bbPct * 100, 100)}%;background:var(--accent-purple)"></div>
                    </div>
                </div>

                <div class="tech-card">
                    <div class="tech-label">Volatility (20D)</div>
                    <div class="tech-value">${((indicators.volatility_20d || 0) * 100).toFixed(2)}%</div>
                    <div class="tech-sub">10D: ${((indicators.volatility_10d || 0) * 100).toFixed(2)}%</div>
                </div>

                <div class="tech-card">
                    <div class="tech-label">Return 1D</div>
                    <div class="tech-value ${(indicators.return_1d || 0) >= 0 ? 'text-green' : 'text-red'}">
                        ${((indicators.return_1d || 0) * 100).toFixed(2)}%
                    </div>
                    <div class="tech-sub">5D: ${((indicators.return_5d || 0) * 100).toFixed(2)}% | 20D: ${((indicators.return_20d || 0) * 100).toFixed(2)}%</div>
                </div>

                <div class="tech-card">
                    <div class="tech-label">Volume Ratio</div>
                    <div class="tech-value">${(indicators.volume_ratio || 0).toFixed(2)}x</div>
                    <div class="tech-sub">vs 5D avg</div>
                </div>

                <div class="tech-card">
                    <div class="tech-label">Price vs SMA20</div>
                    <div class="tech-value ${(indicators.price_vs_sma20 || 1) >= 1 ? 'text-green' : 'text-red'}">
                        ${((indicators.price_vs_sma20 || 1) * 100 - 100).toFixed(2)}%
                    </div>
                    <div class="tech-sub">${(indicators.price_vs_sma20 || 1) >= 1 ? 'Above' : 'Below'} SMA20</div>
                </div>

                <div class="tech-card">
                    <div class="tech-label">News Sentiment</div>
                    <div class="tech-value" style="color: ${(indicators.sentiment_score || 0) > 0 ? 'var(--profit-green)' : (indicators.sentiment_score || 0) < 0 ? 'var(--loss-red)' : 'var(--text-muted)'}">
                        ${(indicators.sentiment_score || 0).toFixed(2)}
                    </div>
                    <div class="tech-sub">${indicators.news_count || 0} articles (24h)</div>
                </div>
            </div>
        `;
    },

    // ─── SIGNALS TABLE ──────────────────────────────────────────
    renderSignalHistory(signals) {
        const el = document.getElementById('signals-content');
        if (!el) return;

        if (!signals || signals.length === 0) {
            el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🎯</div><div class="empty-state-text">Chưa có tín hiệu Ensemble</div></div>';
            return;
        }

        const rows = signals.slice(0, 50).map(s => {
            const date = s.timestamp ? s.timestamp.split(' ')[0] : '—';
            return `
                <tr>
                    <td>${date}</td>
                    <td>${this.signalBadge(s.signal)}</td>
                    <td class="${s.score >= 0 ? 'text-green' : 'text-red'}">${this.formatScore(s.score)}</td>
                    <td>${s.sentiment_impact ? s.sentiment_impact.toFixed(4) : '0'}</td>
                </tr>
            `;
        }).join('');

        el.innerHTML = `
            <table class="signals-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Signal</th>
                        <th>Score</th>
                        <th>Sentiment</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    },

    // ─── NEWS FEED ──────────────────────────────────────────────
    renderNewsFeed(articles) {
        const el = document.getElementById('news-content');
        if (!el) return;

        if (!articles || articles.length === 0) {
            el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📰</div><div class="empty-state-text">Chưa có tin tức</div></div>';
            return;
        }

        el.innerHTML = `<div class="news-list">${articles.map(a => {
            const scoreColor = (a.sentiment_score || 0) > 0 ? 'text-green' : (a.sentiment_score || 0) < 0 ? 'text-red' : 'text-muted';
            return `
                <div class="news-item">
                    <div class="news-sentiment-dot ${a.sentiment_label || 'NEUTRAL'}"></div>
                    <div class="news-body">
                        <div class="news-title"><a href="${a.url || '#'}" target="_blank" rel="noopener">${a.title || 'Untitled'}</a></div>
                        <div class="news-meta">
                            <span>${a.source || ''}</span>
                            <span>${a.pub_date ? a.pub_date.split(' ')[0] : ''}</span>
                            <span class="news-score ${scoreColor}">${(a.sentiment_score || 0) >= 0 ? '+' : ''}${(a.sentiment_score || 0).toFixed(2)}</span>
                            <span>${a.sentiment_label || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('')}</div>`;
    },

    // ─── MODEL PREDICTIONS BREAKDOWN ────────────────────────────
    renderModelBreakdown(predictions) {
        const el = document.getElementById('models-content');
        if (!el) return;

        if (!predictions || !predictions.models) {
            el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🧠</div><div class="empty-state-text">Chưa có dữ liệu dự đoán</div></div>';
            return;
        }

        const models = predictions.models;
        let cards = '';

        // Classification models
        for (const key of ['cls_1d', 'cls_5d', 'cls_20d']) {
            const m = models[key];
            if (!m) continue;
            const horizon = key.replace('cls_', '');
            cards += `
                <div class="model-card">
                    <div class="model-card-header">
                        <span class="model-card-title">Classification ${horizon.toUpperCase()}</span>
                        <span class="model-card-type cls">CLS</span>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-row">
                            <span class="prob-label text-green">UP</span>
                            <div class="prob-bar"><div class="prob-fill" style="width:${(m.p_up * 100).toFixed(0)}%;background:var(--profit-green)"></div></div>
                            <span class="prob-value">${(m.p_up * 100).toFixed(1)}%</span>
                        </div>
                        <div class="prob-row">
                            <span class="prob-label text-muted">FLAT</span>
                            <div class="prob-bar"><div class="prob-fill" style="width:${(m.p_flat * 100).toFixed(0)}%;background:var(--neutral-gray)"></div></div>
                            <span class="prob-value">${(m.p_flat * 100).toFixed(1)}%</span>
                        </div>
                        <div class="prob-row">
                            <span class="prob-label text-red">DOWN</span>
                            <div class="prob-bar"><div class="prob-fill" style="width:${(m.p_down * 100).toFixed(0)}%;background:var(--loss-red)"></div></div>
                            <span class="prob-value">${(m.p_down * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    <div style="margin-top:8px;font-size:0.72rem;color:var(--text-muted)">
                        Score: <span class="${m.score >= 0 ? 'text-green' : 'text-red'}" style="font-family:var(--font-mono);font-weight:600">${m.score >= 0 ? '+' : ''}${m.score.toFixed(4)}</span>
                    </div>
                </div>
            `;
        }

        // Regression models
        for (const key of ['reg_1d', 'reg_5d', 'reg_20d']) {
            const m = models[key];
            if (!m) continue;
            const horizon = key.replace('reg_', '');
            const retPct = (m.expected_return * 100).toFixed(3);
            const retColor = m.expected_return >= 0 ? 'text-green' : 'text-red';
            cards += `
                <div class="model-card">
                    <div class="model-card-header">
                        <span class="model-card-title">Regression ${horizon.toUpperCase()}</span>
                        <span class="model-card-type reg">REG</span>
                    </div>
                    <div style="text-align:center;padding:12px 0">
                        <div style="font-size:0.72rem;color:var(--text-muted);margin-bottom:4px">Expected Return</div>
                        <div class="tech-value ${retColor}" style="font-size:1.6rem">${m.expected_return >= 0 ? '+' : ''}${retPct}%</div>
                    </div>
                    <div style="font-size:0.72rem;color:var(--text-muted)">
                        Score: <span class="${m.score >= 0 ? 'text-green' : 'text-red'}" style="font-family:var(--font-mono);font-weight:600">${m.score >= 0 ? '+' : ''}${m.score.toFixed(4)}</span>
                    </div>
                </div>
            `;
        }

        // Summary
        const summary = `
            <div class="backtest-metrics" style="margin-bottom:16px">
                <div class="bt-metric">
                    <div class="bt-metric-label">Ensemble Score</div>
                    <div class="bt-metric-value ${predictions.ensemble_score >= 0 ? 'text-green' : 'text-red'}">
                        ${predictions.ensemble_score >= 0 ? '+' : ''}${predictions.ensemble_score.toFixed(4)}
                    </div>
                </div>
                <div class="bt-metric">
                    <div class="bt-metric-label">Signal</div>
                    <div class="bt-metric-value">${this.signalBadge(predictions.signal)}</div>
                </div>
                <div class="bt-metric">
                    <div class="bt-metric-label">Sentiment Impact</div>
                    <div class="bt-metric-value">${predictions.sentiment_impact >= 0 ? '+' : ''}${predictions.sentiment_impact.toFixed(4)}</div>
                </div>
                <div class="bt-metric">
                    <div class="bt-metric-label">CLS / REG Score</div>
                    <div class="bt-metric-value text-mono" style="font-size:0.85rem">
                        ${(predictions.weighted_score_cls || 0).toFixed(3)} / ${(predictions.weighted_score_reg || 0).toFixed(3)}
                    </div>
                </div>
            </div>
        `;

        el.innerHTML = summary + `<div class="model-grid">${cards}</div>`;
    },

    // ─── BACKTEST SUMMARY ───────────────────────────────────────
    renderFuturePredictions(forecastData) {
        const el = document.getElementById('future-content');
        if (!el) return;

        if (!forecastData || !forecastData.forecast || forecastData.forecast.length === 0) {
            el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">F</div><div class="empty-state-text">No future predictions available</div></div>';
            return;
        }

        const pct = (value) => {
            if (value == null) return '--';
            return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`;
        };
        const fmtDate = (value) => value ? String(value).split(' ')[0] : '--';
        const directionClass = (direction) => direction === 'UP'
            ? 'text-green'
            : direction === 'DOWN'
                ? 'text-red'
                : 'text-muted';

        const summary = `
            <div class="backtest-metrics" style="margin-bottom:16px">
                <div class="bt-metric">
                    <div class="bt-metric-label">Base Price</div>
                    <div class="bt-metric-value">${this.formatPrice(forecastData.base_price)}</div>
                </div>
                <div class="bt-metric">
                    <div class="bt-metric-label">Latest Signal</div>
                    <div class="bt-metric-value">${this.signalBadge(forecastData.signal)}</div>
                </div>
                <div class="bt-metric">
                    <div class="bt-metric-label">Final Score</div>
                    <div class="bt-metric-value ${forecastData.final_score >= 0 ? 'text-green' : 'text-red'}">
                        ${this.formatScore(forecastData.final_score)}
                    </div>
                </div>
                <div class="bt-metric">
                    <div class="bt-metric-label">As Of</div>
                    <div class="bt-metric-value text-mono" style="font-size:0.85rem">${fmtDate(forecastData.decision_timestamp)}</div>
                </div>
            </div>
        `;

        const cards = forecastData.forecast.map(item => {
            const retClass = (item.expected_return || 0) >= 0 ? 'text-green' : 'text-red';
            const conf = item.classification_confidence == null
                ? '--'
                : `${(item.classification_confidence * 100).toFixed(1)}%`;
            return `
                <div class="model-card">
                    <div class="model-card-header">
                        <span class="model-card-title">Forecast ${item.horizon}D</span>
                        <span class="model-card-type reg">${fmtDate(item.target_timestamp)}</span>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
                        <div>
                            <div class="tech-label">Projected Price</div>
                            <div class="tech-value">${this.formatPrice(item.projected_price)}</div>
                        </div>
                        <div>
                            <div class="tech-label">Expected Return</div>
                            <div class="tech-value ${retClass}">${pct(item.expected_return)}</div>
                        </div>
                        <div>
                            <div class="tech-label">Direction</div>
                            <div class="tech-value ${directionClass(item.direction)}">${item.direction || '--'}</div>
                        </div>
                        <div>
                            <div class="tech-label">CLS Confidence</div>
                            <div class="tech-value">${conf}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        el.innerHTML = summary + `<div class="model-grid">${cards}</div>`;
    },

    renderBacktestPanel(btData) {
        const el = document.getElementById('backtest-content');
        if (!el) return;

        if (!btData || !btData.available) {
            el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📊</div><div class="empty-state-text">Chưa có dữ liệu Backtest. Chạy <code>python main.py --run-backtest SYMBOL</code></div></div>';
            return;
        }

        el.innerHTML = '<div id="backtest-chart"></div>';
        // Slight delay to let DOM render
        requestAnimationFrame(() => {
            Charts.renderBacktestChart('backtest-chart', btData.data, btData.columns);
        });
    },

    // ─── MODEL HEALTH (Sidebar) ─────────────────────────────────
    renderModelHealth(modelsStatus) {
        const el = document.getElementById('model-health-content');
        if (!el) return;

        if (!modelsStatus || modelsStatus.length === 0) {
            el.innerHTML = '<div class="text-muted" style="font-size:0.72rem">Chưa có model nào</div>';
            return;
        }

        el.innerHTML = modelsStatus.map(m => {
            let metric = '';
            if (m.test_metrics) {
                if (m.mode === 'classification') {
                    metric = `Acc: ${(m.test_metrics.accuracy * 100).toFixed(1)}%`;
                } else {
                    metric = `Dir: ${((m.test_metrics.directional_accuracy || 0) * 100).toFixed(1)}%`;
                }
            }
            const modeTag = m.mode === 'classification' ? 'CLS' : 'REG';
            return `
                <div class="model-health-item">
                    <span class="model-health-name">${m.symbol} ${m.horizon}D ${modeTag}</span>
                    <span class="model-health-metric ${m.mode === 'classification' ? 'text-purple' : 'text-blue'}">${metric}</span>
                </div>
            `;
        }).join('');
    }
};
