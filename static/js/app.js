/* ═══════════════════════════════════════════════════════════
   SMART CHURN PREDICTOR — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

const PLOTLY_THEME = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(17,24,39,0.5)',
    font: { color: '#f0f2f5', family: 'Inter, sans-serif' },
    margin: { l: 50, r: 30, t: 50, b: 50 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' }
};

const COLORS = {
    red: '#ef4444', orange: '#f97316', green: '#10b981',
    blue: '#3b82f6', purple: '#8b5cf6', cyan: '#06b6d4',
    models: { 'Logistic Regression': '#3b82f6', 'Decision Tree': '#10b981', 'Random Forest': '#f97316', 'XGBoost': '#ef4444' }
};

let bulkFile = null;
let bulkResults = null;

// ─── Chart Insights Helper ───────────────────────────────────
function chartInsights(points) {
    return `<div class="insight-card">
        <div class="insight-card-header">📌 How to read this chart</div>
        ${points.map((p, i) => `<div class="insight-point"><span class="insight-num">${i + 1}</span><span>${p}</span></div>`).join('')}
    </div>`;
}

// ─── Navigation ──────────────────────────────────────────────
function navigateTo(page) {
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelector(`.nav-item[data-page="${page}"]`).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (page === 'home') loadDashboard();
    if (page === 'browse') loadBrowseHistory();
    if (page === 'analytics') loadAnalytics();
    if (page === 'comparison') loadComparison();
    if (page === 'explainability') loadExplainability();
}

// ─── Dashboard KPIs ──────────────────────────────────────────
async function loadDashboard() {
    try {
        const res = await fetch('/api/dashboard');
        const data = await res.json();
        animateNumber('kpi-total', data.total_customers);
        animateNumber('kpi-active', data.active);
        animateNumber('kpi-churned', data.churned);
        document.getElementById('kpi-rate').textContent = data.churn_rate + '%';
    } catch (e) { console.error('Dashboard error:', e); }
}

function animateNumber(id, target) {
    const el = document.getElementById(id);
    const duration = 1200;
    const start = 0;
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.floor(start + (target - start) * eased).toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ─── Single Prediction — Identity & Lock Logic ──────────────
function confirmCustomerIdentity() {
    const custId = document.getElementById('inp-cust-id').value.trim();
    const custName = document.getElementById('inp-cust-name').value.trim();

    if (!custId || !custName) {
        alert('Please enter both Customer ID and Customer Name before confirming.');
        return;
    }

    // Lock identity fields
    document.getElementById('inp-cust-id').disabled = true;
    document.getElementById('inp-cust-name').disabled = true;
    document.getElementById('confirm-identity-btn').disabled = true;
    document.getElementById('confirm-identity-btn').style.opacity = '0.5';

    // Show locked banner
    document.getElementById('identity-display').textContent = `${custName} (ID: ${custId})`;
    document.getElementById('identity-locked-banner').style.display = 'block';

    // Enable the data card
    const dataCard = document.getElementById('customer-data-card');
    dataCard.style.opacity = '1';
    dataCard.style.pointerEvents = 'auto';
}

// ─── Single Prediction ──────────────────────────────────────
async function runPrediction() {
    const payload = {
        'Age': parseInt(document.getElementById('inp-age').value),
        'Gender': document.getElementById('inp-gender').value,
        'Tenure': parseInt(document.getElementById('inp-tenure').value),
        'Usage Frequency': parseInt(document.getElementById('inp-usage').value),
        'Support Calls': parseInt(document.getElementById('inp-support').value),
        'Payment Delay': parseInt(document.getElementById('inp-delay').value),
        'Subscription Type': document.getElementById('inp-sub').value,
        'Contract Length': document.getElementById('inp-contract').value,
        'Total Spend': parseFloat(document.getElementById('inp-spend').value),
        'Last Interaction': parseInt(document.getElementById('inp-interaction').value)
    };

    const container = document.getElementById('prediction-results');
    container.style.display = 'block';
    container.innerHTML = '<div class="spinner"></div><div class="loading-text">Analyzing with AI...</div>';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Lock all data fields after prediction
        lockDataFields();

        renderPredictionResults(data);

        // Show the "Add Another Customer" button
        document.getElementById('add-another-btn-container').style.display = 'block';
    } catch (e) {
        container.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error: ${e.message}</div>`;
    }
}

function lockDataFields() {
    const fieldsToLock = [
        'inp-age', 'inp-gender', 'inp-tenure', 'inp-usage',
        'inp-support', 'inp-delay', 'inp-sub', 'inp-contract',
        'inp-spend', 'inp-interaction'
    ];
    fieldsToLock.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = true;
    });
    document.getElementById('fields-locked-badge').style.display = 'block';
}

function resetForNewCustomer() {
    // Unlock and clear identity fields
    document.getElementById('inp-cust-id').value = '';
    document.getElementById('inp-cust-name').value = '';
    document.getElementById('inp-cust-id').disabled = false;
    document.getElementById('inp-cust-name').disabled = false;
    document.getElementById('confirm-identity-btn').disabled = false;
    document.getElementById('confirm-identity-btn').style.opacity = '1';
    document.getElementById('identity-locked-banner').style.display = 'none';
    document.getElementById('identity-display').textContent = '';

    // Unlock and reset all data fields to default values
    const defaults = {
        'inp-age': 35, 'inp-tenure': 12, 'inp-usage': 15,
        'inp-support': 2, 'inp-delay': 5, 'inp-interaction': 10, 'inp-spend': 500
    };
    Object.entries(defaults).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) { el.value = val; el.disabled = false; }
    });

    ['inp-gender', 'inp-sub', 'inp-contract'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.selectedIndex = 0; el.disabled = false; }
    });

    document.getElementById('fields-locked-badge').style.display = 'none';

    // Grey out data card again until identity is confirmed
    const dataCard = document.getElementById('customer-data-card');
    dataCard.style.opacity = '0.45';
    dataCard.style.pointerEvents = 'none';

    // Hide results and reset button
    document.getElementById('prediction-results').style.display = 'none';
    document.getElementById('prediction-results').innerHTML = '';
    document.getElementById('add-another-btn-container').style.display = 'none';

    // Scroll to top of predict section
    document.getElementById('page-predict').scrollIntoView({ behavior: 'smooth', block: 'start' });
}



function renderPredictionResults(data) {
    const container = document.getElementById('prediction-results');
    const isChurn = data.prediction === 1;
    const prob = (data.probability * 100).toFixed(1);
    const reasons = data.reasons || {};
    const cd = data.customer_data || {};

    // --- Make container visible FIRST so Plotly can get real dimensions ---
    container.style.display = 'block';

    let html = '';

    // ── Alert Banner ──────────────────────────────────────────
    html += `<div class="result-banner ${isChurn ? 'churn' : 'safe'}" style="margin-bottom:20px;">
        <span class="banner-icon">${isChurn ? '⚠️' : '✅'}</span>
        <div style="flex:1;">
            <strong style="font-size:1.1rem;">${isChurn ? '🚨 HIGH ALERT: This customer is likely to CHURN' : '✅ SAFE: This customer is likely to STAY'}</strong>
            <div style="font-size:0.9rem;opacity:0.85;margin-top:6px;display:flex;gap:24px;flex-wrap:wrap;">
                <span>📊 Churn Probability: <strong>${prob}%</strong></span>
                <span>🎯 Risk Level: <strong>${data.risk_level}</strong></span>
                <span>📋 Subscription: <strong>${cd['Subscription Type'] || '—'}</strong></span>
                <span>📜 Contract: <strong>${cd['Contract Length'] || '—'}</strong></span>
                <span>⏱ Tenure: <strong>${cd['Tenure'] || '—'} months</strong></span>
            </div>
        </div>
    </div>`;

    // ── Gauge + Distribution charts ───────────────────────────
    html += `<div class="grid-2" style="margin-bottom:24px;">
        <div>
            <div class="chart-container" id="gauge-chart" style="min-height:320px;"></div>
            ${chartInsights([
        'Green zone (0–40%): Low risk — customer shows stable behaviour, no urgent action needed',
        'Amber zone (40–75%): Elevated risk — monitor closely and consider a proactive retention offer',
        'Red zone (75–100%): Critical risk — immediate intervention is recommended to prevent loss'
    ])}
        </div>
        <div>
            <div class="chart-container" id="dist-chart" style="min-height:320px;"></div>
            ${chartInsights([
        'Taller green bar means the model is more confident the customer will stay',
        'Taller red bar signals multiple churn risk factors have been detected',
        'A near-equal split indicates uncertainty — use the risk factor details below to dig deeper'
    ])}
        </div>
    </div>`;

    // ── Explanation: Why Will This Customer Churn / Stay ─────
    const churnReasons = reasons.churn_reasons || [];
    const stayReasons = reasons.stay_reasons || [];

    const iconMap = {
        'Tenure': '⏱', 'Support Calls': '📞', 'Payment Delay': '💳',
        'Usage Frequency': '📱', 'Total Spend': '💰', 'Contract Length': '📜',
        'Last Interaction': '🕐', 'Subscription Type': '📦', 'Age': '👤', 'Gender': '🧑'
    };

    html += `<div class="card" style="margin-bottom:24px;border:1px solid ${isChurn ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'};">
        <div class="card-title" style="font-size:1.15rem;">🔍 Why This Customer Will ${isChurn ? 'Churn' : 'Stay'} — Detailed Analysis</div>
        <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:20px;line-height:1.6;">
            The AI analysed <strong>${Object.keys(iconMap).length}</strong> customer behavioural and financial factors. Below are the key drivers ranked by impact on the prediction.
        </p>
        <div class="grid-2">
            <div>
                <div style="font-size:0.95rem;font-weight:700;color:${COLORS.red};margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid rgba(239,68,68,0.2);">
                    ⚠️ Churn Risk Factors
                </div>
                ${churnReasons.length > 0
            ? churnReasons.map(r => `
                        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:12px;padding:14px;margin-bottom:12px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                                <span style="font-size:1.2rem;">${iconMap[r.feature] || '🔹'}</span>
                                <strong style="color:var(--accent-red);font-size:0.95rem;">${r.feature}</strong>
                                <span style="margin-left:auto;font-size:0.75rem;background:rgba(239,68,68,0.2);color:var(--accent-red);padding:2px 8px;border-radius:20px;font-weight:700;">impact: ${(r.impact * 100).toFixed(1)}%</span>
                            </div>
                            <p style="color:var(--text-secondary);font-size:0.875rem;line-height:1.65;margin:0;">${r.reason}</p>
                        </div>`).join('')
            : `<div style="color:var(--text-muted);padding:16px;text-align:center;background:rgba(16,185,129,0.05);border-radius:10px;">✨ No significant risk factors detected</div>`
        }
            </div>
            <div>
                <div style="font-size:0.95rem;font-weight:700;color:${COLORS.green};margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid rgba(16,185,129,0.2);">
                    ✅ Retention / Loyalty Factors
                </div>
                ${stayReasons.length > 0
            ? stayReasons.map(r => `
                        <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.15);border-radius:12px;padding:14px;margin-bottom:12px;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                                <span style="font-size:1.2rem;">${iconMap[r.feature] || '🔹'}</span>
                                <strong style="color:var(--accent-green);font-size:0.95rem;">${r.feature}</strong>
                                <span style="margin-left:auto;font-size:0.75rem;background:rgba(16,185,129,0.2);color:var(--accent-green);padding:2px 8px;border-radius:20px;font-weight:700;">impact: ${(r.impact * 100).toFixed(1)}%</span>
                            </div>
                            <p style="color:var(--text-secondary);font-size:0.875rem;line-height:1.65;margin:0;">${r.reason}</p>
                        </div>`).join('')
            : `<div style="color:var(--text-muted);padding:16px;text-align:center;background:rgba(239,68,68,0.05);border-radius:10px;">⚠️ No strong retention factors found</div>`
        }
            </div>
        </div>
    </div>`;

    // ── Feature Impact Chart ──────────────────────────────────
    html += `<div class="chart-container" id="impact-chart" style="margin-bottom:12px;min-height:380px;"></div>
    ${chartInsights([
        'Red bars push this prediction toward churn; green bars push toward retention',
        'Longer bar = stronger influence on <em>this specific customer\'s</em> prediction',
        'The feature with the longest bar is the single biggest driver — address it first for maximum retention impact'
    ])}
    <div style="margin-bottom:24px;"></div>`;

    // ── Actionable Recommendations ────────────────────────────
    const recs = reasons.recommendations || [];
    if (recs.length) {
        html += `<div class="card" style="margin-bottom:24px;border:1px solid rgba(99,102,241,0.2);">
            <div class="card-title">💡 Actionable Retention Recommendations</div>
            <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:16px;">Based on the identified risk factors, here are targeted strategies to reduce churn probability:</p>
            <div style="display:flex;flex-direction:column;gap:10px;">
            ${recs.map((r, i) => `
                <div style="display:flex;gap:14px;align-items:flex-start;background:rgba(99,102,241,0.06);border-radius:10px;padding:14px;border:1px solid rgba(99,102,241,0.12);">
                    <span style="background:rgba(99,102,241,0.2);color:var(--accent-purple);width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.85rem;flex-shrink:0;">${i + 1}</span>
                    <span style="color:var(--text-primary);font-size:0.9rem;line-height:1.6;">${r}</span>
                </div>`).join('')}
            </div>
        </div>`;
    }

    // ── All Models Prediction Table ───────────────────────────
    html += `<div class="card" style="margin-bottom:24px;">
        <div class="card-title">⚖️ All Model Predictions</div>
        <div class="table-container"><table>
            <thead><tr><th>Model</th><th>Prediction</th><th>Probability</th></tr></thead>
            <tbody>
                ${Object.entries(data.all_models).map(([name, m]) => `<tr>
                    <td style="font-weight:600;">${name}</td>
                    <td><span class="badge ${m.prediction === 1 ? 'badge-danger' : 'badge-success'}">${m.prediction === 1 ? 'Churn' : 'Stay'}</span></td>
                    <td>${(m.probability * 100).toFixed(1)}%</td>
                </tr>`).join('')}
            </tbody>
        </table></div>
    </div>`;

    // ── Feature Importance ────────────────────────────────────
    html += `<div class="chart-container" id="fi-chart" style="margin-bottom:12px;min-height:380px;"></div>
    ${chartInsights([
        'Higher score = the model relies on this feature most when making predictions across all customers',
        'Unlike the impact chart above, this reflects patterns across the <em>entire dataset</em>, not just this customer',
        'Use the top-3 features to focus your retention strategy, data quality efforts, and campaign targeting'
    ])}
    <div style="margin-bottom:24px;"></div>`;

    container.innerHTML = html;

    // ── Render Plotly charts: use requestAnimationFrame + delay ──
    // Two frames ensures the browser has laid out & painted the elements
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            setTimeout(() => {
                try { renderGaugeChart(data.probability, data.risk_level); } catch (e) { console.error('GaugeChart error', e); }
                try { renderDistChart(data.probability); } catch (e) { console.error('DistChart error', e); }
                try { renderImpactChart(reasons.feature_impacts || []); } catch (e) { console.error('ImpactChart error', e); }
                try { renderFeatureImportance(data.feature_importance); } catch (e) { console.error('FIChart error', e); }
            }, 150);
        });
    });
}


function renderGaugeChart(prob, riskLevel) {
    const color = prob < 0.4 ? COLORS.green : prob < 0.75 ? COLORS.orange : COLORS.red;
    Plotly.newPlot('gauge-chart', [{
        type: 'indicator',
        mode: 'gauge+number',
        value: prob * 100,
        number: { suffix: '%', font: { size: 52, color: '#f0f2f5' } },
        title: { text: `<b>Churn Probability</b><br><span style="font-size:0.8em;color:${color}">${riskLevel}</span>`, font: { size: 18, color: '#f0f2f5' } },
        gauge: {
            axis: { range: [0, 100], tickcolor: '#6b7280' },
            bar: { color, thickness: 0.8 },
            bgcolor: '#1f2937',
            borderwidth: 1, bordercolor: '#374151',
            steps: [
                { range: [0, 40], color: 'rgba(16,185,129,0.15)' },
                { range: [40, 75], color: 'rgba(245,158,11,0.15)' },
                { range: [75, 100], color: 'rgba(239,68,68,0.15)' }
            ]
        }
    }], { ...PLOTLY_THEME, height: 320, margin: { t: 90, b: 10, l: 30, r: 30 } }, { responsive: true, displayModeBar: false });
}

function renderDistChart(prob) {
    Plotly.newPlot('dist-chart', [{
        type: 'bar',
        x: ['Will Stay', 'Will Churn'],
        y: [(1 - prob) * 100, prob * 100],
        marker: {
            color: [COLORS.green, COLORS.red],
            line: { color: '#374151', width: 1 }
        },
        text: [`${((1 - prob) * 100).toFixed(1)}%`, `${(prob * 100).toFixed(1)}%`],
        textposition: 'auto',
        textfont: { size: 18, color: '#f0f2f5' }
    }], {
        ...PLOTLY_THEME,
        height: 320,
        title: { text: '<b>Prediction Distribution</b>', font: { size: 18, color: '#f0f2f5' } },
        yaxis: { ...PLOTLY_THEME.yaxis, title: 'Probability (%)', range: [0, 100] }
    }, { responsive: true, displayModeBar: false });
}

function renderImpactChart(impacts) {
    if (!impacts.length) return;
    const sorted = [...impacts].sort((a, b) => a.shap_value - b.shap_value);
    Plotly.newPlot('impact-chart', [{
        type: 'bar',
        y: sorted.map(f => f.feature),
        x: sorted.map(f => f.shap_value),
        orientation: 'h',
        marker: {
            color: sorted.map(f => f.shap_value > 0 ? COLORS.red : COLORS.green),
            line: { color: '#374151', width: 1 }
        },
        text: sorted.map(f => (f.shap_value > 0 ? '+' : '') + f.shap_value.toFixed(3)),
        textposition: 'auto'
    }], {
        ...PLOTLY_THEME,
        height: 400,
        title: { text: '<b>How Each Feature Influenced This Prediction</b>', font: { size: 16, color: '#f0f2f5' } },
        xaxis: {
            ...PLOTLY_THEME.xaxis,
            title: 'Impact (+ = churn, - = stay)',
            zeroline: true, zerolinecolor: '#f0f2f5', zerolinewidth: 2
        }
    }, { responsive: true, displayModeBar: false });
}

function renderFeatureImportance(fi) {
    if (!fi.length) return;
    const sorted = fi.sort((a, b) => a.importance - b.importance);
    Plotly.newPlot('fi-chart', [{
        type: 'bar',
        y: sorted.map(f => f.feature),
        x: sorted.map(f => f.importance),
        orientation: 'h',
        marker: {
            color: sorted.map((_, i) => `hsl(${180 + i * 18}, 70%, 55%)`),
            line: { color: '#374151', width: 1 }
        },
        text: sorted.map(f => f.importance.toFixed(4)),
        textposition: 'auto'
    }], {
        ...PLOTLY_THEME,
        height: 400,
        title: { text: '<b>Feature Importance (Random Forest)</b>', font: { size: 16, color: '#f0f2f5' } },
        xaxis: { ...PLOTLY_THEME.xaxis, title: 'Importance Score' }
    }, { responsive: true, displayModeBar: false });
}

// ─── Customer Lookup & Browse ────────────────────────────────

// ── Per-Customer Detail Search ──────────────────────────────
async function searchCustomerDetail() {
    const query = document.getElementById('lookup-search-input').value.trim();
    if (!query) { alert('Please enter a Customer ID or Name.'); return; }

    // Hide previous results
    document.getElementById('history-results').style.display = 'none';
    const panel = document.getElementById('customer-detail-panel');
    panel.style.display = 'block';
    panel.innerHTML = '<div class="spinner"></div><div class="loading-text">Loading customer profile...</div>';

    try {
        const res = await fetch(`/api/customer_detail?search=${encodeURIComponent(query)}`);
        const data = await res.json();

        if (!res.ok || data.error) {
            panel.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> ${data.error || 'Customer not found.'}</div>`;
            return;
        }

        renderCustomerDetail(data);
    } catch (e) {
        panel.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error: ${e.message}</div>`;
    }
}

function renderCustomerDetail(data) {
    const panel = document.getElementById('customer-detail-panel');
    const p = data.profile;
    const cmp = data.comparison;

    // Restore the panel structure
    panel.innerHTML = `
        <div id="customer-profile-banner" class="card" style="margin-bottom:20px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;right:0;width:300px;height:100%;background:linear-gradient(135deg,transparent,rgba(99,102,241,0.07));pointer-events:none;"></div>
            <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
                <div id="customer-avatar" style="width:72px;height:72px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2rem;font-weight:800;flex-shrink:0;"></div>
                <div style="flex:1;">
                    <div style="font-size:1.5rem;font-weight:800;" id="detail-name">—</div>
                    <div style="color:var(--text-secondary);font-size:0.9rem;margin-top:2px;" id="detail-id">—</div>
                    <div style="margin-top:8px;display:flex;gap:10px;flex-wrap:wrap;">
                        <span id="detail-churn-badge"></span>
                        <span id="detail-subscription-badge" style="padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:700;background:rgba(59,130,246,0.15);color:var(--accent-blue);border:1px solid rgba(59,130,246,0.3);"></span>
                        <span id="detail-contract-badge" style="padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:700;background:rgba(6,182,212,0.15);color:var(--accent-cyan);border:1px solid rgba(6,182,212,0.3);"></span>
                    </div>
                </div>
                <div class="grid-2" style="gap:16px;min-width:280px;" id="detail-kpis"></div>
            </div>
        </div>
        <div class="grid-2" style="margin-bottom:20px;">
            <div>
                <div class="chart-container" id="hist-comparison-chart"></div>
                <div id="hist-bar-insights"></div>
            </div>
            <div>
                <div class="chart-container" id="hist-radar-chart"></div>
                <div id="hist-radar-insights"></div>
            </div>
        </div>
        <div class="card">
            <div class="card-title">📋 Full Customer Record</div>
            <div class="table-container">
                <table id="detail-fields-table">
                    <thead><tr><th>Field</th><th>Value</th></tr></thead>
                    <tbody id="detail-fields-body"></tbody>
                </table>
            </div>
        </div>`;

    const isChurned = data.is_churned;
    const name = p['Customer Name'] || 'Unknown';
    const initials = name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
    const avatarColors = isChurned
        ? 'background:rgba(239,68,68,0.2);color:#ef4444;border:2px solid rgba(239,68,68,0.4);'
        : 'background:rgba(16,185,129,0.2);color:#10b981;border:2px solid rgba(16,185,129,0.4);';

    document.getElementById('customer-avatar').style.cssText += avatarColors;
    document.getElementById('customer-avatar').textContent = initials;
    document.getElementById('detail-name').textContent = name;

    const idCol = Object.keys(p).find(k => k.toLowerCase().includes('customerid') || k.toLowerCase() === 'id');
    document.getElementById('detail-id').textContent = idCol ? `ID: ${p[idCol]}` : '';

    // Churn badge
    const churnBadge = document.getElementById('detail-churn-badge');
    churnBadge.textContent = isChurned ? '❌ Churned' : '✅ Active';
    churnBadge.style.cssText = `padding:4px 14px;border-radius:20px;font-size:0.85rem;font-weight:800;
        background:${isChurned ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)'};
        color:${isChurned ? 'var(--accent-red)' : 'var(--accent-green)'};
        border:1px solid ${isChurned ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'};`;

    document.getElementById('detail-subscription-badge').textContent = p['Subscription Type'] ? `📦 ${p['Subscription Type']}` : '';
    document.getElementById('detail-contract-badge').textContent = p['Contract Length'] ? `📋 ${p['Contract Length']}` : '';

    // KPI tiles
    const kpiFields = [
        { label: '🕐 Tenure', key: 'Tenure', suffix: ' mo' },
        { label: '💰 Total Spend', key: 'Total Spend', prefix: '$' },
        { label: '📞 Support Calls', key: 'Support Calls' },
        { label: '⏱ Last Interaction', key: 'Last Interaction', suffix: ' days' }
    ];
    const kpiContainer = document.getElementById('detail-kpis');
    kpiContainer.innerHTML = kpiFields.map(f => {
        const val = p[f.key] !== null && p[f.key] !== undefined ? p[f.key] : '—';
        return `<div style="background:rgba(99,102,241,0.08);border-radius:12px;padding:14px;text-align:center;border:1px solid rgba(99,102,241,0.15);">
            <div style="font-size:0.78rem;color:var(--text-secondary);margin-bottom:4px;">${f.label}</div>
            <div style="font-size:1.3rem;font-weight:800;">${f.prefix || ''}${val}${val !== '—' ? (f.suffix || '') : ''}</div>
        </div>`;
    }).join('');

    // Charts (after DOM is ready)
    setTimeout(() => {
        // Grouped bar chart: customer vs. average
        const compLabels = Object.keys(cmp);
        const custVals = compLabels.map(k => cmp[k].customer);
        const avgVals = compLabels.map(k => cmp[k].average);

        Plotly.newPlot('hist-comparison-chart', [
            { name: `${name}`, type: 'bar', x: compLabels, y: custVals, marker: { color: isChurned ? COLORS.red : COLORS.green }, text: custVals.map(String), textposition: 'auto', textfont: { color: '#fff', size: 12 } },
            { name: 'Dataset Average', type: 'bar', x: compLabels, y: avgVals, marker: { color: COLORS.blue }, text: avgVals.map(v => String(Math.round(v))), textposition: 'auto', textfont: { color: '#fff', size: 12 } }
        ], {
            ...PLOTLY_THEME, height: 380, barmode: 'group',
            title: { text: '<b>Customer vs. Average Comparison</b>', font: { size: 16, color: '#f0f2f5' } },
            legend: { bgcolor: 'rgba(31,41,55,0.8)', font: { color: '#f0f2f5' } }
        }, { responsive: true, displayModeBar: false });

        // Radar chart
        const radarFields = ['Tenure', 'Total Spend', 'Usage Frequency', 'Support Calls', 'Payment Delay'];
        const radarExisting = radarFields.filter(f => cmp[f]);
        const histBarEl = document.getElementById('hist-bar-insights');
        if (histBarEl) histBarEl.innerHTML = chartInsights([
            'This customer\'s bars vs. the dataset average reveal how atypical their behaviour is',
            'Values significantly above average for Support Calls or Payment Delay are classic churn signals',
            'Gaps between the customer and average help personalise retention offers for this specific individual'
        ]);

        if (radarExisting.length >= 3) {
            const normalize = (val, key) => {
                const avg = cmp[key]?.average || 1;
                return Math.min((val / (avg * 2)) * 100, 100);
            };
            Plotly.newPlot('hist-radar-chart', [
                {
                    type: 'scatterpolar', fill: 'toself', name: name,
                    r: radarExisting.map(k => normalize(cmp[k].customer, k)),
                    theta: radarExisting,
                    fillcolor: isChurned ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)',
                    line: { color: isChurned ? COLORS.red : COLORS.green, width: 2 }
                },
                {
                    type: 'scatterpolar', fill: 'toself', name: 'Avg Customer',
                    r: radarExisting.map(() => 50),
                    theta: radarExisting,
                    fillcolor: 'rgba(59,130,246,0.1)',
                    line: { color: COLORS.blue, width: 2, dash: 'dash' }
                }
            ], {
                ...PLOTLY_THEME, height: 380,
                polar: {
                    radialaxis: { visible: true, range: [0, 100], gridcolor: 'rgba(255,255,255,0.06)', tickfont: { color: '#6b7280' } },
                    angularaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
                    bgcolor: 'rgba(17,24,39,0.5)'
                },
                title: { text: '<b>Behaviour Profile (vs. Avg)</b>', font: { size: 16, color: '#f0f2f5' } },
                legend: { bgcolor: 'rgba(31,41,55,0.8)', font: { color: '#f0f2f5' } }
            }, { responsive: true, displayModeBar: false });

            const histRadarEl = document.getElementById('hist-radar-insights');
            if (histRadarEl) histRadarEl.innerHTML = chartInsights([
                'The customer polygon vs. dashed average baseline shows behavioural deviations at a glance',
                'Areas where the customer polygon extends beyond the average indicate higher-than-typical values',
                'A tightly balanced shape near the centre typically indicates a stable, average-risk customer'
            ]);
        }
    }, 100);

    // Full record table
    const tbody = document.getElementById('detail-fields-body');
    const skipFields = ['Churn Date'];
    tbody.innerHTML = Object.entries(p)
        .filter(([k]) => !skipFields.includes(k))
        .map(([k, v]) => {
            let displayVal = v === null || v === '' ? '<span style="color:var(--text-muted);">—</span>' : v;
            if (k === 'Churn') {
                displayVal = isChurned
                    ? '<span class="badge badge-danger">Churned</span>'
                    : '<span class="badge badge-success">Active</span>';
            }
            return `<tr><td style="font-weight:600;color:var(--text-secondary);">${k}</td><td>${displayVal}</td></tr>`;
        }).join('');
}



// Set default dates on load
document.addEventListener('DOMContentLoaded', () => {
    const end = new Date();
    const start = new Date();
    start.setDate(end.getDate() - 30);

    document.getElementById('history-end-date').valueAsDate = end;
    document.getElementById('history-start-date').valueAsDate = start;

    // Check URL hash or default to home
    const initialPage = window.location.hash ? window.location.hash.substring(1) : 'home';
    if (document.querySelector(`.nav-item[data-page="${initialPage}"]`)) {
        navigateTo(initialPage);
    } else {
        navigateTo('home');
    }
});

async function loadBrowseHistory() {
    const start = document.getElementById('history-start-date').value;
    const end = document.getElementById('history-end-date').value;

    let url = '/api/customer_history?';
    if (start && end) url += `start=${start}&end=${end}`;

    const resultsDiv = document.getElementById('history-results');
    resultsDiv.style.display = 'block';
    resultsDiv.style.opacity = '0.5';

    try {
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        animateNumber('hist-kpi-found', data.metrics.total_found);
        animateNumber('hist-kpi-churned', data.metrics.churned_in_period);

        // Animated currency
        const spendEl = document.getElementById('hist-kpi-spend');
        const targetSpend = data.metrics.total_spend;
        const spendStart = performance.now();
        (function animateSpend(now) {
            const p = Math.min((now - spendStart) / 1200, 1);
            const e = 1 - Math.pow(1 - p, 3);
            spendEl.textContent = '$' + (targetSpend * e).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            if (p < 1) requestAnimationFrame(animateSpend);
        })(spendStart);

        document.getElementById('history-thead').innerHTML = `<tr>${data.columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
        const tbody = document.getElementById('history-tbody');
        if (!data.data.length) {
            tbody.innerHTML = `<tr><td colspan="${data.columns.length}" style="text-align:center;padding:28px;color:var(--text-muted);">No customers found in this date range.</td></tr>`;
        } else {
            tbody.innerHTML = data.data.map(row =>
                `<tr>${data.columns.map(col => {
                    let v = row[col];
                    if (v === null || v === 'null' || v === 'None') v = '-';
                    if (col === 'Churn Date' && v !== '-') v = `<span class="badge badge-danger" style="font-size:0.75rem;">${v}</span>`;
                    if (col === 'Customer Name') v = `<strong>${v}</strong>`;
                    return `<td>${v}</td>`;
                }).join('')}</tr>`
            ).join('');
        }
        resultsDiv.style.opacity = '1';
    } catch (e) {
        resultsDiv.style.opacity = '1';
        resultsDiv.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error: ${e.message}</div>`;
    }
}

// ─── Revenue Chart (Home Page) ───────────────────────────────
function renderRevenueChart() {
    const el = document.getElementById('revenue-chart');
    if (!el) return;
    Plotly.newPlot('revenue-chart', [{
        type: 'bar',
        orientation: 'h',
        x: [400, 65, 1850],
        y: ['Acquire New Customer', 'Retain Existing Customer', 'Avg Lifetime Value'],
        marker: {
            color: ['rgba(239,68,68,0.75)', 'rgba(16,185,129,0.75)', 'rgba(59,130,246,0.75)'],
            line: { color: ['#ef4444', '#10b981', '#3b82f6'], width: 2 }
        },
        text: ['$400', '$65', '$1,850'],
        textposition: 'outside',
        textfont: { color: '#f0f2f5', size: 13, family: 'Inter, sans-serif' },
        hovertemplate: '%{y}: <b>$%{x}</b><extra></extra>'
    }], {
        ...PLOTLY_THEME,
        height: 220,
        margin: { l: 180, r: 60, t: 10, b: 30 },
        xaxis: { ...PLOTLY_THEME.xaxis, title: 'Cost / Value (USD)', tickprefix: '$' },
        yaxis: { ...PLOTLY_THEME.yaxis, tickfont: { size: 12, color: '#a3adc2' } },
        showlegend: false
    }, { responsive: true, displayModeBar: false });
}

// ─── Bulk Prediction ─────────────────────────────────────────
function handleFileUpload(input) {
    if (!input.files.length) return;
    bulkFile = input.files[0];
    document.getElementById('file-info').style.display = 'block';
    document.getElementById('file-info').innerHTML = `<div style="color:var(--accent-green);font-weight:600;">✅ File selected: ${bulkFile.name} (${(bulkFile.size / 1024).toFixed(1)} KB)</div>`;
    document.getElementById('bulk-predict-btn').style.display = 'block';
}

async function runBulkPrediction() {
    if (!bulkFile) return;
    const fd = new FormData();
    fd.append('file', bulkFile);

    const container = document.getElementById('bulk-results');
    container.style.display = 'block';
    container.innerHTML = '<div class="spinner"></div><div class="loading-text">Processing predictions...</div>';

    try {
        const res = await fetch('/api/bulk-predict', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        bulkResults = data;
        renderBulkResults(data);
    } catch (e) {
        container.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error: ${e.message}</div>`;
    }
}

function renderBulkResults(data) {
    const container = document.getElementById('bulk-results');
    let html = '';

    // KPIs
    html += `<div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Total Customers</div><div class="kpi-value">${data.total}</div></div>
        <div class="kpi-card"><div class="kpi-label">Predicted Churn</div><div class="kpi-value">${data.churned} <small>(${data.churn_rate}%)</small></div></div>
        <div class="kpi-card"><div class="kpi-label">High Risk</div><div class="kpi-value">${data.high_risk}</div></div>
        <div class="kpi-card"><div class="kpi-label">Avg Probability</div><div class="kpi-value">${(data.avg_probability * 100).toFixed(1)}%</div></div>
    </div>`;

    // Distribution charts
    html += `<div class="grid-2">
        <div class="chart-container" id="bulk-pie"></div>
        <div class="chart-container" id="bulk-risk-bar"></div>
    </div>`;

    // Results table
    html += `<div class="card" style="margin-bottom:24px;">
        <div class="card-title">📋 Detailed Results</div>
        <div class="table-container" style="max-height:500px;overflow-y:auto;"><table>
            <thead><tr><th>#</th><th>Prediction</th><th>Probability</th><th>Risk Level</th></tr></thead>
            <tbody>${data.results.map(r => `<tr>
                <td>${r.index}</td>
                <td><span class="badge ${r.prediction === 1 ? 'badge-danger' : 'badge-success'}">${r.prediction === 1 ? 'Churn' : 'Stay'}</span></td>
                <td>${(r.probability * 100).toFixed(1)}%</td>
                <td><span class="badge ${r.risk_level === 'High Risk' ? 'badge-danger' : r.risk_level === 'Medium Risk' ? 'badge-warning' : 'badge-success'}">${r.risk_level}</span></td>
            </tr>`).join('')}</tbody>
        </table></div>
    </div>`;

    // Download button
    html += `<button class="btn btn-green btn-full" onclick="downloadBulkCSV()">📥 Download Results as CSV</button>`;

    container.innerHTML = html;

    // Charts
    setTimeout(() => {
        Plotly.newPlot('bulk-pie', [{
            type: 'pie', labels: ['Will Stay', 'Will Churn'], values: [data.total - data.churned, data.churned],
            marker: { colors: [COLORS.green, COLORS.red] },
            textfont: { color: '#f0f2f5' }, hole: 0.4
        }], { ...PLOTLY_THEME, height: 350, title: { text: '<b>Churn Distribution</b>', font: { size: 16, color: '#f0f2f5' } } }, { responsive: true, displayModeBar: false });

        Plotly.newPlot('bulk-risk-bar', [{
            type: 'bar',
            x: ['Low Risk', 'Medium Risk', 'High Risk'],
            y: [data.low_risk, data.medium_risk, data.high_risk],
            marker: { color: [COLORS.green, COLORS.orange, COLORS.red] },
            text: [data.low_risk, data.medium_risk, data.high_risk],
            textposition: 'auto',
            textfont: { size: 16, color: '#f0f2f5' }
        }], { ...PLOTLY_THEME, height: 350, title: { text: '<b>Risk Level Distribution</b>', font: { size: 16, color: '#f0f2f5' } } }, { responsive: true, displayModeBar: false });
    }, 100);
}

async function downloadBulkCSV() {
    if (!bulkResults) return;
    try {
        const res = await fetch('/api/download-csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(bulkResults)
        });
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'churn_predictions.csv';
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) { console.error('Download error:', e); }
}

// ─── Model Analytics ─────────────────────────────────────────
async function loadAnalytics() {
    try {
        const res = await fetch('/api/model-analytics');
        const data = await res.json();

        // Confusion Matrices — custom HTML cards with TP/TN/FP/FN
        const grid = document.getElementById('confusion-grid');
        grid.innerHTML = '';
        for (const [name, m] of Object.entries(data.analytics)) {
            const cm = m.confusion_matrix;
            // cm[0][0]=TN, cm[0][1]=FP, cm[1][0]=FN, cm[1][1]=TP
            const TN = cm[0][0], FP = cm[0][1], FN = cm[1][0], TP = cm[1][1];
            const total = TN + FP + FN + TP;
            const acc = ((TP + TN) / total * 100).toFixed(1);
            grid.innerHTML += `
            <div class="card cm-card">
                <div class="cm-model-name">${name}</div>
                <div class="cm-accuracy">Accuracy: <strong>${acc}%</strong></div>
                <div class="cm-legend-row">
                    <span class="cm-legend-item"><span class="cm-dot cm-dot-tn"></span>True Negative</span>
                    <span class="cm-legend-item"><span class="cm-dot cm-dot-fp"></span>False Positive</span>
                    <span class="cm-legend-item"><span class="cm-dot cm-dot-fn"></span>False Negative</span>
                    <span class="cm-legend-item"><span class="cm-dot cm-dot-tp"></span>True Positive</span>
                </div>
                <div class="cm-axis-label cm-axis-predicted">← Predicted →</div>
                <div class="cm-grid-wrap">
                    <div class="cm-axis-label cm-axis-actual">← Actual →</div>
                    <div class="cm-grid">
                        <div class="cm-header-cell"></div>
                        <div class="cm-header-cell cm-pred-stay">Predicted: Stay</div>
                        <div class="cm-header-cell cm-pred-churn">Predicted: Churn</div>

                        <div class="cm-header-cell cm-actual-stay">Actual: Stay</div>
                        <div class="cm-cell cm-tn">
                            <div class="cm-cell-label">TN</div>
                            <div class="cm-cell-value">${TN.toLocaleString()}</div>
                            <div class="cm-cell-desc">Correctly predicted<br>as <strong>Stay</strong></div>
                        </div>
                        <div class="cm-cell cm-fp">
                            <div class="cm-cell-label">FP</div>
                            <div class="cm-cell-value">${FP.toLocaleString()}</div>
                            <div class="cm-cell-desc">Stayer flagged as<br><strong>Churn</strong> (wasted outreach)</div>
                        </div>

                        <div class="cm-header-cell cm-actual-churn">Actual: Churn</div>
                        <div class="cm-cell cm-fn">
                            <div class="cm-cell-label">FN ⚠️</div>
                            <div class="cm-cell-value">${FN.toLocaleString()}</div>
                            <div class="cm-cell-desc">Churner missed by<br>model — <strong>costly error</strong></div>
                        </div>
                        <div class="cm-cell cm-tp">
                            <div class="cm-cell-label">TP</div>
                            <div class="cm-cell-value">${TP.toLocaleString()}</div>
                            <div class="cm-cell-desc">Correctly predicted<br>as <strong>Churn</strong></div>
                        </div>
                    </div>
                </div>
            </div>`;
        }

        // Bar chart
        const metricNames = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'];
        const metricKeys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc'];
        const barTraces = Object.entries(data.analytics).map(([name, m]) => ({
            type: 'bar', name,
            x: metricNames, y: metricKeys.map(k => m[k]),
            marker: { color: COLORS.models[name] }
        }));
        Plotly.newPlot('metrics-chart', barTraces, {
            ...PLOTLY_THEME, height: 500, barmode: 'group',
            title: { text: '<b>Performance Metrics Comparison</b>', font: { size: 18, color: '#f0f2f5' } },
            yaxis: { ...PLOTLY_THEME.yaxis, title: 'Score', range: [0, 1.05] },
            legend: { bgcolor: 'rgba(31,41,55,0.8)' }
        }, { responsive: true, displayModeBar: false });

        // Inject insights
        const confEl = document.getElementById('confusion-insights');
        if (confEl) confEl.innerHTML = chartInsights([
            'Green TN + TP cells (diagonal) = correct predictions — a high-accuracy model maximises both',
            'Orange FN (bottom-left) = real churners the model missed — the most costly business error; minimise this',
            'Yellow FP (top-right) = loyal customers wrongly flagged — wastes retention budget on people who would have stayed'
        ]);
        const metEl = document.getElementById('metrics-insights');
        if (metEl) metEl.innerHTML = chartInsights([
            'Accuracy = overall correct predictions; useful but misleading on imbalanced datasets',
            'F1 Score = harmonic mean of Precision and Recall — the single best metric for churn models',
            'Recall (Sensitivity) is critical: a low Recall means many real churners go undetected and unretained'
        ]);

    } catch (e) { console.error('Analytics error:', e); }
}

function showAnalyticsTab(tab) {
    document.querySelectorAll('.analytics-tab').forEach(t => t.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('analytics-' + tab).style.display = 'block';
    event.target.classList.add('active');
}

// ─── Model Comparison ────────────────────────────────────────
async function loadComparison() {
    try {
        const res = await fetch('/api/dashboard');
        const data = await res.json();

        const models = Object.entries(data.model_performance);
        const metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'];
        const keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc'];
        const metricDesc = {
            'Accuracy':  'Overall % correct predictions',
            'Precision': 'Of predicted churners, how many truly churned',
            'Recall':    'Of real churners, how many were caught',
            'F1 Score':  'Balance of Precision & Recall — best single metric',
            'AUC':       'Ability to rank churners above non-churners'
        };
        const modelIcons = {
            'Logistic Regression': '📘',
            'Decision Tree': '🌳',
            'Random Forest': '🌲',
            'XGBoost': '🚀'
        };

        // Hide old plain table
        const oldTable = document.getElementById('comparison-table-body');
        if (oldTable) oldTable.closest('.card').style.display = 'none';

        // Per-metric: find best & worst values for color scaling
        const metricRanges = {};
        keys.forEach((k, i) => {
            const vals = models.map(([, m]) => parseFloat(m[k]));
            metricRanges[k] = { min: Math.min(...vals), max: Math.max(...vals) };
        });

        // Compute overall winner (highest average across all metrics)
        const avgScores = models.map(([name, m]) => ({
            name,
            avg: keys.reduce((s, k) => s + parseFloat(m[k]), 0) / keys.length
        }));
        avgScores.sort((a, b) => b.avg - a.avg);
        const overallWinner = avgScores[0].name;

        // Per metric: rank each model
        const metricRanks = {};
        keys.forEach((k, i) => {
            const sorted = [...models].sort((a, b) => parseFloat(b[1][k]) - parseFloat(a[1][k]));
            metricRanks[k] = {};
            sorted.forEach(([name], rank) => { metricRanks[k][name] = rank + 1; });
        });

        function heatColor(val, min, max) {
            // 0 = worst red, 1 = best green — interpolate
            const t = max === min ? 1 : (val - min) / (max - min);
            if (t >= 0.8) return { bg: 'rgba(16,185,129,0.18)', border: 'rgba(16,185,129,0.4)', text: '#34d399' };
            if (t >= 0.5) return { bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)', text: '#fbbf24' };
            return { bg: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.3)', text: '#f87171' };
        }

        function rankBadge(rank) {
            if (rank === 1) return `<span class="sc-rank sc-rank-1">1st</span>`;
            if (rank === 2) return `<span class="sc-rank sc-rank-2">2nd</span>`;
            if (rank === 3) return `<span class="sc-rank sc-rank-3">3rd</span>`;
            return `<span class="sc-rank sc-rank-4">4th</span>`;
        }

        // Build scorecard HTML
        let sc = `<div class="sc-wrap">`;

        // Header row
        sc += `<div class="sc-row sc-header-row">
            <div class="sc-cell sc-model-col"></div>`;
        metrics.forEach((metric, i) => {
            sc += `<div class="sc-cell sc-header-cell">
                <div class="sc-metric-name">${metric}</div>
                <div class="sc-metric-desc">${metricDesc[metric]}</div>
            </div>`;
        });
        sc += `<div class="sc-cell sc-header-cell">Avg Score</div></div>`;

        // Model rows
        models.forEach(([name, m]) => {
            const isWinner = name === overallWinner;
            sc += `<div class="sc-row${isWinner ? ' sc-winner-row' : ''}">
                <div class="sc-cell sc-model-col">
                    ${isWinner ? '<span class="sc-trophy">🏆</span>' : ''}
                    <span class="sc-model-icon">${modelIcons[name] || '🤖'}</span>
                    <span class="sc-model-name">${name}</span>
                </div>`;

            keys.forEach(k => {
                const val = parseFloat(m[k]);
                const c = heatColor(val, metricRanges[k].min, metricRanges[k].max);
                const rank = metricRanks[k][name];
                sc += `<div class="sc-cell sc-value-cell" style="background:${c.bg};border-color:${c.border};">
                    <div class="sc-value" style="color:${c.text};">${val.toFixed(1)}%</div>
                    <div class="sc-bar-wrap"><div class="sc-bar" style="width:${val}%;background:${c.text};"></div></div>
                    ${rankBadge(rank)}
                </div>`;
            });

            // Average score cell
            const avg = keys.reduce((s, k) => s + parseFloat(m[k]), 0) / keys.length;
            const avgC = heatColor(avg, Math.min(...avgScores.map(x=>x.avg)), Math.max(...avgScores.map(x=>x.avg)));
            sc += `<div class="sc-cell sc-value-cell sc-avg-cell" style="background:${avgC.bg};border-color:${avgC.border};">
                <div class="sc-value" style="color:${avgC.text};">${avg.toFixed(1)}%</div>
            </div>`;

            sc += `</div>`;
        });

        // Legend
        sc += `<div class="sc-legend">
            <span class="sc-legend-item"><span class="sc-legend-dot" style="background:rgba(16,185,129,0.5)"></span>Best in group</span>
            <span class="sc-legend-item"><span class="sc-legend-dot" style="background:rgba(245,158,11,0.5)"></span>Mid range</span>
            <span class="sc-legend-item"><span class="sc-legend-dot" style="background:rgba(239,68,68,0.5)"></span>Lowest in group</span>
        </div>`;

        sc += `</div>`;

        // Hide old plain table (scorecard replaces it)
        const plainTable = document.getElementById('comparison-plain-table');
        if (plainTable) plainTable.style.display = 'none';

        const chartEl = document.getElementById('comparison-bar-chart');
        if (chartEl) chartEl.innerHTML = sc;

        const compEl = document.getElementById('comparison-bar-insights');
        if (compEl) compEl.innerHTML = chartInsights([
            'Each cell is green (top score), amber (mid), or red (lowest) for that metric — read across a row to see one model\'s full profile',
            'F1 Score is the most important column: it balances catching real churners against false alarms — prioritise it when picking a model',
            'The 🏆 row has the highest average score across all metrics — this is the recommended model for production deployment'
        ]);

    } catch (e) { console.error('Comparison error:', e); }
}

// ─── Explainability ──────────────────────────────────────────
async function loadExplainability() {
    const loading = document.getElementById('shap-loading');
    const results = document.getElementById('shap-results');
    const model = document.getElementById('shap-model').value;

    loading.style.display = 'block';
    results.style.display = 'none';

    try {
        const res = await fetch('/api/explainability');
        const data = await res.json();
        const importance = data[model];

        loading.style.display = 'none';
        results.style.display = 'block';

        const sorted = [...importance].sort((a, b) => a.importance - b.importance);
        Plotly.newPlot('shap-chart', [{
            type: 'bar',
            y: sorted.map(f => f.feature),
            x: sorted.map(f => f.importance),
            orientation: 'h',
            marker: {
                color: sorted.map((_, i) => `hsl(${160 + i * 20}, 70%, 55%)`),
                line: { color: '#374151', width: 1 }
            },
            text: sorted.map(f => f.importance.toFixed(4)),
            textposition: 'auto'
        }], {
            ...PLOTLY_THEME, height: 450,
            title: { text: `<b>SHAP Feature Importance — ${model}</b>`, font: { size: 18, color: '#f0f2f5' } },
            xaxis: { ...PLOTLY_THEME.xaxis, title: 'Mean |SHAP Value|' }
        }, { responsive: true, displayModeBar: false });

        const shapChartInsightsEl = document.getElementById('shap-chart-insights');
        if (shapChartInsightsEl) shapChartInsightsEl.innerHTML = chartInsights([
            'Each bar represents the average absolute SHAP value — how much that feature shifts predictions across all customers',
            'Taller bars are features the model considers most when deciding churn vs. stay for any customer',
            'Use the top features to guide data collection priorities, retention campaign targeting, and business strategy'
        ]);

        // Key insights
        const top3 = importance.slice(0, 3);
        document.getElementById('shap-insights').innerHTML = `
            <div class="card-title">💡 Key Insights</div>
            <div class="grid-3">
                <div style="text-align:center;padding:16px;">
                    <div style="font-size:2rem;margin-bottom:8px;">🥇</div>
                    <div style="font-weight:700;">${top3[0]?.feature || '—'}</div>
                    <div style="color:var(--accent-green);font-size:0.85rem;">Most Important</div>
                </div>
                <div style="text-align:center;padding:16px;">
                    <div style="font-size:2rem;margin-bottom:8px;">🥈</div>
                    <div style="font-weight:700;">${top3[1]?.feature || '—'}</div>
                    <div style="color:var(--accent-blue);font-size:0.85rem;">Second Most</div>
                </div>
                <div style="text-align:center;padding:16px;">
                    <div style="font-size:2rem;margin-bottom:8px;">🥉</div>
                    <div style="font-weight:700;">${top3[2]?.feature || '—'}</div>
                    <div style="color:var(--accent-orange);font-size:0.85rem;">Third Most</div>
                </div>
            </div>
            <p style="color:var(--text-secondary);margin-top:16px;line-height:1.7;">
                These features should be prioritized when developing customer retention strategies. 
                <strong>${top3[0]?.feature}</strong> has the strongest impact on churn predictions,
                followed by <strong>${top3[1]?.feature}</strong> and <strong>${top3[2]?.feature}</strong>.
            </p>
        `;

    } catch (e) {
        loading.style.display = 'none';
        results.style.display = 'block';
        results.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error: ${e.message}</div>`;
    }
}

// ─── Decision Tree ───────────────────────────────────────────
async function runDTreePrediction() {
    const payload = {
        'Age': parseInt(document.getElementById('dt-age').value),
        'Gender': document.getElementById('dt-gender').value,
        'Tenure': parseInt(document.getElementById('dt-tenure').value),
        'Usage Frequency': parseInt(document.getElementById('dt-usage').value),
        'Support Calls': parseInt(document.getElementById('dt-support').value),
        'Payment Delay': parseInt(document.getElementById('dt-delay').value),
        'Subscription Type': document.getElementById('dt-sub').value,
        'Contract Length': document.getElementById('dt-contract').value,
        'Total Spend': parseFloat(document.getElementById('dt-spend').value),
        'Last Interaction': parseInt(document.getElementById('dt-interaction').value)
    };

    const container = document.getElementById('dtree-results');
    container.style.display = 'block';
    container.innerHTML = '<div class="spinner"></div><div class="loading-text">Tracing decision path...</div>';

    try {
        const res = await fetch('/api/decision-tree', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderDTreeResults(data);
    } catch (e) {
        container.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error: ${e.message}</div>`;
    }
}

function renderDTreeResults(data) {
    const container = document.getElementById('dtree-results');
    const isChurn = data.prediction === 1;
    const prob = (data.probability * 100).toFixed(1);

    let html = '';

    // Banner
    html += `<div class="result-banner ${isChurn ? 'churn' : 'safe'}">
        <span class="banner-icon">${isChurn ? '⚠️' : '✅'}</span>
        <div>
            <strong>${isChurn ? 'CHURN PREDICTED: Customer is likely to leave' : 'SAFE: Customer is likely to stay'}</strong>
            <div style="font-size:0.9rem;opacity:0.8;margin-top:4px;">
                Churn Probability: ${prob}% | Risk Level: ${data.risk_level} | Tree Depth: ${data.tree_depth} | Leaves: ${data.tree_leaves}
            </div>
        </div>
    </div>`;

    // Tree Visualization (full width)
    html += `<div class="chart-container" id="dtree-viz" style="margin-bottom:12px;min-height:650px;"></div>
    ${chartInsights([
        'The blue highlighted path = the exact sequence of decisions the model took for this specific customer',
        'Each node shows the feature checked and the threshold used at that split point',
        'Follow the path top to bottom to understand step-by-step why the model predicted churn or stay'
    ])}
    <div style="margin-bottom:24px;"></div>`;

    // Gauge + Feature Importance
    html += `<div class="grid-2">
        <div>
            <div class="chart-container" id="dt-gauge-chart"></div>
            ${chartInsights([
        'The Decision Tree uses rule-based logic — the gauge shows the confidence of its leaf node prediction',
        'Compare this gauge with the main Single Prediction gauge to see if both models agree',
        'High agreement between models strengthens confidence in the final churn or stay verdict'
    ])}
        </div>
        <div>
            <div class="chart-container" id="dt-fi-chart"></div>
            ${chartInsights([
        'Higher score = this feature appeared more often at important splits across the decision tree',
        'Top features reflect what the tree relies on most — they are the strongest predictors in this model',
        'Compare with the Random Forest feature importance to see if the models agree on key drivers'
    ])}
        </div>
    </div>`;

    container.innerHTML = html;

    // Render charts
    setTimeout(() => {
        // Tree graph
        if (data.tree_viz) {
            renderTreeChart(data.tree_viz, data.prediction);
        }

        // Gauge
        const color = data.probability < 0.4 ? COLORS.green : data.probability < 0.75 ? COLORS.orange : COLORS.red;
        Plotly.newPlot('dt-gauge-chart', [{
            type: 'indicator',
            mode: 'gauge+number',
            value: data.probability * 100,
            number: { suffix: '%', font: { size: 48, color: '#f0f2f5' } },
            title: { text: '<b>Churn Probability</b><br><span style="font-size:0.8em;color:' + color + '">' + data.risk_level + '</span>', font: { size: 16, color: '#f0f2f5' } },
            gauge: {
                axis: { range: [0, 100], tickcolor: '#6b7280' },
                bar: { color, thickness: 0.8 },
                bgcolor: '#1f2937',
                borderwidth: 1, bordercolor: '#374151',
                steps: [
                    { range: [0, 40], color: 'rgba(16,185,129,0.15)' },
                    { range: [40, 75], color: 'rgba(245,158,11,0.15)' },
                    { range: [75, 100], color: 'rgba(239,68,68,0.15)' }
                ]
            }
        }], { ...PLOTLY_THEME, height: 320, margin: { t: 80, b: 10, l: 30, r: 30 } }, { responsive: true, displayModeBar: false });

        // Feature Importance
        const fi = data.feature_importance;
        const sorted = [...fi].sort((a, b) => a.importance - b.importance);
        Plotly.newPlot('dt-fi-chart', [{
            type: 'bar',
            y: sorted.map(f => f.feature),
            x: sorted.map(f => f.importance),
            orientation: 'h',
            marker: {
                color: sorted.map((_, i) => `hsl(${120 + i * 24}, 65%, 55%)`),
                line: { color: '#374151', width: 1 }
            },
            text: sorted.map(f => f.importance.toFixed(4)),
            textposition: 'auto'
        }], {
            ...PLOTLY_THEME, height: 320,
            title: { text: '<b>Decision Tree Feature Importance</b>', font: { size: 14, color: '#f0f2f5' } },
            xaxis: { ...PLOTLY_THEME.xaxis, title: 'Importance' },
            margin: { l: 120, r: 30, t: 50, b: 50 }
        }, { responsive: true, displayModeBar: false });
    }, 100);
}

function renderTreeChart(viz, prediction) {
    const pathNodes = viz.nodes.filter(n => n.on_path);
    const pathLeaves = pathNodes.filter(n => n.is_leaf);
    const pathInternal = pathNodes.filter(n => !n.is_leaf);
    const otherLeaves = viz.nodes.filter(n => !n.on_path && n.is_leaf);
    const otherInternal = viz.nodes.filter(n => !n.on_path && !n.is_leaf);

    // Build edge traces
    const dimEdgeX = [], dimEdgeY = [];
    const brightEdgeX = [], brightEdgeY = [];

    viz.edges.forEach(e => {
        const [x0, y0, x1, y1, onPath] = e;
        if (onPath) {
            brightEdgeX.push(x0, x1, null);
            brightEdgeY.push(y0, y1, null);
        } else {
            dimEdgeX.push(x0, x1, null);
            dimEdgeY.push(y0, y1, null);
        }
    });

    const traces = [];

    // Dim edges (non-path)
    traces.push({
        x: dimEdgeX, y: dimEdgeY, mode: 'lines',
        line: { color: 'rgba(107,114,128,0.3)', width: 1.5 },
        hoverinfo: 'none', showlegend: false
    });

    // Bright edges (decision path)
    traces.push({
        x: brightEdgeX, y: brightEdgeY, mode: 'lines',
        line: { color: '#60a5fa', width: 5 },
        hoverinfo: 'none', showlegend: false, name: 'Decision Path'
    });

    // Non-path internal nodes
    if (otherInternal.length) {
        traces.push({
            x: otherInternal.map(n => n.x), y: otherInternal.map(n => n.y),
            mode: 'markers+text', type: 'scatter',
            marker: { size: 10, color: '#1e293b', line: { color: '#475569', width: 1.5 } },
            text: otherInternal.map(n => n.label),
            textposition: 'top center',
            textfont: { size: 9, color: '#94a3b8', family: 'Inter, sans-serif' },
            hoverinfo: 'text', showlegend: false
        });
    }

    // Non-path leaf nodes
    if (otherLeaves.length) {
        traces.push({
            x: otherLeaves.map(n => n.x), y: otherLeaves.map(n => n.y),
            mode: 'markers', type: 'scatter',
            marker: {
                size: 10,
                color: otherLeaves.map(n => n.label === 'Churn' ? 'rgba(239,68,68,0.35)' : 'rgba(16,185,129,0.35)'),
                line: { color: otherLeaves.map(n => n.label === 'Churn' ? 'rgba(239,68,68,0.7)' : 'rgba(16,185,129,0.7)'), width: 1.5 }
            },
            text: otherLeaves.map(n => n.label), hoverinfo: 'text', showlegend: false
        });
    }

    // Path internal nodes (highlighted, labeled)
    if (pathInternal.length) {
        traces.push({
            x: pathInternal.map(n => n.x), y: pathInternal.map(n => n.y),
            mode: 'markers+text', type: 'scatter',
            marker: {
                size: 22,
                color: 'rgba(59,130,246,0.95)',
                line: { color: '#bfdbfe', width: 3 },
                symbol: 'circle'
            },
            text: pathInternal.map(n => n.label),
            textposition: 'top center',
            textfont: { size: 11, color: '#bfdbfe', family: 'Inter, sans-serif' },
            hoverinfo: 'text', showlegend: false, name: 'Decision Path'
        });
    }

    // Path leaf node (large diamond, result)
    if (pathLeaves.length) {
        const leaf = pathLeaves[0];
        const leafColor = leaf.label === 'Churn' ? '#ef4444' : '#10b981';
        const leafBorder = leaf.label === 'Churn' ? '#fca5a5' : '#6ee7b7';
        traces.push({
            x: [leaf.x], y: [leaf.y],
            mode: 'markers+text', type: 'scatter',
            marker: { size: 32, color: leafColor, line: { color: leafBorder, width: 4 }, symbol: 'diamond' },
            text: [leaf.label === 'Churn' ? '⬥ CHURN' : '⬥ STAY'],
            textposition: 'top center',
            textfont: { size: 14, color: '#f0f2f5', family: 'Inter, sans-serif' },
            hoverinfo: 'text', showlegend: false
        });
    }

    Plotly.newPlot('dtree-viz', traces, {
        ...PLOTLY_THEME,
        height: 800,
        title: {
            text: '<b>Decision Tree — Decision Path Highlighted in Blue</b><br><span style="font-size:0.78em;color:#94a3b8">Scroll to zoom • Drag to pan • Blue path = your customer\'s exact prediction route</span>',
            font: { size: 16, color: '#f0f2f5' }
        },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false, fixedrange: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false, fixedrange: false },
        margin: { l: 30, r: 30, t: 90, b: 30 },
        dragmode: 'pan',
        plot_bgcolor: 'rgba(10,15,28,0.8)'
    }, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['autoScale2d', 'lasso2d', 'select2d'] });
}



// ─── Drag & Drop ─────────────────────────────────────────────

const dropZone = document.getElementById('upload-zone');
if (dropZone) {
    ['dragenter', 'dragover'].forEach(ev => dropZone.addEventListener(ev, e => {
        e.preventDefault(); dropZone.classList.add('dragover');
    }));
    ['dragleave', 'drop'].forEach(ev => dropZone.addEventListener(ev, e => {
        e.preventDefault(); dropZone.classList.remove('dragover');
    }));
    dropZone.addEventListener('drop', e => {
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) {
            bulkFile = file;
            document.getElementById('file-info').style.display = 'block';
            document.getElementById('file-info').innerHTML = `<div style="color:var(--accent-green);font-weight:600;">✅ File dropped: ${file.name} (${(file.size / 1024).toFixed(1)} KB)</div>`;
            document.getElementById('bulk-predict-btn').style.display = 'block';
        }
    });
}

// ─── Init ────────────────────────────────────────────────────
