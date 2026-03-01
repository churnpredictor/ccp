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

// ─── Navigation ──────────────────────────────────────────────
function navigateTo(page) {
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelector(`.nav-item[data-page="${page}"]`).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (page === 'home') loadDashboard();
    if (page === 'history') loadCustomerHistory();
    if (page === 'analytics') loadAnalytics();
    if (page === 'comparison') loadComparison();
    if (page === 'explainability') loadExplainability();
}

// ─── Dashboard ───────────────────────────────────────────────
async function loadDashboard() {
    try {
        const res = await fetch('/api/dashboard');
        const data = await res.json();

        animateNumber('kpi-total', data.total_customers);
        animateNumber('kpi-active', data.active);
        animateNumber('kpi-churned', data.churned);
        document.getElementById('kpi-rate').textContent = data.churn_rate + '%';

        const tbody = document.getElementById('home-perf-body');
        tbody.innerHTML = '';
        for (const [name, m] of Object.entries(data.model_performance)) {
            const best = name === data.best_model;
            tbody.innerHTML += `<tr style="${best ? 'background:rgba(16,185,129,0.08)' : ''}">
                <td style="font-weight:700;">${best ? '🏆 ' : ''}${name}</td>
                <td>${m.accuracy}%</td><td>${m.precision}%</td>
                <td>${m.recall}%</td><td>${m.f1_score}%</td><td>${m.auc}%</td>
            </tr>`;
        }
        document.getElementById('home-best-model').textContent =
            `🏆 Best Model: ${data.best_model} (F1 Score: ${data.best_f1}%)`;
    } catch (e) { console.error('Dashboard load error:', e); }
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
        renderPredictionResults(data);
    } catch (e) {
        container.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error: ${e.message}</div>`;
    }
}

function renderPredictionResults(data) {
    const container = document.getElementById('prediction-results');
    const isChurn = data.prediction === 1;
    const prob = (data.probability * 100).toFixed(1);
    const reasons = data.reasons || {};

    let html = '';

    // Banner
    html += `<div class="result-banner ${isChurn ? 'churn' : 'safe'}">
        <span class="banner-icon">${isChurn ? '⚠️' : '✅'}</span>
        <div>
            <strong>${isChurn ? 'HIGH ALERT: Customer is likely to CHURN' : 'SAFE: Customer is likely to STAY'}</strong>
            <div style="font-size:0.9rem;opacity:0.8;margin-top:4px;">
                Churn Probability: ${prob}% | Risk Level: ${data.risk_level}
            </div>
        </div>
    </div>`;

    // Gauge + Distribution charts
    html += `<div class="grid-2">
        <div class="chart-container" id="gauge-chart"></div>
        <div class="chart-container" id="dist-chart"></div>
    </div>`;

    // Churn Reasons Section
    html += `<div class="card" style="margin-bottom:24px;">
        <div class="card-title">🔍 Why This Customer Will ${isChurn ? 'Churn' : 'Stay'}</div>
        <div class="grid-2">
            <div>
                <h4 style="color:${COLORS.red};margin-bottom:14px;">⚠️ Risk Factors</h4>
                ${(reasons.churn_reasons || []).length > 0
            ? reasons.churn_reasons.map(r => `<div class="reason-card risk">
                        <span class="reason-icon">⚠️</span>
                        <div class="reason-text"><strong>${r.feature}:</strong> ${r.reason}</div>
                    </div>`).join('')
            : '<div style="color:var(--text-muted);padding:12px;">No significant risk factors detected ✨</div>'
        }
            </div>
            <div>
                <h4 style="color:${COLORS.green};margin-bottom:14px;">✅ Retention Factors</h4>
                ${(reasons.stay_reasons || []).length > 0
            ? reasons.stay_reasons.map(r => `<div class="reason-card safe">
                        <span class="reason-icon">✅</span>
                        <div class="reason-text"><strong>${r.feature}:</strong> ${r.reason}</div>
                    </div>`).join('')
            : '<div style="color:var(--text-muted);padding:12px;">No strong retention factors ⚠️</div>'
        }
            </div>
        </div>
    </div>`;

    // Feature Impact Chart
    html += `<div class="chart-container" id="impact-chart" style="margin-bottom:24px;"></div>`;

    // Recommendations
    html += `<div class="card" style="margin-bottom:24px;">
        <div class="card-title">💡 Actionable Recommendations</div>
        ${(reasons.recommendations || []).map(r => `<div class="recommendation">
            <span>💡</span><span>${r}</span>
        </div>`).join('')}
    </div>`;

    // All Models Comparison
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

    // Feature Importance
    html += `<div class="chart-container" id="fi-chart" style="margin-bottom:24px;"></div>`;

    container.innerHTML = html;

    // Render charts after HTML is inserted
    setTimeout(() => {
        renderGaugeChart(data.probability, data.risk_level);
        renderDistChart(data.probability);
        renderImpactChart(reasons.feature_impacts || []);
        renderFeatureImportance(data.feature_importance);
    }, 100);
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

// ─── Customer History & Search ───────────────────────────────
function toggleDateFilter() {
    const isEnabled = document.getElementById('enable-date-filter').checked;
    const inputs = document.getElementById('date-inputs-container').querySelectorAll('input');
    inputs.forEach(input => {
        input.disabled = !isEnabled;
        if (!isEnabled) input.style.opacity = '0.5';
        else input.style.opacity = '1';
    });
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

async function loadCustomerHistory() {
    const searchInput = document.getElementById('history-search-input').value.trim();
    const useDates = document.getElementById('enable-date-filter').checked;

    let url = '/api/customer_history?';
    if (searchInput) {
        url += `search=${encodeURIComponent(searchInput)}`;
    } else if (useDates) {
        const start = document.getElementById('history-start-date').value;
        const end = document.getElementById('history-end-date').value;
        if (start && end) {
            url += `start=${start}&end=${end}`;
        }
    }

    const resultsDiv = document.getElementById('history-results');
    resultsDiv.style.display = 'block';
    resultsDiv.style.opacity = '0.5';

    try {
        const res = await fetch(url);
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        // Update KPIs
        animateNumber('hist-kpi-found', data.metrics.total_found);

        // Format Spend KPI properly
        const startVal = parseFloat(document.getElementById('hist-kpi-spend').textContent.replace(/[^0-9.-]+/g, "")) || 0;
        const targetVal = data.metrics.total_spend;

        // Simple manual animation for currency
        const duration = 1200;
        const startTime = performance.now();
        const el = document.getElementById('hist-kpi-spend');

        function updateCurrency(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = startVal + (targetVal - startVal) * eased;

            el.textContent = '$' + current.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            if (progress < 1) requestAnimationFrame(updateCurrency);
        }
        requestAnimationFrame(updateCurrency);

        animateNumber('hist-kpi-churned', data.metrics.churned_in_period);

        // Render Table Headers
        const thead = document.getElementById('history-thead');
        thead.innerHTML = `<tr>${data.columns.map(col => `<th>${col}</th>`).join('')}</tr>`;

        // Render Table Body
        const tbody = document.getElementById('history-tbody');
        if (data.data.length === 0) {
            tbody.innerHTML = `<tr><td colspan="${data.columns.length}" style="text-align:center; padding: 24px;">No matching customers found.</td></tr>`;
        } else {
            tbody.innerHTML = data.data.map(row => {
                const cells = data.columns.map(col => {
                    let val = row[col];
                    if (val === null || val === 'null' || val === 'None') val = '-';
                    // Special coloring for Churn Date
                    if (col === 'Churn Date' && val !== '-') {
                        val = `<span class="badge badge-danger" style="font-size:0.75rem;">${val}</span>`;
                    }
                    if (col === 'Customer Name') {
                        val = `<strong>${val}</strong>`;
                    }
                    return `<td>${val}</td>`;
                }).join('');
                return `<tr>${cells}</tr>`;
            }).join('');
        }

        resultsDiv.style.opacity = '1';

    } catch (e) {
        resultsDiv.style.opacity = '1';
        resultsDiv.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error fetching history: ${e.message}</div>`;
    }
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

        // Confusion Matrices
        const grid = document.getElementById('confusion-grid');
        grid.innerHTML = '';
        for (const [name, m] of Object.entries(data.analytics)) {
            grid.innerHTML += `<div class="chart-container" id="cm-${name.replace(/\s/g, '')}"></div>`;
        }
        setTimeout(() => {
            for (const [name, m] of Object.entries(data.analytics)) {
                const cm = m.confusion_matrix;
                Plotly.newPlot('cm-' + name.replace(/\s/g, ''), [{
                    type: 'heatmap', z: cm, x: ['Stay', 'Churn'], y: ['Stay', 'Churn'],
                    colorscale: [[0, '#1f2937'], [0.5, '#f97316'], [1, '#ef4444']],
                    text: cm.map(r => r.map(v => v.toString())),
                    texttemplate: '%{text}', textfont: { size: 20, color: '#fff' },
                    showscale: false
                }], {
                    ...PLOTLY_THEME, height: 320,
                    title: { text: `<b>${name}</b>`, font: { size: 14, color: '#f0f2f5' } },
                    xaxis: { title: 'Predicted' }, yaxis: { title: 'Actual', autorange: 'reversed' }
                }, { responsive: true, displayModeBar: false });
            }
        }, 100);

        // ROC Curves
        const traces = [];
        for (const [name, r] of Object.entries(data.roc)) {
            traces.push({
                type: 'scatter', mode: 'lines', x: r.fpr, y: r.tpr,
                name: `${name} (AUC=${r.auc.toFixed(3)})`,
                line: { color: COLORS.models[name], width: 3 }
            });
        }
        traces.push({
            type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1],
            name: 'Random', line: { color: '#6b7280', dash: 'dash', width: 2 }
        });
        Plotly.newPlot('roc-chart', traces, {
            ...PLOTLY_THEME, height: 500,
            title: { text: '<b>ROC Curve Comparison</b>', font: { size: 18, color: '#f0f2f5' } },
            xaxis: { ...PLOTLY_THEME.xaxis, title: 'False Positive Rate', range: [0, 1] },
            yaxis: { ...PLOTLY_THEME.yaxis, title: 'True Positive Rate', range: [0, 1] },
            legend: { bgcolor: 'rgba(31,41,55,0.8)', bordercolor: '#374151', borderwidth: 1 }
        }, { responsive: true, displayModeBar: false });

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

        const tbody = document.getElementById('comparison-table-body');
        tbody.innerHTML = '';
        for (const [name, m] of Object.entries(data.model_performance)) {
            const best = name === data.best_model;
            tbody.innerHTML += `<tr style="${best ? 'background:rgba(16,185,129,0.08)' : ''}">
                <td style="font-weight:700;">${best ? '🏆 ' : ''}${name}</td>
                <td>${m.accuracy}%</td><td>${m.precision}%</td>
                <td>${m.recall}%</td><td>${m.f1_score}%</td><td>${m.auc}%</td>
            </tr>`;
        }

        // Radar chart
        const metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'];
        const keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc'];
        const traces = Object.entries(data.model_performance).map(([name, m]) => ({
            type: 'scatterpolar',
            r: keys.map(k => m[k]),
            theta: metrics,
            fill: 'toself',
            name, fillcolor: COLORS.models[name] + '20',
            line: { color: COLORS.models[name], width: 2 }
        }));
        Plotly.newPlot('radar-chart', traces, {
            ...PLOTLY_THEME, height: 500,
            polar: {
                radialaxis: { visible: true, range: [0, 100], gridcolor: 'rgba(255,255,255,0.06)' },
                angularaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
                bgcolor: 'rgba(17,24,39,0.5)'
            },
            legend: { bgcolor: 'rgba(31,41,55,0.8)' }
        }, { responsive: true, displayModeBar: false });

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
    html += `<div class="chart-container" id="dtree-viz" style="margin-bottom:24px;min-height:650px;"></div>`;

    // Gauge + Feature Importance
    html += `<div class="grid-2">
        <div class="chart-container" id="dt-gauge-chart"></div>
        <div class="chart-container" id="dt-fi-chart"></div>
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
        line: { color: 'rgba(107,114,128,0.15)', width: 1 },
        hoverinfo: 'none', showlegend: false
    });

    // Bright edges (decision path)
    traces.push({
        x: brightEdgeX, y: brightEdgeY, mode: 'lines',
        line: { color: '#3b82f6', width: 3.5 },
        hoverinfo: 'none', showlegend: false, name: 'Decision Path'
    });

    // Non-path internal nodes (tiny dots)
    if (otherInternal.length) {
        traces.push({
            x: otherInternal.map(n => n.x), y: otherInternal.map(n => n.y),
            mode: 'markers', type: 'scatter',
            marker: { size: 5, color: '#1f2937', line: { color: '#374151', width: 1 } },
            text: otherInternal.map(n => n.label), hoverinfo: 'text', showlegend: false
        });
    }

    // Non-path leaf nodes (small colored dots)
    if (otherLeaves.length) {
        traces.push({
            x: otherLeaves.map(n => n.x), y: otherLeaves.map(n => n.y),
            mode: 'markers', type: 'scatter',
            marker: {
                size: 8,
                color: otherLeaves.map(n => n.label === 'Churn' ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.25)'),
                line: { color: otherLeaves.map(n => n.label === 'Churn' ? 'rgba(239,68,68,0.5)' : 'rgba(16,185,129,0.5)'), width: 1 }
            },
            text: otherLeaves.map(n => n.label), hoverinfo: 'text', showlegend: false
        });
    }

    // Path internal nodes (large, labeled)
    if (pathInternal.length) {
        traces.push({
            x: pathInternal.map(n => n.x), y: pathInternal.map(n => n.y),
            mode: 'markers+text', type: 'scatter',
            marker: { size: 18, color: 'rgba(59,130,246,0.9)', line: { color: '#60a5fa', width: 2 }, symbol: 'circle' },
            text: pathInternal.map(n => n.label),
            textposition: 'top center',
            textfont: { size: 10, color: '#93c5fd', family: 'Inter, sans-serif' },
            hoverinfo: 'text', showlegend: false, name: 'Path Nodes'
        });
    }

    // Path leaf node (large diamond, prominent)
    if (pathLeaves.length) {
        const leaf = pathLeaves[0];
        const leafColor = leaf.label === 'Churn' ? '#ef4444' : '#10b981';
        traces.push({
            x: [leaf.x], y: [leaf.y],
            mode: 'markers+text', type: 'scatter',
            marker: { size: 26, color: leafColor, line: { color: '#f0f2f5', width: 3 }, symbol: 'diamond' },
            text: [leaf.label === 'Churn' ? '🔴 CHURN' : '🟢 STAY'],
            textposition: 'top center',
            textfont: { size: 13, color: '#f0f2f5', family: 'Inter, sans-serif' },
            hoverinfo: 'text', showlegend: false
        });
    }

    Plotly.newPlot('dtree-viz', traces, {
        ...PLOTLY_THEME,
        height: 650,
        title: { text: '<b>🌳 Full Decision Tree — Decision Path Highlighted</b><br><span style="font-size:0.75em;color:#9ca3af">Zoom and pan to explore • Blue path = your customer\'s prediction route</span>', font: { size: 16, color: '#f0f2f5' } },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false, fixedrange: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false, fixedrange: false },
        margin: { l: 20, r: 20, t: 80, b: 20 },
        dragmode: 'pan'
    }, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['autoScale2d', 'lasso2d', 'select2d'] });
}


// ─── Monthly Churn Trends ────────────────────────────────────
async function loadMonthlyChurn() {
    const fromMonth = document.getElementById('monthly-from').value;
    const toMonth = document.getElementById('monthly-to').value;

    const payload = {};
    if (fromMonth) payload.from_month = fromMonth;
    if (toMonth) payload.to_month = toMonth;

    const kpis = document.getElementById('monthly-kpis');
    const results = document.getElementById('monthly-results');
    kpis.style.display = 'none';
    results.style.display = 'none';
    results.innerHTML = '<div class="spinner"></div><div class="loading-text">Loading monthly data...</div>';
    results.style.display = 'block';

    try {
        const res = await fetch('/api/monthly-churn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Set month pickers to available range if not set
        if (!fromMonth && data.available_months.length) {
            document.getElementById('monthly-from').value = data.available_months[0];
        }
        if (!toMonth && data.available_months.length) {
            document.getElementById('monthly-to').value = data.available_months[data.available_months.length - 1];
        }

        renderMonthlyResults(data);
    } catch (e) {
        results.innerHTML = `<div class="result-banner churn"><span class="banner-icon">❌</span> Error: ${e.message}</div>`;
    }
}

function renderMonthlyResults(data) {
    const kpis = document.getElementById('monthly-kpis');
    const results = document.getElementById('monthly-results');
    const s = data.summary;

    // KPI Cards
    kpis.innerHTML = `
        <div class="kpi-card">
            <div class="kpi-icon">👥</div>
            <div class="kpi-value">${s.total_customers.toLocaleString()}</div>
            <div class="kpi-label">Total Customers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🔴</div>
            <div class="kpi-value" style="color:${COLORS.red}">${s.total_churned.toLocaleString()}</div>
            <div class="kpi-label">Churned</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🟢</div>
            <div class="kpi-value" style="color:${COLORS.green}">${s.total_stayed.toLocaleString()}</div>
            <div class="kpi-label">Retained</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📈</div>
            <div class="kpi-value" style="color:${COLORS.orange}">${s.avg_churn_rate}%</div>
            <div class="kpi-label">Avg Churn Rate</div>
        </div>`;
    kpis.style.display = 'grid';

    // Reset results area
    results.innerHTML = `
        <div class="chart-container" id="monthly-bar-chart" style="margin-bottom:24px;"></div>
        <div class="chart-container" id="monthly-rate-chart"></div>`;
    results.style.display = 'block';

    setTimeout(() => {
        // Grouped Bar Chart — Churned vs Stayed per month
        Plotly.newPlot('monthly-bar-chart', [
            {
                x: data.months, y: data.stayed, name: 'Stayed',
                type: 'bar',
                marker: { color: 'rgba(16,185,129,0.7)', line: { color: COLORS.green, width: 1 } }
            },
            {
                x: data.months, y: data.churned, name: 'Churned',
                type: 'bar',
                marker: { color: 'rgba(239,68,68,0.7)', line: { color: COLORS.red, width: 1 } }
            }
        ], {
            ...PLOTLY_THEME, height: 420, barmode: 'group',
            title: { text: '<b>Monthly Churn vs Retained Customers</b>', font: { size: 16, color: '#f0f2f5' } },
            xaxis: { ...PLOTLY_THEME.xaxis, title: 'Month', tickangle: -45 },
            yaxis: { ...PLOTLY_THEME.yaxis, title: 'Number of Customers' },
            legend: { font: { color: '#f0f2f5' }, bgcolor: 'transparent' },
            margin: { l: 60, r: 30, t: 60, b: 80 }
        }, { responsive: true, displayModeBar: false });

        // Churn Rate Trend Line
        Plotly.newPlot('monthly-rate-chart', [{
            x: data.months, y: data.churn_rate,
            type: 'scatter', mode: 'lines+markers',
            name: 'Churn Rate',
            line: { color: COLORS.orange, width: 3, shape: 'spline' },
            marker: { size: 8, color: COLORS.orange, line: { color: '#f0f2f5', width: 1 } },
            fill: 'tozeroy',
            fillcolor: 'rgba(245,158,11,0.08)'
        }], {
            ...PLOTLY_THEME, height: 350,
            title: { text: '<b>Monthly Churn Rate Trend (%)</b>', font: { size: 16, color: '#f0f2f5' } },
            xaxis: { ...PLOTLY_THEME.xaxis, title: 'Month', tickangle: -45 },
            yaxis: { ...PLOTLY_THEME.yaxis, title: 'Churn Rate (%)', rangemode: 'tozero' },
            margin: { l: 60, r: 30, t: 60, b: 80 }
        }, { responsive: true, displayModeBar: false });
    }, 100);
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
loadDashboard();
