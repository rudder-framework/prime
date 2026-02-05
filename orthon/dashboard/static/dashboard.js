// PRISM Live Dashboard JavaScript

let ws = null;
let isConnected = false;
let effDimChart = null;
let eigenvalsChart = null;
let updateRate = 0;
let lastEffDim = null;
let lastEigenval = null;

// Chart data storage
const maxDataPoints = 200;
const effDimData = [];
const eigenval1Data = [];
const eigenval2Data = [];
const eigenval3Data = [];

// Initialize charts
function initCharts() {
    // Effective Dimension Chart
    const effDimCtx = document.getElementById('eff-dim-chart').getContext('2d');
    effDimChart = new Chart(effDimCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Effective Dimension',
                data: effDimData,
                borderColor: '#4cc9f0',
                backgroundColor: 'rgba(76, 201, 240, 0.1)',
                fill: true,
                tension: 0.2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    type: 'linear',
                    display: true,
                    title: { display: true, text: 'Sample', color: '#a0a0a0' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#a0a0a0' }
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Eff Dim', color: '#a0a0a0' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#a0a0a0' },
                    suggestedMin: 0,
                    suggestedMax: 5
                }
            },
            plugins: {
                legend: { labels: { color: '#eaeaea' } }
            }
        }
    });

    // Eigenvalues Chart
    const eigenvalsCtx = document.getElementById('eigenvals-chart').getContext('2d');
    eigenvalsChart = new Chart(eigenvalsCtx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Eigenvalue 1',
                    data: eigenval1Data,
                    borderColor: '#4caf50',
                    backgroundColor: 'transparent',
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'Eigenvalue 2',
                    data: eigenval2Data,
                    borderColor: '#ffc107',
                    backgroundColor: 'transparent',
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'Eigenvalue 3',
                    data: eigenval3Data,
                    borderColor: '#9c27b0',
                    backgroundColor: 'transparent',
                    tension: 0.2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    type: 'linear',
                    display: true,
                    title: { display: true, text: 'Sample', color: '#a0a0a0' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#a0a0a0' }
                },
                y: {
                    type: 'logarithmic',
                    display: true,
                    title: { display: true, text: 'Eigenvalue (log)', color: '#a0a0a0' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#a0a0a0' }
                }
            },
            plugins: {
                legend: { labels: { color: '#eaeaea' } }
            }
        }
    });
}

// Connect to WebSocket
function connect() {
    const sourceType = document.getElementById('source-select').value;
    const wsUrl = `ws://${window.location.host}/ws/${sourceType}`;

    console.log(`Connecting to ${wsUrl}`);

    ws = new WebSocket(wsUrl);

    ws.onopen = function() {
        console.log('WebSocket connected');
        setConnectionStatus('connected', 'Connected');
        isConnected = true;
        document.getElementById('connect-btn').textContent = 'Disconnect';
        document.getElementById('connect-btn').classList.add('connected');
        document.getElementById('data-source').textContent = sourceType;
    };

    ws.onclose = function() {
        console.log('WebSocket disconnected');
        setConnectionStatus('disconnected', 'Disconnected');
        isConnected = false;
        document.getElementById('connect-btn').textContent = 'Connect';
        document.getElementById('connect-btn').classList.remove('connected');
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        setConnectionStatus('error', 'Error');
    };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
}

// Disconnect WebSocket
function disconnect() {
    if (ws) {
        ws.close();
        ws = null;
    }
}

// Toggle connection
function toggleConnection() {
    if (isConnected) {
        disconnect();
    } else {
        connect();
    }
}

// Set connection status
function setConnectionStatus(status, text) {
    const dot = document.getElementById('status-dot');
    const textEl = document.getElementById('status-text');

    dot.className = 'status-dot ' + status;
    textEl.textContent = text;
}

// Handle incoming messages
function handleMessage(data) {
    if (data.type === 'status') {
        console.log('Status:', data.message);
        return;
    }

    if (data.type === 'error') {
        console.error('Error:', data.message);
        return;
    }

    if (data.type === 'analysis_update') {
        updateMetrics(data.instant_results, data.batch_results);
        updateCharts(data.instant_results);
        updateAlerts(data.alerts);
        updateSystemInfo(data.status_summary, data.update_rate);
        updateRawData(data.raw_data);
    }
}

// Update metrics display
function updateMetrics(instant, batch) {
    if (instant.status === 'active') {
        // Effective dimension
        const effDim = instant.eff_dim;
        const effDimEl = document.getElementById('eff-dim-value');
        effDimEl.textContent = effDim.toFixed(3);

        // Update trend
        const effDimTrend = document.getElementById('eff-dim-trend');
        if (lastEffDim !== null) {
            const diff = effDim - lastEffDim;
            if (Math.abs(diff) > 0.001) {
                effDimTrend.textContent = diff > 0 ? `+${diff.toFixed(3)}` : diff.toFixed(3);
                effDimTrend.className = 'metric-trend ' + (diff > 0 ? 'up' : 'down');
            }
        }
        lastEffDim = effDim;

        // Color coding based on thresholds
        const effDimCard = document.getElementById('eff-dim-card');
        effDimCard.className = 'metric-card';
        if (effDim < 1.5) {
            effDimCard.classList.add('critical');
        } else if (effDim < 2.0) {
            effDimCard.classList.add('warning');
        }

        // Primary eigenvalue
        const eigenval = instant.eigenval_1;
        const eigenvalEl = document.getElementById('eigenval-value');
        eigenvalEl.textContent = eigenval.toExponential(3);

        // Eigenvalue trend
        const eigenvalTrend = document.getElementById('eigenval-trend');
        if (lastEigenval !== null) {
            const diff = eigenval - lastEigenval;
            const pctDiff = (diff / lastEigenval) * 100;
            if (Math.abs(pctDiff) > 0.1) {
                eigenvalTrend.textContent = pctDiff > 0 ? `+${pctDiff.toFixed(1)}%` : `${pctDiff.toFixed(1)}%`;
                eigenvalTrend.className = 'metric-trend ' + (pctDiff > 0 ? 'up' : 'down');
            }
        }
        lastEigenval = eigenval;

        // Analysis stage
        const stageEl = document.getElementById('stage-value');
        stageEl.textContent = formatStage(instant.analysis_stage);

        // Progress bar
        const progressEl = document.getElementById('analysis-progress');
        const stageProgress = getStageProgress(instant.analysis_stage);
        progressEl.style.width = stageProgress + '%';
    }

    // Lyapunov (from batch results)
    if (batch && batch.lyapunov !== undefined) {
        const lyapEl = document.getElementById('lyapunov-value');
        lyapEl.textContent = batch.lyapunov.toFixed(6);

        const lyapStatus = document.getElementById('lyapunov-status');
        lyapStatus.textContent = batch.lyapunov_status;
        lyapStatus.className = 'metric-status ' + batch.lyapunov_status;
    }
}

// Format analysis stage for display
function formatStage(stage) {
    const stages = {
        'initializing': 'Initializing...',
        'computing_eigenstructure': 'Computing Eigenstructure',
        'analyzing_dynamics': 'Analyzing Dynamics',
        'computing_complexity': 'Computing Complexity',
        'full_analysis': 'Full Analysis Active'
    };
    return stages[stage] || stage;
}

// Get progress percentage for stage
function getStageProgress(stage) {
    const progress = {
        'initializing': 10,
        'computing_eigenstructure': 30,
        'analyzing_dynamics': 50,
        'computing_complexity': 75,
        'full_analysis': 100
    };
    return progress[stage] || 0;
}

// Update charts
function updateCharts(instant) {
    if (instant.status !== 'active') return;

    const sampleCount = instant.sample_count;

    // Add new data points
    effDimData.push({ x: sampleCount, y: instant.eff_dim });
    eigenval1Data.push({ x: sampleCount, y: instant.eigenval_1 || 0.001 });
    eigenval2Data.push({ x: sampleCount, y: instant.eigenval_2 || 0.001 });
    eigenval3Data.push({ x: sampleCount, y: instant.eigenval_3 || 0.001 });

    // Remove old data points
    while (effDimData.length > maxDataPoints) {
        effDimData.shift();
        eigenval1Data.shift();
        eigenval2Data.shift();
        eigenval3Data.shift();
    }

    // Update charts
    effDimChart.update('none');
    eigenvalsChart.update('none');
}

// Update alerts display
function updateAlerts(alerts) {
    const container = document.getElementById('alerts-container');

    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="no-alerts">No active alerts</div>';
        return;
    }

    container.innerHTML = alerts.map(alert => `
        <div class="alert-item ${alert.level}">
            <div class="alert-header">
                <span class="alert-level">${alert.level}</span>
                <span class="alert-time">${formatTime(alert.timestamp)}</span>
            </div>
            <div class="alert-message">${alert.message}</div>
        </div>
    `).join('');
}

// Format timestamp
function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
}

// Update system info
function updateSystemInfo(summary, rate) {
    if (summary) {
        document.getElementById('sample-count').textContent = summary.sample_count.toLocaleString();
        document.getElementById('uptime').textContent = formatUptime(summary.uptime_seconds);

        // Buffer progress
        const windowFill = Math.round(summary.window_fill * 100);
        const batchFill = Math.round(summary.batch_fill * 100);

        document.getElementById('window-fill').textContent = windowFill + '%';
        document.getElementById('window-progress').style.width = windowFill + '%';

        document.getElementById('batch-fill').textContent = batchFill + '%';
        document.getElementById('batch-progress').style.width = batchFill + '%';
    }

    if (rate !== undefined && rate !== null) {
        document.getElementById('update-rate').textContent = rate.toFixed(1) + ' Hz';
    }
}

// Format uptime
function formatUptime(seconds) {
    if (seconds < 60) return Math.round(seconds) + 's';
    if (seconds < 3600) return Math.round(seconds / 60) + 'm ' + Math.round(seconds % 60) + 's';
    return Math.round(seconds / 3600) + 'h ' + Math.round((seconds % 3600) / 60) + 'm';
}

// Update raw data display
function updateRawData(rawData) {
    const container = document.getElementById('raw-data-container');

    if (!rawData || Object.keys(rawData).length === 0) {
        return;
    }

    container.innerHTML = Object.entries(rawData).map(([name, value]) => `
        <div class="sensor-item">
            <div class="sensor-name">${name}</div>
            <div class="sensor-value">${formatSensorValue(value)}</div>
        </div>
    `).join('');
}

// Format sensor value
function formatSensorValue(value) {
    if (Math.abs(value) >= 1000) {
        return value.toExponential(2);
    } else if (Math.abs(value) < 0.01) {
        return value.toExponential(2);
    } else {
        return value.toFixed(2);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    console.log('PRISM Dashboard initialized');
});
