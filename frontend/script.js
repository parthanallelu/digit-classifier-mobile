const canvas = document.getElementById('digitCanvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const predictionValue = document.getElementById('predictionValue');
const confidenceValue = document.getElementById('confidenceValue');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const terminal = document.getElementById('terminal');
const waterBarsContainer = document.getElementById('waterBars');

// Containers for toggling prediction results
const predictionPlaceholder = document.getElementById('prediction-placeholder');
const predictionResult = document.getElementById('prediction-result');

// Vision Viz Canvases
const vizRaw = document.getElementById('viz-raw').getContext('2d');
const vizCentered = document.getElementById('viz-centered').getContext('2d');

let isDrawing = false;

// Initial Setup
function init() {
    setupCanvas();
    createBars();
    setupTabs();
    addLog('NEURAL_CORE_OS V2.1.0 Loaded.', 'stage');
    addLog('Waiting for input stream...');
    fetchModelStats();
}

function setupCanvas() {
    ctx.fillStyle = 'white'; // White background for the canvas
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 14;
    ctx.strokeStyle = '#111827'; // Dark stroke
}

function setupTabs() {
    const tabInf = document.getElementById('tab-inference');
    const tabDiag = document.getElementById('tab-diagnostics');
    const viewInf = document.getElementById('inference-view');
    const viewDiag = document.getElementById('diagnostics-view');

    tabInf.addEventListener('click', () => {
        tabInf.classList.add('active');
        tabDiag.classList.remove('active');
        viewInf.classList.remove('hide');
        viewDiag.classList.add('hide');
    });

    tabDiag.addEventListener('click', () => {
        tabDiag.classList.add('active');
        tabInf.classList.remove('active');
        viewDiag.classList.remove('hide');
        viewInf.classList.add('hide');
    });
}

function createBars() {
    waterBarsContainer.innerHTML = '';
    for (let i = 0; i <= 9; i++) {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-label">${i}</div>
            <div class="prob-bar-bg">
                <div class="prob-fill" id="fill-${i}" style="width: 0%"></div>
            </div>
            <div class="prob-value" id="val-${i}">0%</div>
        `;
        waterBarsContainer.appendChild(row);
    }
}

function addLog(msg, type = 'normal') {
    const p = document.createElement('p');
    p.className = 'log-entry';
    p.innerHTML = `<span class="timestamp">[${new Date().toLocaleTimeString()}]</span> > ${msg}`;
    terminal.appendChild(p);
    terminal.scrollTop = terminal.scrollHeight;
}

// Canvas Interaction
function getXY(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * (canvas.width / rect.width),
        y: (e.clientY - rect.top) * (canvas.height / rect.height)
    };
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const pos = getXY(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    document.getElementById('canvas-placeholder').classList.add('hide');
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        const pos = getXY(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }
});

window.addEventListener('mouseup', () => {
    isDrawing = false;
});

clearBtn.addEventListener('click', () => {
    setupCanvas();
    predictionPlaceholder.classList.remove('hide');
    predictionResult.classList.add('hide');
    document.getElementById('canvas-placeholder').classList.remove('hide');
    
    predictionValue.innerText = '-';
    confidenceValue.innerText = '0.0%';
    statusText.innerText = 'Idle';
    statusBadge.className = 'status-pill-minimal';
    
    resetBars();
    clearViz();
    addLog('Matrix reset complete. Memory cleared.', 'stage');
});

function resetBars() {
    for (let i = 0; i <= 9; i++) {
        document.getElementById(`fill-${i}`).style.width = '0%';
        document.getElementById(`val-${i}`).innerText = '0%';
    }
}

function clearViz() {
    [vizRaw, vizCentered].forEach(v => {
        v.fillStyle = '#f9fafb';
        v.fillRect(0, 0, 64, 64);
    });
}

// Prediction Logic
predictBtn.addEventListener('click', async () => {
    const image = canvas.toDataURL('image/png');
    
    predictBtn.disabled = true;
    predictBtn.innerText = 'PROCESSING...';
    addLog('Capturing Frame Data...', 'stage');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image })
        });

        if (!response.ok) throw new Error('Neural Core Unavailable');

        const data = await response.json();
        updateUI(data);
    } catch (err) {
        addLog(`CRITICAL ERROR: ${err.message}`, 'error');
        predictionPlaceholder.classList.remove('hide');
        predictionResult.classList.add('hide');
        statusText.innerText = 'Offline';
        statusBadge.className = 'status-pill-minimal invalid';
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerText = 'Infer Digit';
    }
});

function updateUI(data) {
    const pred = data.prediction;
    const conf = (data.confidence * 100).toFixed(1);
    const status = data.status || 'valid';

    predictionPlaceholder.classList.add('hide');
    predictionResult.classList.remove('hide');

    predictionValue.innerText = pred;
    confidenceValue.innerText = `${conf}%`;

    // Update Status Pill
    statusText.innerText = status.charAt(0).toUpperCase() + status.slice(1);
    statusBadge.className = `status-pill-minimal ${status}`;

    // Logs
    if (data.logs) {
        data.logs.forEach(msg => addLog(msg));
    }

    // Probability Bars
    if (data.probabilities) {
        data.probabilities.forEach((p, i) => {
            const perc = (p * 100).toFixed(0);
            document.getElementById(`fill-${i}`).style.width = `${perc}%`;
            document.getElementById(`val-${i}`).innerText = `${perc}%`;
        });
    }

    // Visualization simulation
    drawSimulation();
}

function drawSimulation() {
    const smallCanvas = document.createElement('canvas');
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const sctx = smallCanvas.getContext('2d');
    sctx.drawImage(canvas, 0, 0, 28, 28);
    
    vizRaw.drawImage(canvas, 0, 0, 64, 64);
    vizCentered.drawImage(smallCanvas, 0, 0, 64, 64);
}

// Model Insights Integration
async function fetchModelStats() {
    addLog('Querying Model Performance Metrics...', 'stage');
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error('Stats Engine Offline');
        
        const data = await response.json();
        if (data.status === 'success') {
            renderMetrics(data);
            renderHeatmap('heatmap-counts', data.confusion_matrix, false);
            renderHeatmap('heatmap-percent', data.confusion_matrix_percent, true);
            addLog('Neural Core Diagnostics Synced Successfully.');
        }
    } catch (err) {
        addLog(`Metrics Retrieval Failed: ${err.message}`, 'error');
    }
}

function renderMetrics(data) {
    document.getElementById('metric-accuracy').innerText = `${(data.accuracy * 100).toFixed(2)}%`;
    document.getElementById('metric-precision').innerText = `${(data.precision * 100).toFixed(2)}%`;
    document.getElementById('metric-recall').innerText = `${(data.recall * 100).toFixed(2)}%`;
    document.getElementById('metric-f1').innerText = `${(data.f1_score * 100).toFixed(2)}%`;
}

function renderHeatmap(containerId, matrix, isPercentage) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    
    let maxVal = 0;
    if (isPercentage) {
        maxVal = 100;
    } else {
        matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));
    }

    // Add Corner Empty Cell
    const corner = document.createElement('div');
    corner.className = 'hm-cell corner';
    container.appendChild(corner);

    // Add Top Labels (Predicted)
    for (let j = 0; j < 10; j++) {
        const hLabel = document.createElement('div');
        hLabel.className = 'hm-cell label';
        hLabel.innerText = j;
        container.appendChild(hLabel);
    }

    matrix.forEach((row, i) => {
        // Add Left Label (Actual)
        const vLabel = document.createElement('div');
        vLabel.className = 'hm-cell label';
        vLabel.innerText = i;
        container.appendChild(vLabel);

        row.forEach((val, j) => {
            const cell = document.createElement('div');
            cell.className = 'hm-cell';
            const level = maxVal === 0 ? 0 : Math.ceil((val / maxVal) * 5);
            cell.classList.add(`lvl-${level}`);
            const displayVal = isPercentage ? val.toFixed(1) : val;
            cell.title = `Actual: ${i}, Predicted: ${j} | Value: ${val}${isPercentage ? '%' : ''}`;
            
            // Show exact value if requested (we'll show all values now for "exactness")
            cell.innerText = displayVal;
            container.appendChild(cell);
        });
    });
}

init();
