const canvas = document.getElementById('digitCanvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const predictionValue = document.getElementById('predictionValue');
const confidenceValue = document.getElementById('confidenceValue');
const statusBadge = document.getElementById('statusBadge');
const terminal = document.getElementById('terminal');
const waterBarsContainer = document.getElementById('waterBars');

// Vision Viz Canvases
const vizRaw = document.getElementById('viz-raw').getContext('2d');
const vizCentered = document.getElementById('viz-centered').getContext('2d');
const vizFinal = document.getElementById('viz-final').getContext('2d');

let isDrawing = false;

// Initial Setup
function init() {
    setupCanvas();
    createBars();
    addLog('NEURAL_CORE_OS V2.0.4 Loaded.', 'stage');
    addLog('Waiting for input stream...');
    fetchModelStats();
}

function setupCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 18;
    ctx.strokeStyle = 'white';
}

function createBars() {
    waterBarsContainer.innerHTML = '';
    for (let i = 0; i <= 9; i++) {
        const wrapper = document.createElement('div');
        wrapper.className = 'bar-wrapper';
        wrapper.innerHTML = `
            <div class="bar-track">
                <div class="water-fill" id="fill-${i}" style="height: 0%"></div>
            </div>
            <div class="label">${i}</div>
        `;
        waterBarsContainer.appendChild(wrapper);
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
    predictionValue.innerText = '-';
    predictionValue.style.color = 'var(--text-main)';
    confidenceValue.innerText = '0.0%';
    statusBadge.innerText = 'IDLE';
    statusBadge.className = 'status-pill idle';
    resetBars();
    clearViz();
    addLog('Matrix reset complete. Memory cleared.', 'stage');
});

function resetBars() {
    for (let i = 0; i <= 9; i++) {
        document.getElementById(`fill-${i}`).style.height = '0%';
    }
}

function clearViz() {
    [vizRaw, vizCentered, vizFinal].forEach(v => {
        v.fillStyle = '#000';
        v.fillRect(0, 0, 60, 60);
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
        predictionValue.innerText = '!';
        predictionValue.style.color = '#ff3232';
        statusBadge.innerText = 'OFFLINE';
        statusBadge.className = 'status-pill invalid';
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerText = 'INFER DIGIT';
    }
});

function updateUI(data) {
    const pred = data.prediction;
    const conf = (data.confidence * 100).toFixed(1);
    const status = data.status || 'valid';

    // Display digit or placeholder symbols
    if (status === 'valid') {
        predictionValue.innerText = pred;
        predictionValue.style.color = '#00ff6a'; // Green
    } else if (status === 'uncertain') {
        predictionValue.innerText = '?';
        predictionValue.style.color = '#ff9d00'; // Yellow
    } else {
        predictionValue.innerText = '!';
        predictionValue.style.color = '#ff3232'; // Red
    }

    confidenceValue.innerText = `${conf}%`;

    // Update Status Pill
    statusBadge.innerText = status.toUpperCase();
    statusBadge.className = `status-pill ${status}`;

    // Logs
    if (data.logs) {
        data.logs.forEach(msg => addLog(msg));
    }

    // Probability Bars
    if (data.probabilities) {
        data.probabilities.forEach((p, i) => {
            document.getElementById(`fill-${i}`).style.height = `${p * 100}%`;
        });
    }

    // Visualization simulation (Since we don't send viz data from backend, we simulate)
    drawSimulation(pred);
}

function drawSimulation(pred) {
    // This is just for "Wow" factor in frontend to show the process steps
    const smallCanvas = document.createElement('canvas');
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const sctx = smallCanvas.getContext('2d');
    sctx.drawImage(canvas, 0, 0, 28, 28);
    
    // Draw on viz-raw
    vizRaw.drawImage(canvas, 0, 0, 60, 60);
    vizCentered.drawImage(smallCanvas, 0, 0, 60, 60);
    vizFinal.filter = 'contrast(200%) grayscale(100%)';
    vizFinal.drawImage(smallCanvas, 0, 0, 60, 60);
}

// New: Model Insights Integration
async function fetchModelStats() {
    addLog('Querying Model Performance Metrics...', 'stage');
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error('Stats Engine Offline');
        
        const data = await response.json();
        if (data.status === 'success') {
            document.getElementById('insightsSection').classList.remove('hide');
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
    document.getElementById('metric-accuracy').innerText = `${(data.accuracy * 100).toFixed(1)}%`;
    document.getElementById('metric-precision').innerText = `${(data.precision * 100).toFixed(1)}%`;
    document.getElementById('metric-recall').innerText = `${(data.recall * 100).toFixed(1)}%`;
    document.getElementById('metric-f1').innerText = `${(data.f1_score * 100).toFixed(1)}%`;
}

function renderHeatmap(containerId, matrix, isPercentage) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    let maxVal = 0;
    if (isPercentage) {
        maxVal = 100;
    } else {
        // For counts, use a capped max to allow smaller discrepancies to show color
        matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));
    }

    matrix.forEach((row, i) => {
        row.forEach((val, j) => {
            const cell = document.createElement('div');
            cell.className = 'hm-cell';
            
            // Normalize level (0-5)
            const level = maxVal === 0 ? 0 : Math.ceil((val / maxVal) * 5);
            cell.classList.add(`lvl-${level}`);
            
            // Format percentage or count for display
            const displayVal = isPercentage ? Math.round(val) : val;
            
            cell.title = `Actual: ${i}, Predicted: ${j} | Value: ${val}${isPercentage ? '%' : ''}`;
            cell.innerText = displayVal > 0 || i === j ? displayVal : '';
            
            container.appendChild(cell);
        });
    });
}

init();
