const canvas = document.getElementById('digitCanvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const predictionValue = document.getElementById('predictionValue');
const confidenceValue = document.getElementById('confidenceValue');
const statusBadge = document.getElementById('statusBadge');
const terminal = document.getElementById('terminal');
const waterBarsContainer = document.getElementById('waterBars');
const sampleGallery = document.getElementById('sampleGallery');

const API_URL = 'https://digit-classifier-backend-0qil.onrender.com/predict';

let isDrawing = false;

// Initial Setup
function init() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 18;
    ctx.strokeStyle = 'white';

    createBars();
    loadSamples();
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

// Draw MNIST-like digits for gallery using simple path data
function loadSamples() {
    sampleGallery.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const box = document.createElement('div');
        box.className = 'mnist-box';
        // Using a centered number as a sample placeholder for MNIST vibe
        box.style.display = 'flex';
        box.style.alignItems = 'center';
        box.style.justifyContent = 'center';
        box.style.fontSize = '12px';
        box.style.fontWeight = '800';
        box.innerText = i;
        sampleGallery.appendChild(box);
    }
}

function addLog(msg, type = 'log') {
    const p = document.createElement('p');
    p.className = `term-msg term-${type}`;
    p.innerText = `> ${msg}`;
    terminal.appendChild(p);
    terminal.scrollTop = terminal.scrollHeight;
}

// Canvas Interaction
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

window.addEventListener('mouseup', () => {
    isDrawing = false;
});

clearBtn.addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionValue.innerText = '-';
    confidenceValue.innerText = 'Ready for input';
    statusBadge.className = 'status-badge idle';
    statusBadge.innerText = 'Idle';
    terminal.innerHTML = '<p class="term-msg">> System ready. Waiting for draw...</p>';
    resetBars();
});

function resetBars() {
    for (let i = 0; i <= 9; i++) {
        document.getElementById(`fill-${i}`).style.height = '0%';
    }
}

// Predict Logic
predictBtn.addEventListener('click', async () => {
    addLog('Capturing canvas data...', 'stage');
    const image = canvas.toDataURL('image/png');
    
    addLog('Transmitting to Neural Core (Render API)...');
    predictBtn.disabled = true;
    predictBtn.innerText = 'Analyzing...';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image })
        });

        if (!response.ok) throw new Error('API unstable or unavailable');

        const data = await response.json();
        updateUI(data);
    } catch (err) {
        addLog(`Error: ${err.message}`, 'error');
        predictionValue.innerText = '!';
        confidenceValue.innerText = 'Connection Failed';
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerText = 'Predict Digit';
    }
});

function updateUI(data) {
    // Prediction & Confidence
    const pred = data.prediction;
    const conf = (data.confidence * 100).toFixed(1);

    predictionValue.innerText = isNaN(pred) ? '?' : pred;
    confidenceValue.innerText = `${conf}% Confidence`;

    // Status Badge
    statusBadge.innerText = pred === "Not a digit" ? "Invalid" : 
                            pred === "Uncertain" ? "Uncertain" : "Valid";
    statusBadge.className = `status-badge ${statusBadge.innerText.toLowerCase()}`;

    // Update Logs
    if (data.logs) {
        data.logs.forEach(msg => addLog(msg));
    }

    // Update Progress Bars
    if (data.probabilities) {
        data.probabilities.forEach((p, i) => {
            const fill = document.getElementById(`fill-${i}`);
            fill.style.height = `${p * 100}%`;
        });
    }

    addLog('Analysis session completeed.', 'stage');
}

init();
