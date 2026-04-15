const canvas = document.getElementById('digitCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictionDisplay = document.getElementById('predictionDisplay');
const confidenceBars = document.getElementById('confidenceBars');

// API ENDPOINT config
const API_URL = 'http://127.0.0.1:5000/predict';

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let drawTimer = null;

// Initialize Canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.lineWidth = 15; // Thick stroke for MNIST style digits
ctx.strokeStyle = 'white';

// Empty bar setup
function resetBars() {
    confidenceBars.innerHTML = '';
    for (let i=0; i<3; i++) {
        confidenceBars.innerHTML += `
            <div class="bar-row">
                <div class="bar-label">-</div>
                <div class="bar-track"><div class="bar-fill" style="width: 0%"></div></div>
                <div class="bar-val">0%</div>
            </div>`;
    }
}
resetBars();

// Interaction Events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', endDrawing);
canvas.addEventListener('mouseout', endDrawing);

// Touch support
canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
canvas.addEventListener('touchend', endDrawing);

clearBtn.addEventListener('click', clearCanvas);

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    isDrawing = true;
    const pos = getPos(e);
    [lastX, lastY] = [pos.x, pos.y];
}

function draw(e) {
    if (!isDrawing) return;
    const pos = getPos(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    
    [lastX, lastY] = [pos.x, pos.y];
    
    // Auto translate drawing into prediction after a delay
    clearTimeout(drawTimer);
    drawTimer = setTimeout(triggerPrediction, 400); 
}

function endDrawing() {
    isDrawing = false;
}

function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    startDrawing(touch);
}

function handleTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    draw(touch);
}

function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Reset Display
    predictionDisplay.className = 'prediction-placeholder';
    predictionDisplay.innerHTML = '<span>Draw a digit to see the prediction</span>';
    resetBars();
}

async function triggerPrediction() {
    const dataURL = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });
        
        if (!response.ok) throw new Error('API Error');
        const result = await response.json();
        
        displayResult(result);
    } catch (err) {
        console.error("Prediction failed:", err);
    }
}

function displayResult(result) {
    if(result.digit === undefined) return;
    
    // Update main prediction box
    predictionDisplay.className = 'prediction-result';
    predictionDisplay.innerHTML = `
        <div class="pred-digit">${result.digit}</div>
        <div class="pred-conf">${(result.confidence * 100).toFixed(1)}% Confidence</div>
    `;
    
    // Update top 3 bars
    confidenceBars.innerHTML = '';
    result.top3.forEach(item => {
        const [digit, prob] = item;
        const color = prob > 0.8 ? 'var(--success)' : 'var(--accent)';
        
        confidenceBars.innerHTML += `
            <div class="bar-row">
                <div class="bar-label">${digit}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: ${prob * 100}%; background: ${color}"></div>
                </div>
                <div class="bar-val">${(prob * 100).toFixed(0)}%</div>
            </div>
        `;
    });
}
