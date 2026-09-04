let session = null;
let isRunning = false;

let classConfidenceThresholds = {
    'motorcycle': 0.10,
    'car': 0.35,
    'bus': 0.40,
    'truck': 0.45
};

let videoElement = document.getElementById('video-source');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');

let classMap = { 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck' };

let countsLeft = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
let countsRight = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
let countsTotal = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };

let previousVehiclePositions = new Map(); 
let uniqueGlobalId = 1;
let countedGlobalIds = new Set();

let lineConfig = { positionRatio: 0.35 }; 
let isDraggingLine = false;
let chartInstance = null;

let lastTime = performance.now();
let frameCount = 0;
let currentFps = 0;
let isInferencing = false;
let enableCountingLine = true; 
let latestVehicles = [];

setInterval(() => {
    const now = new Date();
    const clockEl = document.getElementById('clock');
    if (clockEl) clockEl.innerText = now.toTimeString().split(' ')[0];
}, 1000);

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('btn-start').addEventListener('click', startAI);
    document.getElementById('btn-stop').addEventListener('click', stopAI);
    document.getElementById('btn-reset').addEventListener('click', resetSystem);
    document.getElementById('btn-capture').addEventListener('click', captureFrame);
    document.getElementById('btn-toggle-line').addEventListener('click', toggleCountingLineUI);
    document.getElementById('btn-reset-line').addEventListener('click', resetLinePosition);

    setupSlider('conf-moto-slider', 'motorcycle', 'conf-moto-val');
    setupSlider('conf-car-slider', 'car', 'conf-car-val');
    setupSlider('conf-bus-slider', 'bus', 'conf-bus-val');
    setupSlider('conf-truck-slider', 'truck', 'conf-truck-val');

    initChart();
    loadModel();
});

function setupSlider(sliderId, vehicleKey, valSpanId) {
    const slider = document.getElementById(sliderId);
    const span = document.getElementById(valSpanId);
    if (slider && span) {
        slider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            classConfidenceThresholds[vehicleKey] = val;
            span.innerText = val.toFixed(2);
        });
    }
}

function toggleCountingLineUI() {
    enableCountingLine = !enableCountingLine;
    const btn = document.getElementById('btn-toggle-line');
    if (enableCountingLine) {
        btn.className = "btn btn-success";
        btn.innerText = "Vạch: ON";
    } else {
        btn.className = "btn btn-danger";
        btn.innerText = "Vạch: OFF";
    }
}

canvas.addEventListener('mousedown', (e) => {
    if (!enableCountingLine) return;
    const rect = canvas.getBoundingClientRect();
    const scaleY = canvas.height / rect.height;
    const mouseY = (e.clientY - rect.top) * scaleY;
    const lineY = canvas.height * lineConfig.positionRatio;
    if (Math.abs(mouseY - lineY) < 40) isDraggingLine = true;
});

window.addEventListener('mousemove', (e) => {
    if (!isDraggingLine || !enableCountingLine) return;
    const rect = canvas.getBoundingClientRect();
    const scaleY = canvas.height / rect.height;
    const mouseY = (e.clientY - rect.top) * scaleY;
    lineConfig.positionRatio = Math.max(0.05, Math.min(0.95, mouseY / canvas.height));
});

window.addEventListener('mouseup', () => { isDraggingLine = false; });
function resetLinePosition() { lineConfig.positionRatio = 0.35; }

const uploadInput = document.getElementById('upload-video');
if (uploadInput) {
    uploadInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            resetSystemDataOnly();
            videoElement.src = URL.createObjectURL(file);
            videoElement.load();
            videoElement.onloadedmetadata = function() {
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                drawDetectionsAndLine([]);
                if (session) {
                    document.getElementById('btn-start').disabled = false;
                    setStatus('ready', 'AI READY');
                }
            };
        }
    });
}

function initChart() {
    const chartCanvas = document.getElementById('trafficChart');
    if (!chartCanvas) return;
    chartInstance = new Chart(chartCanvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Car', 'Motorcycle', 'Bus', 'Truck'],
            datasets: [
                { label: 'Bên Trái', data: [0, 0, 0, 0], backgroundColor: '#2563eb' },
                { label: 'Bên Phải', data: [0, 0, 0, 0], backgroundColor: '#16a34a' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#f8fafc', font: { size: 9 } } },
                x: { grid: { display: false }, ticks: { color: '#f8fafc', font: { size: 9 } } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 9 } } } }
        }
    });
}

async function loadModel() {
    try {
        setStatus('ready', 'LOADING...');
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        const folders = ['./', 'model/', './model/'];
        const modelNames = ['yolov10n.onnx', 'yolov10s.onnx'];

        for (let f of folders) {
            for (let m of modelNames) {
                try {
                    session = await ort.InferenceSession.create(f + m, { executionProviders: ['wasm'] });
                    if (session) break;
                } catch (err) {}
            }
            if (session) break;
        }

        if (!session) throw new Error("Model not found");
        setStatus('ready', 'AI READY');
        if (videoElement.src) document.getElementById('btn-start').disabled = false;
    } catch (e) {
        setStatus('stopped', 'AI ERROR');
    }
}

function setStatus(cls, text) {
    const badge = document.getElementById('system-status');
    if (badge) {
        badge.className = `status-pill ${cls}`;
        badge.innerText = text;
    }
}

function startAI() {
    if (!videoElement.src || !session) return;
    isRunning = true;
    videoElement.play();
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('btn-capture').disabled = false;
    setStatus('ready', 'RUNNING');
    requestAnimationFrame(processFrame);
}

function stopAI() {
    isRunning = false;
    videoElement.pause();
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('btn-capture').disabled = true;
    setStatus('stopped', 'AI STOPPED');
}

function resetSystemDataOnly() {
    countsLeft = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    countsRight = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    countsTotal = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    previousVehiclePositions.clear();
    countedGlobalIds.clear();
    uniqueGlobalId = 1;
    latestVehicles = [];
    updateUIStats();
}

function resetSystem() {
    stopAI();
    resetSystemDataOnly();
    resetLinePosition();
    if (videoElement && videoElement.src) {
        videoElement.currentTime = 0;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        drawDetectionsAndLine([]);
    }
}

function processFrame() {
    if (!isRunning) return;
    if (videoElement.paused || videoElement.ended) {
        stopAI();
        return;
    }

    const now = performance.now();
    frameCount++;
    if (now - lastTime >= 1000) {
        currentFps = (frameCount * 1000) / (now - lastTime);
        document.getElementById('fps-display').innerText = currentFps.toFixed(1);
        frameCount = 0;
        lastTime = now;
    }

    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    drawDetectionsAndLine(latestVehicles);

    if (!isInferencing) {
        isInferencing = true;
        setTimeout(async () => {
            try {
                const { tensor, ratio, dw, dh } = preprocessWithLetterbox(canvas, 640);
                const results = await session.run({ [session.inputNames[0]]: tensor });
                const dets = parseYolov10Output(results[session.outputNames[0]], canvas.width, canvas.height, ratio, dw, dh);
                latestVehicles = processCountingAndTracking(dets);
                updateUIStats();
            } catch (err) {} 
            finally { isInferencing = false; }
        }, 0);
    }
    requestAnimationFrame(processFrame);
}

function preprocessWithLetterbox(srcCanvas, targetSize = 640) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetSize; tempCanvas.height = targetSize;
    const tCtx = tempCanvas.getContext('2d');
    const sw = srcCanvas.width, sh = srcCanvas.height;
    const ratio = Math.min(targetSize / sw, targetSize / sh);
    const nw = sw * ratio, nh = sh * ratio;
    const dw = (targetSize - nw) / 2, dh = (targetSize - nh) / 2;

    tCtx.fillStyle = '#111827';
    tCtx.fillRect(0, 0, targetSize, targetSize);
    tCtx.drawImage(srcCanvas, dw, dh, nw, nh);

    const imgData = tCtx.getImageData(0, 0, targetSize, targetSize);
    const data = imgData.data;
    const float32Data = new Float32Array(3 * targetSize * targetSize);
    for (let i = 0; i < targetSize * targetSize; i++) {
        float32Data[i] = data[i * 4] / 255.0;
        float32Data[targetSize * targetSize + i] = data[i * 4 + 1] / 255.0;
        float32Data[2 * targetSize * targetSize + i] = data[i * 4 + 2] / 255.0;
    }
    return { tensor: new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]), ratio, dw, dh };
}

function parseYolov10Output(output, origW, origH, ratio, dw, dh) {
    const dets = [];
    const data = output.data;
    const dims = output.dims;

    const parseBox = (x1, y1, x2, y2, conf, clsId) => {
        let rx1 = (x1 - dw) / ratio, ry1 = (y1 - dh) / ratio;
        let rx2 = (x2 - dw) / ratio, ry2 = (y2 - dh) / ratio;
        rx1 = Math.max(0, Math.min(origW, rx1));
        ry1 = Math.max(0, Math.min(origH, ry1));
        rx2 = Math.max(0, Math.min(origW, rx2));
        ry2 = Math.max(0, Math.min(origH, ry2));

        const w = rx2 - rx1, h = ry2 - ry1;
        if (w < 2 || h < 2) return;

        const className = classMap[clsId];
        if (className) {
            const threshold = classConfidenceThresholds[className] || 0.25;
            if (conf >= threshold) {
                dets.push({ bbox: [rx1, ry1, w, h], className, confidence: conf });
            }
        }
    };

    if (dims && dims.length === 3) {
        if (dims[2] === 6) {
            for (let i = 0; i < dims[1]; i++) {
                let off = i * 6;
                parseBox(data[off], data[off+1], data[off+2], data[off+3], data[off+4], Math.round(data[off+5]));
            }
        } else if (dims[1] === 6) {
            for (let i = 0; i < dims[2]; i++) {
                parseBox(data[i], data[dims[2]+i], data[2*dims[2]+i], data[3*dims[2]+i], data[4*dims[2]+i], Math.round(data[5*dims[2]+i]));
            }
        }
    }
    return dets;
}

function processCountingAndTracking(detections) {
    let currentFrameVehicles = [];
    const directionMode = document.getElementById('counting-direction').value;
    const lineCoord = lineConfig.positionRatio * canvas.height;

    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const cx = x + w / 2, cy = y + h / 2;

        let matchedId = null;
        let minDist = 120;
        for (let [id, pos] of previousVehiclePositions.entries()) {
            if (pos.className === det.className) {
                let dist = Math.hypot(cx - pos.cx, cy - pos.cy);
                if (dist < minDist) { minDist = dist; matchedId = id; }
            }
        }

        if (!matchedId) {
            matchedId = uniqueGlobalId++;
        }

        let oldPos = previousVehiclePositions.get(matchedId);
        if (oldPos && enableCountingLine && !countedGlobalIds.has(matchedId)) {
            let crossed = false;
            if (directionMode === 'both') {
                if ((oldPos.cy < lineCoord && cy >= lineCoord) || (oldPos.cy > lineCoord && cy <= lineCoord)) crossed = true;
            } else if (directionMode === 'down') {
                if (oldPos.cy < lineCoord && cy >= lineCoord) crossed = true;
            } else if (directionMode === 'up') {
                if (oldPos.cy > lineCoord && cy <= lineCoord) crossed = true;
            }

            if (crossed) {
                countedGlobalIds.add(matchedId);
                const isLeftSide = cx < (canvas.width / 2);
                if (isLeftSide) {
                    countsLeft[det.className]++;
                    countsLeft.total++;
                } else {
                    countsRight[det.className]++;
                    countsRight.total++;
                }
                countsTotal[det.className]++;
                countsTotal.total++;
            }
        }

        previousVehiclePositions.set(matchedId, { cx, cy, className: det.className });
        currentFrameVehicles.push({ id: matchedId, bbox: [x, y, w, h], className: det.className, confidence: det.confidence });
    });
    return currentFrameVehicles;
}

function drawDetectionsAndLine(vehicles) {
    if (enableCountingLine) {
        const lineY = lineConfig.positionRatio * canvas.height;
        ctx.strokeStyle = isDraggingLine ? '#38bdf8' : '#ef4444';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(0, lineY);
        ctx.lineTo(canvas.width, lineY);
        ctx.stroke();

        ctx.fillStyle = isDraggingLine ? '#38bdf8' : '#ef4444';
        ctx.font = 'bold 13px Segoe UI';
        ctx.fillText("VẠCH ĐẾM PHƯƠNG TIỆN", 15, lineY - 8);
    }

    if (vehicles) {
        vehicles.forEach(v => {
            const [x, y, w, h] = v.bbox;
            const color = getCategoryColor(v.className);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = color;
            ctx.fillRect(x, y > 18 ? y - 18 : 0, 110, 16);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 10px Segoe UI';
            ctx.fillText(`${v.className.toUpperCase()} #${v.id} (${(v.confidence*100).toFixed(0)}%)`, x + 2, y > 18 ? y - 5 : 12);
        });
    }
}

function getCategoryColor(cls) {
    switch (cls) {
        case 'car': return '#2563eb';
        case 'motorcycle': return '#16a34a';
        case 'bus': return '#d97706';
        case 'truck': return '#dc2626';
        default: return '#38bdf8';
    }
}

function updateUIStats() {
    const setText = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
    setText('count-car-left', countsLeft.car); setText('count-car-right', countsRight.car); setText('count-car', countsTotal.car);
    setText('count-moto-left', countsLeft.motorcycle); setText('count-moto-right', countsRight.motorcycle); setText('count-moto', countsTotal.motorcycle);
    setText('count-bus-left', countsLeft.bus); setText('count-bus-right', countsRight.bus); setText('count-bus', countsTotal.bus);
    setText('count-truck-left', countsLeft.truck); setText('count-truck-right', countsRight.truck); setText('count-truck', countsTotal.truck);
    setText('count-left-total', countsLeft.total); setText('count-right-total', countsRight.total); setText('count-total', countsTotal.total);

    let density = 'LOW', dClass = 'density-low';
    if (countsTotal.total >= 40) { density = 'HIGH'; dClass = 'density-high'; }
    else if (countsTotal.total >= 15) { density = 'MEDIUM'; dClass = 'density-med'; }

    const badge = document.getElementById('density-status');
    if (badge) { badge.className = `density-badge ${dClass}`; badge.innerText = density; }

    const banner = document.getElementById('congestion-banner');
    if (banner) {
        if (density === 'HIGH') {
            banner.style.background = '#dc2626';
            banner.innerText = '⚠️ TRAFFIC CONGESTION WARNING';
        } else {
            banner.style.background = '#16a34a';
            banner.innerText = '✓ TRAFFIC NORMAL';
        }
    }

    if (chartInstance) {
        chartInstance.data.datasets[0].data = [countsLeft.car, countsLeft.motorcycle, countsLeft.bus, countsLeft.truck];
        chartInstance.data.datasets[1].data = [countsRight.car, countsRight.motorcycle, countsRight.bus, countsRight.truck];
        chartInstance.update();
    }
}

function captureFrame() {
    const link = document.createElement('a');
    link.download = `capture-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
}