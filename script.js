let session = null;
let isRunning = false;

let classConfidenceThresholds = {
    'motorcycle': 0.05,
    'car': 0.30,
    'bus': 0.45,
    'truck': 0.25
};

let videoElement = document.getElementById('video-source');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let inferenceCanvas = document.createElement('canvas');
let inferenceCtx = inferenceCanvas.getContext('2d');

let classMap = { 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck' };

let countsLeft = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
let countsRight = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
let countsTotal = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };

// Dùng Map lưu lịch sử vị trí ngắn hạn để xét hướng qua vạch
let recentVehicles = new Map();
let uniqueIdCounter = 1;

let lineConfig = { positionRatio: 0.35 }; 
let isDraggingLine = false;
let chartInstance = null;

let lastTime = performance.now();
let frameCount = 0;
let currentFps = 0;
let isInferencing = false;
let enableCountingLine = true; 
let latestDetections = [];
let videoObjectUrl = null;

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
        slider.value = classConfidenceThresholds[vehicleKey];
        span.innerText = classConfidenceThresholds[vehicleKey].toFixed(2);
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
    drawScene(latestDetections);
}

canvas.addEventListener('pointerdown', (e) => {
    if (!enableCountingLine) return;
    const rect = canvas.getBoundingClientRect();
    const scaleY = canvas.height / rect.height;
    const mouseY = (e.clientY - rect.top) * scaleY;
    const lineY = canvas.height * lineConfig.positionRatio;
    if (Math.abs(mouseY - lineY) < 40) {
        isDraggingLine = true;
        canvas.setPointerCapture(e.pointerId);
    }
});

window.addEventListener('pointermove', (e) => {
    if (!isDraggingLine || !enableCountingLine) return;
    const rect = canvas.getBoundingClientRect();
    const scaleY = canvas.height / rect.height;
    const mouseY = (e.clientY - rect.top) * scaleY;
    lineConfig.positionRatio = Math.max(0.05, Math.min(0.95, mouseY / canvas.height));
    drawScene(latestDetections);
});

window.addEventListener('pointerup', () => { isDraggingLine = false; });
window.addEventListener('pointercancel', () => { isDraggingLine = false; });
function resetLinePosition() {
    lineConfig.positionRatio = 0.35;
    drawScene(latestDetections);
}

const uploadInput = document.getElementById('upload-video');
if (uploadInput) {
    uploadInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            if (isRunning) stopAI();
            resetSystemDataOnly();
            if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
            videoObjectUrl = URL.createObjectURL(file);
            videoElement.src = videoObjectUrl;
            videoElement.load();
            videoElement.onloadedmetadata = function() {
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                inferenceCanvas.width = canvas.width;
                inferenceCanvas.height = canvas.height;
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                drawScene([]);
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
                } catch (err) {
                    console.warn(`Không tải được model ${f}${m}:`, err.message);
                }
            }
            if (session) break;
        }

        if (!session) throw new Error("Model not found");
        setStatus('ready', 'AI READY');
        if (videoElement.src) document.getElementById('btn-start').disabled = false;
    } catch (e) {
        console.error('Không thể tải model:', e);
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

async function startAI() {
    if (!videoElement.src || !session) return;
    try {
        await videoElement.play();
    } catch (err) {
        console.error('Không thể phát video:', err);
        stopAI();
        return;
    }
    isRunning = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('btn-capture').disabled = false;
    setStatus('ready', 'RUNNING');
    requestAnimationFrame(processFrame);
}

function stopAI() {
    isRunning = false;
    videoElement.pause();
    document.getElementById('btn-start').disabled = !(videoElement.src && session);
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('btn-capture').disabled = true;
    setStatus('stopped', 'AI STOPPED');
}

function resetSystemDataOnly() {
    countsLeft = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    countsRight = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    countsTotal = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    recentVehicles.clear();
    uniqueIdCounter = 1;
    latestDetections = [];
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
        drawScene([]);
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
    drawScene(latestDetections);

    if (!isInferencing) {
        isInferencing = true;
        // Inference phải dùng frame gốc, không dùng canvas đã có box và vạch đỏ.
        inferenceCtx.drawImage(videoElement, 0, 0, inferenceCanvas.width, inferenceCanvas.height);
        setTimeout(async () => {
            try {
                const { tensor, ratio, dw, dh } = preprocessWithLetterbox(inferenceCanvas, 640);
                const results = await session.run({ [session.inputNames[0]]: tensor });
                const dets = parseYolov10Output(results[session.outputNames[0]], canvas.width, canvas.height, ratio, dw, dh);
                latestDetections = matchAndCountVehicles(dets);
                updateUIStats();
            } catch (err) {
                console.error('Lỗi xử lý frame:', err);
                setStatus('stopped', 'AI ERROR');
                stopAI();
            }
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
    if (!output || !output.data || !output.dims || output.dims.length !== 3) {
        throw new Error('Output model không đúng định dạng 3 chiều');
    }
    const data = output.data;
    const dims = output.dims;

    const parseBox = (x1, y1, x2, y2, conf, clsId) => {
        if (![x1, y1, x2, y2, conf, clsId].every(Number.isFinite)) return;
        if (Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1.5) {
            x1 *= 640; y1 *= 640; x2 *= 640; y2 *= 640;
        }
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
                parseBox(data[i], data[dims[2]+i], data[2*dims[2]+i], data[3*dims[2]+i], data[4*dims[2]+i], data[5*dims[2]+i]);
            }
        }
    }
    return dets;
}

function calculateIoU(firstBox, secondBox) {
    const [firstX, firstY, firstW, firstH] = firstBox;
    const [secondX, secondY, secondW, secondH] = secondBox;
    const intersectionX = Math.max(firstX, secondX);
    const intersectionY = Math.max(firstY, secondY);
    const intersectionRight = Math.min(firstX + firstW, secondX + secondW);
    const intersectionBottom = Math.min(firstY + firstH, secondY + secondH);
    const intersectionArea = Math.max(0, intersectionRight - intersectionX) * Math.max(0, intersectionBottom - intersectionY);
    const unionArea = firstW * firstH + secondW * secondH - intersectionArea;
    return unionArea > 0 ? intersectionArea / unionArea : 0;
}

function suppressOverlappingDetections(detections) {
    const filtered = [];
    const detectionsByClass = new Map();

    detections.forEach(detection => {
        if (!detectionsByClass.has(detection.className)) detectionsByClass.set(detection.className, []);
        detectionsByClass.get(detection.className).push(detection);
    });

    detectionsByClass.forEach((classDetections, className) => {
        classDetections.sort((first, second) => second.confidence - first.confidence);
        const overlapThreshold = className === 'motorcycle' ? 0.70 : 0.55;
        while (classDetections.length > 0) {
            const bestDetection = classDetections.shift();
            filtered.push(bestDetection);
            for (let index = classDetections.length - 1; index >= 0; index--) {
                if (calculateIoU(bestDetection.bbox, classDetections[index].bbox) >= overlapThreshold) {
                    classDetections.splice(index, 1);
                }
            }
        }
    });

    return filtered;
}

function matchAndCountVehicles(detections) {
    let activeVehicles = [];
    const directionMode = document.getElementById('counting-direction').value;
    const lineY = lineConfig.positionRatio * canvas.height;
    const nowTime = Date.now();

    // Giữ track đủ lâu khi inference chậm hoặc model bỏ sót vài frame.
    for (let [id, val] of recentVehicles.entries()) {
        if (nowTime - val.time > 8000) recentVehicles.delete(id);
    }

    const orderedDetections = [...detections].sort((first, second) => second.confidence - first.confidence);
    const candidateMatches = [];
    const maxMatchDistance = Math.max(220, Math.min(canvas.width, canvas.height) * 0.20);

    orderedDetections.forEach((det, detectionIndex) => {
        const [x, y, w, h] = det.bbox;
        const cx = x + w / 2, cy = y + h / 2;
        for (let [id, val] of recentVehicles.entries()) {
            if (val.className === det.className) {
                const elapsedSeconds = Math.min((nowTime - val.time) / 1000, 1);
                const predictedX = val.cx + (val.vx || 0) * elapsedSeconds;
                const predictedY = val.cy + (val.vy || 0) * elapsedSeconds;
                const distance = Math.hypot(cx - predictedX, cy - predictedY);
                const overlap = val.bbox ? calculateIoU(det.bbox, val.bbox) : 0;
                if (overlap >= 0.05 || distance <= maxMatchDistance) {
                    candidateMatches.push({ detectionIndex, id, score: overlap * 1000 - distance });
                }
            }
        }
    });

    candidateMatches.sort((first, second) => second.score - first.score);
    const assignedIds = new Map();
    const usedIds = new Set();
    const usedDetections = new Set();
    candidateMatches.forEach(match => {
        if (!usedIds.has(match.id) && !usedDetections.has(match.detectionIndex)) {
            assignedIds.set(match.detectionIndex, match.id);
            usedIds.add(match.id);
            usedDetections.add(match.detectionIndex);
        }
    });

    orderedDetections.forEach((det, detectionIndex) => {
        const [x, y, w, h] = det.bbox;
        const cx = x + w / 2, cy = y + h / 2;
        let assignedId = assignedIds.get(detectionIndex);

        if (!assignedId) {
            assignedId = uniqueIdCounter++;
        }

        let oldData = recentVehicles.get(assignedId);

        // Kiểm tra cắt vạch
        if (oldData && enableCountingLine && !oldData.counted) {
            let oldCy = oldData.cy;
            let crossed = false;

            if (directionMode === 'both') {
                if ((oldCy < lineY && cy >= lineY) || (oldCy > lineY && cy <= lineY)) crossed = true;
            } else if (directionMode === 'down') {
                if (oldCy < lineY && cy >= lineY) crossed = true;
            } else if (directionMode === 'up') {
                if (oldCy > lineY && cy <= lineY) crossed = true;
            }

            if (crossed) {
                oldData.counted = true;
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

        let isCounted = oldData ? oldData.counted : false;
        const elapsedSeconds = oldData ? Math.max((nowTime - oldData.time) / 1000, 0.001) : 0;
        const velocityX = oldData ? (cx - oldData.cx) / elapsedSeconds : 0;
        const velocityY = oldData ? (cy - oldData.cy) / elapsedSeconds : 0;
        recentVehicles.set(assignedId, {
            cx,
            cy,
            bbox: det.bbox,
            className: det.className,
            counted: isCounted,
            time: nowTime,
            vx: Math.max(-1000, Math.min(1000, velocityX)),
            vy: Math.max(-1000, Math.min(1000, velocityY))
        });
        activeVehicles.push({ id: assignedId, bbox: [x, y, w, h], className: det.className, confidence: det.confidence });
    });

    return activeVehicles;
}

function drawScene(vehicles) {
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

window.addEventListener('beforeunload', () => {
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
});