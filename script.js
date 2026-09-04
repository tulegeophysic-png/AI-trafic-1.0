let session = null;
let isRunning = false;

// Ngưỡng Confidence riêng biệt cho từng loại xe (đã tối ưu hoá)
let classConfidenceThresholds = {
    'motorcycle': 0.10, // Xe máy nhỏ, giữ ngưỡng thấp để bắt xe xa
    'car': 0.35,        // Ô tô con
    'bus': 0.40,        // Xe khách lớn
    'truck': 0.45       // Xe tải, ngưỡng cao để lọc triệt để lỗi nhận diện nhầm
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

let lowDensityThreshold = 5;
let highDensityThreshold = 15;

let lineConfig = { positionRatio: 0.35 }; 
let isDraggingLine = false;
let chartInstance = null;

let lastTime = performance.now();
let frameCount = 0;
let currentFps = 0;
let isInferencing = false;
let enableCountingLine = true; 
let latestVehicles = [];

// Đồng hồ hệ thống thời gian thực
setInterval(() => {
    const now = new Date();
    const clockEl = document.getElementById('clock');
    if (clockEl) clockEl.innerText = now.toTimeString().split(' ')[0];
}, 1000);

// --- GẮN KẾT SỰ KIỆN GIAO DIỆN ---
document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btn-start');
    if (btnStart) btnStart.addEventListener('click', startAI);

    const btnStop = document.getElementById('btn-stop');
    if (btnStop) btnStop.addEventListener('click', stopAI);

    const btnReset = document.getElementById('btn-reset');
    if (btnReset) btnReset.addEventListener('click', resetSystem);

    const btnCapture = document.getElementById('btn-capture');
    if (btnCapture) btnCapture.addEventListener('click', captureFrame);

    const btnToggleLine = document.getElementById('btn-toggle-line');
    if (btnToggleLine) btnToggleLine.addEventListener('click', toggleCountingLineUI);

    const btnResetLine = document.getElementById('btn-reset-line');
    if (btnResetLine) btnResetLine.addEventListener('click', resetLinePosition);

    // Lắng nghe sự kiện thay đổi của 4 thanh trượt confidence riêng biệt
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
    if (btn) {
        if (enableCountingLine) {
            btn.className = "btn btn-success";
            btn.innerText = "Vạch: ON";
        } else {
            btn.className = "btn btn-danger";
            btn.innerText = "Vạch: OFF";
        }
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

window.addEventListener('mouseup', () => {
    isDraggingLine = false;
});

function resetLinePosition() {
    lineConfig.positionRatio = 0.35;
}

const uploadInput = document.getElementById('upload-video');
if (uploadInput) {
    uploadInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            resetSystemDataOnly();
            const fileNameEl = document.getElementById('file-name');
            if (fileNameEl) fileNameEl.innerText = file.name;
            videoElement.src = URL.createObjectURL(file);
            videoElement.load();
            videoElement.onloadedmetadata = function() {
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                drawDetectionsAndLine([]);
                if (session) {
                    const startBtn = document.getElementById('btn-start');
                    if (startBtn) startBtn.disabled = false;
                    setStatus('active', 'AI READY');
                }
            };
        }
    });
}

function initChart() {
    const chartCanvas = document.getElementById('trafficChart');
    if (!chartCanvas) return;
    const ctxChart = chartCanvas.getContext('2d');
    chartInstance = new Chart(ctxChart, {
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
                y: { beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#f8fafc', font: { size: 10 } } },
                x: { grid: { display: false }, ticks: { color: '#f8fafc', font: { size: 10 } } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });
}

async function loadModel() {
    try {
        setStatus('waiting', 'LOADING MODEL...');
        const modelFileNames = ['yolov10n.onnx', 'yolov10s.onnx'];
        const folders = ['./', 'model/', './model/'];

        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        for (let folder of folders) {
            for (let name of modelFileNames) {
                try {
                    let path = folder + name;
                    session = await ort.InferenceSession.create(path, { executionProviders: ['wasm'] });
                    if (session) break;
                } catch (innerErr) {}
            }
            if (session) break;
        }

        if (!session) throw new Error("Không tìm thấy model.");
        setStatus('active', 'AI READY');
        if (videoElement.src) {
            const startBtn = document.getElementById('btn-start');
            if (startBtn) startBtn.disabled = false;
        }
    } catch (e) {
        setStatus('error', 'AI ERROR');
        console.error("Lỗi tải model:", e);
    }
}

function setStatus(statusClass, text) {
    const badge = document.getElementById('system-status');
    if (badge) {
        badge.className = `status-badge ${statusClass}`;
        badge.innerText = text;
    }
}

function startAI() {
    if (!videoElement.src || !session) return;
    isRunning = true;
    videoElement.play();
    
    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');
    const capBtn = document.getElementById('btn-capture');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    if (capBtn) capBtn.disabled = false;

    setStatus('active', 'AI RUNNING');
    requestAnimationFrame(processFrame);
}

function stopAI() {
    isRunning = false;
    videoElement.pause();

    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');
    const capBtn = document.getElementById('btn-capture');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (capBtn) capBtn.disabled = true;

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
        if (videoElement.readyState >= 2) {
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            drawDetectionsAndLine([]);
        }
    }
    setCongestion(false);
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
        const fpsEl = document.getElementById('fps-display');
        if (fpsEl) fpsEl.innerText = currentFps.toFixed(1);
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
                const inputName = session.inputNames[0];
                const results = await session.run({ [inputName]: tensor });
                const output = results[session.outputNames[0]];

                const detections = parseYolov10Output(output, canvas.width, canvas.height, ratio, dw, dh);
                latestVehicles = processCountingAndTracking(detections);
                updateUIStats();
            } catch (err) {
                console.error("Inference error:", err);
            } finally {
                isInferencing = false;
            }
        }, 0);
    }

    requestAnimationFrame(processFrame);
}

function preprocessWithLetterbox(sourceCanvas, targetSize = 640) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetSize;
    tempCanvas.height = targetSize;
    const tempCtx = tempCanvas.getContext('2d');

    const sw = sourceCanvas.width;
    const sh = sourceCanvas.height;
    const ratio = Math.min(targetSize / sw, targetSize / sh);
    const nw = sw * ratio;
    const nh = sh * ratio;
    const dw = (targetSize - nw) / 2;
    const dh = (targetSize - nh) / 2;

    tempCtx.fillStyle = '#111827';
    tempCtx.fillRect(0, 0, targetSize, targetSize);
    tempCtx.drawImage(sourceCanvas, dw, dh, nw, nh);

    const imgData = tempCtx.getImageData(0, 0, targetSize, targetSize);
    const { data } = imgData;
    const float32Data = new Float32Array(3 * targetSize * targetSize);

    for (let i = 0; i < targetSize * targetSize; i++) {
        float32Data[i] = data[i * 4] / 255.0;                     
        float32Data[targetSize * targetSize + i] = data[i * 4 + 1] / 255.0;      
        float32Data[2 * targetSize * targetSize + i] = data[i * 4 + 2] / 255.0;  
    }
    return {
        tensor: new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]),
        ratio, dw, dh
    };
}

function parseYolov10Output(output, origWidth, origHeight, ratio, dw, dh) {
    const dets = [];
    const data = output.data;
    const dims = output.dims;

    const parseBox = (x1, y1, x2, y2, conf, clsId) => {
        let rx1 = (x1 - dw) / ratio;
        let ry1 = (y1 - dh) / ratio;
        let rx2 = (x2 - dw) / ratio;
        let ry2 = (y2 - dh) / ratio;

        rx1 = Math.max(0, Math.min(origWidth, rx1));
        ry1 = Math.max(0, Math.min(origHeight, ry1));
        rx2 = Math.max(0, Math.min(origWidth, rx2));
        ry2 = Math.max(0, Math.min(origHeight, ry2));

        const boxW = rx2 - rx1;
        const boxH = ry2 - ry1;

        if (boxW < 2 || boxH < 2) return; 

        const className = classMap[clsId];
        if (className) {
            // Lấy ngưỡng confidence riêng biệt tương ứng với loại phương tiện
            const specificThreshold = classConfidenceThresholds[className] || 0.25;

            if (conf >= specificThreshold) {
                dets.push({
                    bbox: [rx1, ry1, boxW, boxH],
                    className: className,
                    confidence: conf
                });
            }
        }
    };

    if (dims && dims.length === 3) {
        if (dims[2] === 6) {
            let numRows = dims[1];
            for (let i = 0; i < numRows; i++) {
                let offset = i * 6;
                parseBox(data[offset], data[offset + 1], data[offset + 2], data[offset + 3], data[offset + 4], Math.round(data[offset + 5]));
            }
        } else if (dims[1] === 6) {
            let numRows = dims[2];
            for (let i = 0; i < numRows; i++) {
                parseBox(data[0 * numRows + i], data[1 * numRows + i], data[2 * numRows + i], data[3 * numRows + i], data[4 * numRows + i], Math.round(data[5 * numRows + i]));
            }
        }
    }
    return dets;
}

function processCountingAndTracking(detections) {
    let currentFrameVehicles = [];

    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const cx = x + w / 2;
        const cy = y + h / 2;

        let matchedId = null;
        let minDistance = 150; 

        for (let [id, pos] of previousVehiclePositions.entries()) {
            if (pos.className === det.className) {
                let dist = Math.hypot(cx - pos.cx, cy - pos.cy);
                if (dist < minDistance) {
                    minDistance = dist;
                    matchedId = id;
                }
            }
        }

        if (!matchedId) {
            matchedId = uniqueGlobalId++;
            
            if (!countedGlobalIds.has(matchedId)) {
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

        currentFrameVehicles.push({
            id: matchedId,
            bbox: [x, y, w, h],
            className: det.className,
            confidence: det.confidence
        });
    });

    return currentFrameVehicles;
}

function drawDetectionsAndLine(vehicles) {
    if (enableCountingLine) {
        const lineCoord = lineConfig.positionRatio * canvas.height;

        ctx.strokeStyle = isDraggingLine ? '#38bdf8' : '#ef4444';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(0, lineCoord);
        ctx.lineTo(canvas.width, lineCoord);
        ctx.stroke();

        ctx.fillStyle = isDraggingLine ? '#38bdf8' : '#ef4444';
        ctx.font = 'bold 15px Segoe UI';
        ctx.fillText(`VẠCH GIÁM SÁT (ĐÃ KÍCH HOẠT TỰ ĐỘNG ĐẾM THEO ID)`, 30, lineCoord - 12);
    }

    if (vehicles && vehicles.length > 0) {
        vehicles.forEach(veh => {
            const [x, y, w, h] = veh.bbox;
            const color = getCategoryColor(veh.className);

            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = color;
            ctx.fillRect(x, y > 20 ? y - 20 : 0, 130, 20);

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 11px Segoe UI';
            ctx.fillText(`${veh.className.toUpperCase()} #${veh.id} (${(veh.confidence*100).toFixed(0)}%)`, x + 4, y > 20 ? y - 6: 14);
        });
    }
}

function getCategoryColor(className) {
    switch (className) {
        case 'car': return '#2563eb';
        case 'motorcycle': return '#16a34a';
        case 'bus': return '#d97706';
        case 'truck': return '#dc2626';
        default: return '#38bdf8';
    }
}

function updateUIStats() {
    const setSafeText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.innerText = val;
    };

    setSafeText('count-car-left', countsLeft.car);
    setSafeText('count-car-right', countsRight.car);
    setSafeText('count-car', countsTotal.car);

    setSafeText('count-moto-left', countsLeft.motorcycle);
    setSafeText('count-moto-right', countsRight.motorcycle);
    setSafeText('count-moto', countsTotal.motorcycle);

    setSafeText('count-bus-left', countsLeft.bus);
    setSafeText('count-bus-right', countsRight.bus);
    setSafeText('count-bus', countsTotal.bus);

    setSafeText('count-truck-left', countsLeft.truck);
    setSafeText('count-truck-right', countsRight.truck);
    setSafeText('count-truck', countsTotal.truck);

    setSafeText('count-left-total', countsLeft.total);
    setSafeText('count-right-total', countsRight.total);
    setSafeText('count-total', countsTotal.total);

    let density = 'LOW';
    let densityClass = 'low';

    if (countsTotal.total >= highDensityThreshold) {
        density = 'HIGH';
        densityClass = 'high';
        setCongestion(true);
    } else if (countsTotal.total >= lowDensityThreshold) {
        density = 'MEDIUM';
        densityClass = 'medium';
        setCongestion(false);
    } else {
        setCongestion(false);
    }

    const densityBadge = document.getElementById('density-status');
    if (densityBadge) {
        densityBadge.className = `density-badge ${densityClass}`;
        densityBadge.innerText = density;
    }

    if (chartInstance) {
        chartInstance.data.datasets[0].data = [countsLeft.car, countsLeft.motorcycle, countsLeft.bus, countsLeft.truck];
        chartInstance.data.datasets[1].data = [countsRight.car, countsRight.motorcycle, countsRight.bus, countsRight.truck];
        chartInstance.update();
    }
}

function setCongestion(isCongested) {
    const banner = document.getElementById('congestion-banner');
    if (!banner) return;
    if (isCongested) {
        banner.className = 'congestion-banner warning';
        banner.innerText = '⚠️ TRAFFIC CONGESTION WARNING';
    } else {
        banner.className = 'congestion-banner normal';
        banner.innerText = '✓ TRAFFIC NORMAL';
    }
}

function captureFrame() {
    const link = document.createElement('a');
    link.download = `ai-traffic-capture-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
}