let session = null;
let isRunning = false;
let confidenceThreshold = 0.50;
let videoElement = document.getElementById('video-source');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');

let classNames = ['car', 'motorcycle', 'bus', 'truck'];
let classMap = { 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck' }; // COCO class indices for vehicle types

let counts = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
let countedIds = new Set();
let tracks = [];
let nextTrackId = 1;

let lowDensityThreshold = 5;
let highDensityThreshold = 15;

let chartInstance = null;
let lastTime = performance.now();
let frameCount = 0;
let currentFps = 0;

// Clock update
setInterval(() => {
    const now = new Date();
    document.getElementById('clock').innerText = now.toTimeString().split(' ')[0];
}, 1000);

// File input handler
document.getElementById('upload-video').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('file-name').innerText = file.name;
        videoElement.src = URL.createObjectURL(file);
        videoElement.load();
        videoElement.onloadedmetadata = function() {
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        };
    }
});

// Initialize Chart.js
function initChart() {
    const ctxChart = document.getElementById('trafficChart').getContext('2d');
    chartInstance = new Chart(ctxChart, {
        type: 'bar',
        data: {
            labels: ['Car', 'Motorcycle', 'Bus', 'Truck'],
            datasets: [{
                label: 'Số lượng phương tiện',
                data: [0, 0, 0, 0],
                backgroundColor: ['#2563eb', '#16a34a', '#d97706', '#dc2626'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, grid: { color: '#334155' }, ticks: { color: '#f8fafc' } },
                x: { grid: { display: false }, ticks: { color: '#f8fafc' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}
initChart();

function updateConfidence(val) {
    confidenceThreshold = parseFloat(val);
    document.getElementById('conf-val').innerText = val;
}

async function loadModel() {
    try {
        setStatus('waiting', 'LOADING MODEL...');
        
        // ⚠️ QUAN TRỌNG: Thay đổi tên file 'yolov10s.onnx' bên dưới 
        // thành tên chính xác file YOLOv10 mà bạn đang đặt trong thư mục model/ trên GitHub
        const modelFileName = 'yolov10s.onnx'; 

        // Danh sách các đường dẫn dự phòng để tránh lỗi 404 trên GitHub Pages
        const modelPaths = [
            `./model/${modelFileName}`,
            `model/${modelFileName}`,
            `./${modelFileName}`
        ];

        for (let path of modelPaths) {
            try {
                console.log("Đang thử tải model từ đường dẫn:", path);
                session = await ort.InferenceSession.create(path, {
                    executionProviders: ['webgpu', 'wasm']
                });
                if (session) {
                    console.log("Tải model thành công từ:", path);
                    break;
                }
            } catch (innerErr) {
                console.warn("Không thể tải từ đường dẫn:", path, innerErr);
            }
        }

        if (!session) {
            throw new Error("Không tìm thấy file model ở tất cả các đường dẫn thử nghiệm.");
        }

        setStatus('active', 'AI READY');
    } catch (e) {
        console.error("Model load failed chi tiết:", e);
        setStatus('error', 'AI ERROR – MODEL LOAD FAILED');
        alert("Không thể tải file model YOLOv10. Vui lòng kiểm tra lại tên file trong thư mục model/ và kết nối mạng!");
    }
}
loadModel();

function setStatus(statusClass, text) {
    const badge = document.getElementById('system-status');
    badge.className = `status-badge ${statusClass}`;
    badge.innerText = text;
}

async function startAI() {
    if (!videoElement.src) {
        alert("Vui lòng chọn video giao thông trước!");
        return;
    }
    if (!session) {
        alert("Model YOLOv10 chưa được tải xong hoặc lỗi đường dẫn file!");
        return;
    }
    isRunning = true;
    videoElement.play();
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('btn-capture').disabled = false;
    setStatus('active', 'AI RUNNING');
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

function resetSystem() {
    stopAI();
    counts = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    countedIds.clear();
    tracks = [];
    nextTrackId = 1;
    updateUIStats();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('file-name').innerText = "Chưa chọn file nào";
    videoElement.src = "";
}

async function processFrame() {
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

    try {
        const inputTensor = preprocessToBlob(canvas);
        const feeds = { images: inputTensor };
        const results = await session.run(feeds);
        const output = results[Object.keys(results)[0]];

        const detections = parseYolov10Output(output, canvas.width, canvas.height);
        updateTrackingAndCounting(detections, canvas.height);
        drawDetections(detections);
        updateUIStats();
    } catch (err) {
        console.error("Inference error:", err);
    }

    requestAnimationFrame(processFrame);
}

function preprocessToBlob(sourceCanvas) {
    const targetSize = 640;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetSize;
    tempCanvas.height = targetSize;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(sourceCanvas, 0, 0, targetSize, targetSize);
    
    const imgData = tempCtx.getImageData(0, 0, targetSize, targetSize);
    const { data } = imgData;
    const float32Data = new Float32Array(3 * targetSize * targetSize);

    for (let i = 0; i < targetSize * targetSize; i++) {
        float32Data[i] = data[i * 4] / 255.0;                      
        float32Data[targetSize * targetSize + i] = data[i * 4 + 1] / 255.0;      
        float32Data[2 * targetSize * targetSize + i] = data[i * 4 + 2] / 255.0;  
    }
    return new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]);
}

function parseYolov10Output(output, origWidth, origHeight) {
    const dets = [];
    const data = output.data;
    const numRows = output.dims[1]; 
    const numCols = output.dims[2]; 

    for (let i = 0; i < numRows; i++) {
        const rowOffset = i * numCols;
        const x1 = (data[rowOffset] / 640) * origWidth;
        const y1 = (data[rowOffset + 1] / 640) * origHeight;
        const x2 = (data[rowOffset + 2] / 640) * origWidth;
        const y2 = (data[rowOffset + 3] / 640) * origHeight;
        const confidence = data[rowOffset + 4];
        const classId = Math.round(data[rowOffset + 5]);

        if (confidence >= confidenceThreshold && classMap[classId]) {
            dets.push({
                bbox: [x1, y1, x2 - x1, y2 - y1],
                className: classMap[classId],
                confidence: confidence
            });
        }
    }
    return dets;
}

function updateTrackingAndCounting(detections, frameHeight) {
    const countingLineY = frameHeight * 0.7;
    let currentTracks = [];

    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const cx = x + w / 2;
        const cy = y + h / 2;

        let matchedTrack = null;
        let minDst = 50;

        tracks.forEach(track => {
            const tcx = track.bbox[0] + track.bbox[2] / 2;
            const tcy = track.bbox[1] + track.bbox[3] / 2;
            const dst = Math.hypot(cx - tcx, cy - tcy);
            if (dst < minDst && track.className === det.className) {
                minDst = dst;
                matchedTrack = track;
            }
        });

        if (matchedTrack) {
            const prevY = matchedTrack.bbox[1] + matchedTrack.bbox[3] / 2;
            matchedTrack.bbox = [x, y, w, h];
            matchedTrack.confidence = det.confidence;

            if (prevY < countingLineY && cy >= countingLineY && !countedIds.has(matchedTrack.id)) {
                countedIds.add(matchedTrack.id);
                counts[det.className]++;
                counts.total++;
            }
            currentTracks.push(matchedTrack);
        } else {
            currentTracks.push({
                id: nextTrackId++,
                bbox: [x, y, w, h],
                className: det.className,
                confidence: det.confidence
            });
        }
    });

    tracks = currentTracks;
}

function drawDetections(detections) {
    const countingLineY = canvas.height * 0.7;

    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, countingLineY);
    ctx.lineTo(canvas.width, countingLineY);
    ctx.stroke();

    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 16px Segoe UI';
    ctx.fillText('COUNTING LINE', 20, countingLineY - 10);

    tracks.forEach(track => {
        const [x, y, w, h] = track.bbox;
        ctx.strokeStyle = getCategoryColor(track.className);
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = getCategoryColor(track.className);
        ctx.fillRect(x, y - 22, 110, 22);

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px Segoe UI';
        ctx.fillText(`${track.className.toUpperCase()} #${track.id} ${(track.confidence * 100).toFixed(0)}%`, x + 4, y - 6);
    });
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
    document.getElementById('count-car').innerText = counts.car;
    document.getElementById('count-moto').innerText = counts.motorcycle;
    document.getElementById('count-bus').innerText = counts.bus;
    document.getElementById('count-truck').innerText = counts.truck;
    document.getElementById('count-total').innerText = counts.total;

    const activeVehicles = tracks.length;
    let density = 'LOW';
    let densityClass = 'low';

    if (activeVehicles >= highDensityThreshold) {
        density = 'HIGH';
        densityClass = 'high';
        setCongestion(true);
    } else if (activeVehicles >= lowDensityThreshold) {
        density = 'MEDIUM';
        densityClass = 'medium';
        setCongestion(false);
    } else {
        setCongestion(false);
    }

    const densityBadge = document.getElementById('density-status');
    densityBadge.className = `density-badge ${densityClass}`;
    densityBadge.innerText = density;

    if (chartInstance) {
        chartInstance.data.datasets[0].data = [counts.car, counts.motorcycle, counts.bus, counts.truck];
        chartInstance.update();
    }
}

function setCongestion(isCongested) {
    const banner = document.getElementById('congestion-banner');
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