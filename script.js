// Đồng hồ thời gian thực
setInterval(() => {
    document.getElementById('clock').innerText = new Date().toLocaleTimeString();
}, 1000);

const videoElem = document.getElementById('video-source');
const canvasElem = document.getElementById('canvas');
const ctx = canvasElem.getContext('2d');
const fileInput = document.getElementById('upload-video');
const fileNameDisplay = document.getElementById('file-name');
const statusElem = document.getElementById('system-status');

let aiRunning = false;
let session = null;
let modelLoaded = false;
let lastTime = performance.now();
let frameCount = 0;

// Biến toàn cục quản lý danh sách xe đang được tracking và chống nhảy số
let trackedObjects = [];
let nextTrackId = 1;
const MAX_MISSING_FRAMES = 5; // Số frame tối đa cho phép mất dấu trước khi xóa xe

// Xử lý khi người dùng chọn file video
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        fileNameDisplay.innerText = file.name;
        const videoUrl = URL.createObjectURL(file);
        videoElem.src = videoUrl;
        videoElem.load();
        videoElem.onloadeddata = () => {
            // Ép hiển thị frame đầu tiên lên canvas
            videoElem.play();
            setTimeout(() => videoElem.pause(), 100);
            updateStatus("AI READY", "waiting");
        };
    }
});

function updateStatus(text, className) {
    statusElem.innerText = text;
    statusElem.className = `status-badge ${className}`;
}

// Tải Model YOLOv10 ONNX trực tiếp từ file cục bộ cùng cấp
async function loadYOLOModel() {
    if (modelLoaded) return true;
    try {
        updateStatus("Đang tải Model YOLOv10...", "waiting");
        
        const modelUrl = "./yolov10.onnx"; 
        
        session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['webgl', 'wasm'] });
        modelLoaded = true;
        return true;
    } catch (err) {
        console.error("Lỗi tải model:", err);
        updateStatus("Lỗi tải Model AI", "stopped");
        alert("Không thể tải mô hình AI. Hãy kiểm tra lại tên file yolov10.onnx!");
        return false;
    }
}

async function startAI() {
    if (!videoElem.src || videoElem.src === window.location.href) {
        alert("Vui lòng bấm 'Chọn Video' trước khi chạy AI!");
        return;
    }

    if (!modelLoaded) {
        let loaded = await loadYOLOModel();
        if (!loaded) return;
    }

    aiRunning = true;
    updateStatus("AI RUNNING", "active");
    videoElem.play();
    runAIFrame();
}

function stopAI() {
    aiRunning = false;
    videoElem.pause();
    updateStatus("AI STOPPED", "stopped");
}

// Vòng lặp xử lý từng khung hình video
async function runAIFrame() {
    if (!aiRunning || videoElem.paused || videoElem.ended) {
        if (videoElem.ended) {
            updateStatus("AI READY", "waiting");
            aiRunning = false;
        }
        return;
    }

    // Tính toán FPS
    let now = performance.now();
    frameCount++;
    if (now - lastTime >= 1000) {
        document.getElementById('fps-display').innerText = Math.round((frameCount * 1000) / (now - lastTime));
        frameCount = 0;
        lastTime = now;
    }

    // Vẽ frame hiện tại từ video lên canvas
    ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);

    try {
        // 1. Tiền xử lý ảnh cho YOLOv10 (Scale về 640x640, chuẩn hóa RGB [0-1])
        const inputTensor = preprocessImage(canvasElem);
        
        // 2. Chạy suy luận qua ONNX Runtime
        const feeds = { images: inputTensor };
        const results = await session.run(feeds);
        const output = results[Object.keys(results)[0]];

        // 3. Phân tích kết quả đầu ra kết hợp Tracking và Buffer mượt số lượng
        const detections = parseYOLOv10Output(output.data, canvasElem.width, canvasElem.height);

        // 4. Vẽ bounding box và tên đối tượng lên màn hình
        renderDetections(detections);

    } catch (err) {
        console.error("Lỗi suy luận khung hình:", err);
    }

    if (aiRunning) {
        requestAnimationFrame(runAIFrame);
    }
}

// Hàm tiền xử lý đưa canvas về Tensor [1, 3, 640, 640]
function preprocessImage(srcCanvas) {
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 640;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(srcCanvas, 0, 0, 640, 640);
    let imgData = tempCtx.getImageData(0, 0, 640, 640);
    let data = imgData.data;

    let red = new Float32Array(640 * 640);
    let green = new Float32Array(640 * 640);
    let blue = new Float32Array(640 * 640);

    for (let i = 0; i < data.length / 4; i++) {
        red[i] = data[i * 4] / 255.0;
        green[i] = data[i * 4 + 1] / 255.0;
        blue[i] = data[i * 4 + 2] / 255.0;
    }

    let floatData = new Float32Array(3 * 640 * 640);
    floatData.set(red, 0);
    floatData.set(green, 640 * 640);
    floatData.set(blue, 2 * 640 * 640);

    return new ort.Tensor('float32', floatData, [1, 3, 640, 640]);
}

// Xử lý ma trận đầu ra YOLOv10 kết hợp thuật toán Tracking & Smoothing để chống nhảy số
function parseYOLOv10Output(outputData, origW, origH) {
    let rawBoxes = [];
    let numBoxes = outputData.length / 6;
    const targetClasses = { 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck' };

    // Hạ ngưỡng confidence xuống 0.35 để bắt xe máy tốt hơn
    for (let i = 0; i < numBoxes; i++) {
        let offset = i * 6;
        let x1_norm = outputData[offset + 0];
        let y1_norm = outputData[offset + 1];
        let x2_norm = outputData[offset + 2];
        let y2_norm = outputData[offset + 3];
        let score = outputData[offset + 4];
        let classId = Math.round(outputData[offset + 5]);

        if (score > 0.35 && targetClasses[classId]) {
            let x1 = x1_norm * (origW / 640);
            let y1 = y1_norm * (origH / 640);
            let x2 = x2_norm * (origW / 640);
            let y2 = y2_norm * (origH / 640);

            rawBoxes.push({ box: [x1, y1, x2, y2], score: score, class: targetClasses[classId] });
        }
    }

    let currentDetections = [];

    // Thuật toán IoU Tracker đơn giản gắn ID qua các frame
    rawBoxes.forEach(det => {
        let [x1, y1, x2, y2] = det.box;
        let cx = (x1 + x2) / 2;
        let cy = (y1 + y2) / 2;
        
        let matchedTrack = null;
        let minDistance = 60; // Khoảng cách pixel tối đa để nhận diện là cùng một xe

        for (let track of trackedObjects) {
            let [tx1, ty1, tx2, ty2] = track.box;
            let tcx = (tx1 + tx2) / 2;
            let tcy = (ty1 + ty2) / 2;
            let distance = Math.hypot(cx - tcx, cy - tcy);

            if (distance < minDistance && track.class === det.class) {
                minDistance = distance;
                matchedTrack = track;
            }
        }

        if (matchedTrack) {
            matchedTrack.box = det.box;
            matchedTrack.score = det.score;
            matchedTrack.missingFrames = 0;
            currentDetections.push(matchedTrack);
        } else {
            let newTrack = {
                id: nextTrackId++,
                box: det.box,
                score: det.score,
                class: det.class,
                missingFrames: 0
            };
            trackedObjects.push(newTrack);
            currentDetections.push(newTrack);
        }
    });

    // Giữ lại các xe bị mất dấu tạm thời trong vài frame (tránh sụt giảm số lượng đột ngột)
    trackedObjects.forEach(track => {
        if (!currentDetections.includes(track)) {
            track.missingFrames++;
            if (track.missingFrames < MAX_MISSING_FRAMES) {
                currentDetections.push(track);
            }
        }
    });

    trackedObjects = trackedObjects.filter(track => track.missingFrames < MAX_MISSING_FRAMES);

    // Thống kê số lượng phương tiện ổn định theo danh sách tracking
    let counts = { Car: 0, Motorcycle: 0, Bus: 0, Truck: 0 };
    currentDetections.forEach(item => {
        if (counts[item.class] !== undefined) {
            counts[item.class]++;
        }
    });

    // Cập nhật giao diện Dashboard đếm số lượng
    document.getElementById('count-car').innerText = counts.Car;
    document.getElementById('count-moto').innerText = counts.Motorcycle;
    document.getElementById('count-bus').innerText = counts.Bus;
    document.getElementById('count-truck').innerText = counts.Truck;
    document.getElementById('count-total').innerText = counts.Car + counts.Motorcycle + counts.Bus + counts.Truck;

    return currentDetections;
}

// Vẽ bounding box và tên đối tượng lên màn hình canvas
function renderDetections(boxes) {
    ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);

    boxes.forEach(item => {
        let [x1, y1, x2, y2] = item.box;
        
        // Vẽ khung Bounding Box
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Vẽ nhãn tên đối tượng, ID & độ tin cậy
        let text = `${item.class} #${item.id} (${Math.round(item.score * 100)}%)`;
        ctx.font = '12px Segoe UI, sans-serif';
        let textWidth = ctx.measureText(text).width;

        ctx.fillStyle = '#22c55e';
        ctx.fillRect(x1, y1 > 22 ? y1 - 22 : 0, textWidth + 10, 22);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 5, (y1 > 22 ? y1 - 22 : 0) + 15);
    });
}