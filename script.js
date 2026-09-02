// Đồng hồ thời gian thực[cite: 1]
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

// Xử lý khi người dùng chọn file video[cite: 1]
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        fileNameDisplay.innerText = file.name;
        const videoUrl = URL.createObjectURL(file);
        videoElem.src = videoUrl;
        videoElem.load();
        videoElem.onloadeddata = () => {
            ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);
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
        
        // Đã sửa đường dẫn trỏ đúng vào file yolov10.onnx nằm cùng cấp
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

// Vòng lặp xử lý từng khung hình video[cite: 1]
async function runAIFrame() {
    if (!aiRunning || videoElem.paused || videoElem.ended) {
        if (videoElem.ended) {
            updateStatus("AI READY", "waiting");
            aiRunning = false;
        }
        return;
    }

    // Tính toán FPS[cite: 1]
    let now = performance.now();
    frameCount++;
    if (now - lastTime >= 1000) {
        document.getElementById('fps-display').innerText = Math.round((frameCount * 1000) / (now - lastTime));
        frameCount = 0;
        lastTime = now;
    }

    // Vẽ frame hiện tại từ video lên canvas[cite: 1]
    ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);

    try {
        // 1. Tiền xử lý ảnh cho YOLOv10 (Scale về 640x640, chuẩn hóa RGB [0-1])[cite: 1]
        const inputTensor = preprocessImage(canvasElem);
        
        // 2. Chạy suy luận qua ONNX Runtime[cite: 1]
        const feeds = { images: inputTensor };
        const results = await session.run(feeds);
        const output = results[Object.keys(results)[0]];

        // 3. Phân tích kết quả đầu ra của YOLOv10 (Đã tích hợp sẵn NMS, xuất ra tensor gọn [300, 6])[cite: 1]
        const detections = parseYOLOv10Output(output.data, canvasElem.width, canvasElem.height);

        // 4. Vẽ bounding box và cập nhật thống kê[cite: 1]
        renderDetections(detections);

    } catch (err) {
        console.error("Lỗi suy luận khung hình:", err);
    }

    if (aiRunning) {
        requestAnimationFrame(runAIFrame);
    }
}

// Hàm tiền xử lý đưa canvas về Tensor [1, 3, 640, 640][cite: 1]
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

// Xử lý ma trận đầu ra YOLOv10 (Cấu trúc đặc thù của YOLOv10: 300 dòng, mỗi dòng chứa [x1, y1, x2, y2, score, class_id])[cite: 1]
function parseYOLOv10Output(outputData, origW, origH) {
    let boxes = [];
    let counts = { car: 0, motorcycle: 0, bus: 0, truck: 0 };

    // ID trong tập COCO dataset: 2: car, 3: motorcycle, 5: bus, 7: truck[cite: 1]
    const targetClasses = { 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck' };
    
    // YOLOv10 xuất ra mảng phẳng chứa 300 bounding box (mỗi box gồm 6 giá trị)[cite: 1]
    let numBoxes = outputData.length / 6;

    for (let i = 0; i < numBoxes; i++) {
        let offset = i * 6;
        let x1_norm = outputData[offset + 0];
        let y1_norm = outputData[offset + 1];
        let x2_norm = outputData[offset + 2];
        let y2_norm = outputData[offset + 3];
        let score = outputData[offset + 4];
        let classId = Math.round(outputData[offset + 5]);

        // Ngưỡng tin cậy (Confidence threshold = 0.45) và lọc đúng các class phương tiện giao thông[cite: 1]
        if (score > 0.30 && targetClasses[classId]) {
            // Quy đổi tọa độ chuẩn hóa về kích thước thực của video gốc[cite: 1]
            let x1 = x1_norm * (origW / 640);
            let y1 = y1_norm * (origH / 640);
            let x2 = x2_norm * (origW / 640);
            let y2 = y2_norm * (origH / 640);

            let label = targetClasses[classId];
            if (classId === 2) counts.car++;
            else if (classId === 3) counts.motorcycle++;
            else if (classId === 5) counts.bus++;
            else if (classId === 7) counts.truck++;

            boxes.push({ box: [x1, y1, x2, y2], score: score, class: label });
        }
    }

    // Cập nhật giao diện Dashboard đếm số lượng[cite: 1]
    document.getElementById('count-car').innerText = counts.car;
    document.getElementById('count-moto').innerText = counts.motorcycle;
    document.getElementById('count-bus').innerText = counts.bus;
    document.getElementById('count-truck').innerText = counts.truck;
    document.getElementById('count-total').innerText = counts.car + counts.motorcycle + counts.bus + counts.truck;

    return boxes;
}

// Vẽ bounding box và tên đối tượng lên màn hình canvas[cite: 1]
function renderDetections(boxes) {
    ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);

    boxes.forEach(item => {
        let [x1, y1, x2, y2] = item.box;
        
        // Vẽ khung Bounding Box[cite: 1]
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Vẽ nhãn tên đối tượng & độ tin cậy[cite: 1]
        let text = `${item.class} (${Math.round(item.score * 100)}%)`;
        ctx.font = '12px Segoe UI, sans-serif';
        let textWidth = ctx.measureText(text).width;

        ctx.fillStyle = '#22c55e';
        ctx.fillRect(x1, y1 > 22 ? y1 - 22 : 0, textWidth + 10, 22);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 5, (y1 > 22 ? y1 - 22 : 0) + 15);
    });
}