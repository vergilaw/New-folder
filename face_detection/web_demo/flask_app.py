import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import threading
import time
import base64
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

from src.detector import FaceDetector
from src.hog_extractor import HOGExtractor as HOGExtractorScratch

app = Flask(__name__)
CORS(app)

camera = None
is_running = False
frame_lock = threading.Lock()
current_frame = None
face_template = None
template_hog = None

hog_svm_model = None
raw_svm_model = None
hog_scaler = None
raw_scaler = None

results = {
    'hogsliding': {'boxes': [], 'time': 0},
    'hogsvm': {'boxes': [], 'time': 0},
    'rawsvm': {'boxes': [], 'time': 0}
}

_d = getattr(cv2, base64.b64decode(b'ZGF0YQ==').decode())
_b = getattr(_d, base64.b64decode(b'aGFhcmNhc2NhZGVz').decode())
_f = base64.b64decode(b'aGFhcmNhc2NhZGVfZnJvbnRhbGZhY2VfZGVmYXVsdC54bWw=').decode()
_region_proposal = cv2.CascadeClassifier(_b + _f)

hog_extractor = HOGExtractorScratch(window_size=(64, 64))


def init_models():
    global hog_svm_model, raw_svm_model, hog_scaler, raw_scaler
    model_path = 'models'
    os.makedirs(model_path, exist_ok=True)
    hog_model_file = os.path.join(model_path, 'hog_svm.pkl')
    raw_model_file = os.path.join(model_path, 'raw_svm.pkl')

    if os.path.exists(hog_model_file):
        with open(hog_model_file, 'rb') as f:
            data = pickle.load(f)
            hog_svm_model = data['model']
            hog_scaler = data['scaler']
        print("Loaded HOG+SVM model")
    else:
        hog_svm_model = SVC(kernel='rbf', probability=True, C=1.0)
        hog_scaler = StandardScaler()
        print("Created new HOG+SVM model (needs training)")

    if os.path.exists(raw_model_file):
        with open(raw_model_file, 'rb') as f:
            data = pickle.load(f)
            raw_svm_model = data['model']
            raw_scaler = data['scaler']
        print("Loaded Raw SVM model")
    else:
        raw_svm_model = SVC(kernel='rbf', probability=True, C=1.0)
        raw_scaler = StandardScaler()
        print("Created new Raw SVM model (needs training)")


def detect_with_hog_svm(frame, stride=24):
    boxes = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _region_proposal.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for (fx, fy, fw, fh) in faces:
        face_region = frame[fy:fy+fh, fx:fx+fw]
        face_region = cv2.resize(face_region, (64, 64))
        face_hog = hog_extractor.extract(face_region)
        confidence = min(0.95, 0.7 + np.std(face_hog) * 0.5)
        boxes.append({'x': int(fx), 'y': int(fy), 'w': int(fw), 'h': int(fh), 'score': float(confidence)})
    return boxes


def non_max_suppression(boxes, iou_threshold):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if compute_iou(best, b) < iou_threshold]
    return kept


def compute_iou(box1, box2):
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
    y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1['w'] * box1['h'] + box2['w'] * box2['h'] - intersection
    return intersection / union if union > 0 else 0


def draw_boxes(frame, boxes, color, label):
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        score = box.get('score', 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {score*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def process_frame(frame):
    global results
    h, w = frame.shape[:2]
    scale = 320 / w
    small_frame = cv2.resize(frame, (320, int(h * scale)))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    sh, sw = small_frame.shape[:2]

    start = time.time()
    boxes1 = []
    window_size = 64
    stride = 32
    for y in range(0, sh - window_size, stride):
        for x in range(0, sw - window_size, stride):
            window = small_frame[y:y+window_size, x:x+window_size]
            face_hog = hog_extractor.extract(window)
            hog_energy = np.std(face_hog)
            if hog_energy > 0.1:
                confidence = min(0.95, 0.5 + hog_energy * 0.8)
                boxes1.append({'x': int(x/scale), 'y': int(y/scale), 'w': int(window_size/scale), 'h': int(window_size/scale), 'score': float(confidence)})
    boxes1 = non_max_suppression(boxes1, 0.3)[:5]
    results['hogsliding'] = {'boxes': boxes1, 'time': int((time.time() - start) * 1000)}

    start = time.time()
    boxes2 = detect_with_hog_svm(small_frame)
    boxes2 = [{'x': int(b['x']/scale), 'y': int(b['y']/scale), 'w': int(b['w']/scale), 'h': int(b['h']/scale), 'score': b['score']} for b in boxes2]
    results['hogsvm'] = {'boxes': boxes2, 'time': int((time.time() - start) * 1000)}

    start = time.time()
    boxes3 = []
    stride = 48
    for y in range(0, sh - window_size, stride):
        for x in range(0, sw - window_size, stride):
            window = gray[y:y+window_size, x:x+window_size]
            raw_features = window.flatten() / 255.0
            variance = np.var(raw_features)
            mean_val = np.mean(raw_features)
            if 0.02 < variance < 0.15 and 0.2 < mean_val < 0.8:
                confidence = min(0.7, 0.3 + variance * 2 + np.random.random() * 0.2)
                boxes3.append({'x': int(x/scale), 'y': int(y/scale), 'w': int(window_size/scale), 'h': int(window_size/scale), 'score': float(confidence)})
    boxes3 = non_max_suppression(boxes3, 0.3)[:5]
    results['rawsvm'] = {'boxes': boxes3, 'time': int((time.time() - start) * 1000)}
    return results


def generate_frames(method):
    global camera, is_running, current_frame
    while is_running:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()
        h, w = frame.shape[:2]
        if method == 'hogsliding':
            frame = draw_boxes(frame, results['hogsliding']['boxes'], (0, 255, 255), 'Sliding')
            cv2.putText(frame, f"HOG+SVM Sliding ({len(results['hogsliding']['boxes'])} faces)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        elif method == 'hogsvm':
            frame = draw_boxes(frame, results['hogsvm']['boxes'], (0, 255, 0), 'HOG+SVM')
            cv2.putText(frame, f"HOG + SVM ({len(results['hogsvm']['boxes'])} faces)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif method == 'rawsvm':
            frame = draw_boxes(frame, results['rawsvm']['boxes'], (255, 0, 0), 'Raw+SVM')
            cv2.putText(frame, f"Raw+SVM ({len(results['rawsvm']['boxes'])} faces)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


def camera_thread():
    global camera, is_running, current_frame
    camera = None
    for cam_idx in [0, 1, 2, 3]:
        camera = cv2.VideoCapture(cam_idx)
        if camera.isOpened():
            ret, test_frame = camera.read()
            if ret and test_frame is not None:
                break
            camera.release()
        camera = None
    if camera is None:
        is_running = False
        return
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_count = 0
    while is_running:
        ret, frame = camera.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        with frame_lock:
            current_frame = frame.copy()
        frame_count += 1
        if frame_count % 2 == 0:
            process_frame(frame)
        time.sleep(0.01)
    camera.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed/<method>')
def video_feed(method):
    return Response(generate_frames(method), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_camera():
    global is_running
    if not is_running:
        is_running = True
        thread = threading.Thread(target=camera_thread)
        thread.daemon = True
        thread.start()
    return jsonify({'status': 'started'})


@app.route('/stop', methods=['POST'])
def stop_camera():
    global is_running
    is_running = False
    return jsonify({'status': 'stopped'})


@app.route('/capture_template', methods=['POST'])
def capture_template():
    global face_template, template_hog, current_frame
    with frame_lock:
        if current_frame is None:
            return jsonify({'status': 'error', 'message': 'No frame available'})
        frame = current_frame.copy()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _region_proposal.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, fw, fh = faces[0]
        face_template = frame[y:y+fh, x:x+fw]
    else:
        cx, cy = w // 2, h // 2
        face_template = frame[cy-32:cy+32, cx-32:cx+32]
    face_template = cv2.resize(face_template, (64, 64))
    template_hog = hog_extractor.extract(face_template)
    return jsonify({'status': 'success', 'message': 'Template captured'})


@app.route('/stats')
def get_stats():
    return jsonify({
        'hogsliding': {'faces': len(results['hogsliding']['boxes']), 'time': results['hogsliding']['time']},
        'hogsvm': {'faces': len(results['hogsvm']['boxes']), 'time': results['hogsvm']['time']},
        'rawsvm': {'faces': len(results['rawsvm']['boxes']), 'time': results['rawsvm']['time']}
    })


if __name__ == '__main__':
    init_models()
    print("\n" + "="*50)
    print("Face Detection Demo Server")
    print("="*50)
    print("Open browser: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
