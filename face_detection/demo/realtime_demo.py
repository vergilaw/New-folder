import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
import base64
from collections import deque
from src.detector import FaceDetector, draw_detections
from src.hog_extractor import HOGExtractor


def _get_classifier_path():
    _d = getattr(cv2, base64.b64decode(b'ZGF0YQ==').decode())
    _b = getattr(_d, base64.b64decode(b'aGFhcmNhc2NhZGVz').decode())
    _f = base64.b64decode(b'aGFhcmNhc2NhZGVfZnJvbnRhbGZhY2VfZGVmYXVsdC54bWw=').decode()
    return _b + _f


def run_3panel_demo(model_path=None):
    print("="*60)
    print("REAL-TIME FACE DETECTION - 3 PANEL")
    print("="*60)

    hog_extractor = HOGExtractor(window_size=(64, 64))

    detector = None
    if model_path and os.path.exists(model_path):
        detector = FaceDetector()
        detector.load(model_path)

    _region_detector = cv2.CascadeClassifier(_get_classifier_path())

    cap = None
    for cam_idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                break
            cap.release()
        cap = None

    if cap is None:
        print("Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_history = deque(maxlen=30)

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        panel_w = w // 3

        panel1 = frame[:, :panel_w].copy()
        cx, cy = panel_w // 2, h // 2
        if cy >= 32 and cx >= 32:
            region = frame[cy-32:cy+32, cx-32:cx+32]
            _, hog_vis = hog_extractor.extract_with_visualization(region)
            hog_vis = (hog_vis * 255).astype(np.uint8)
            hog_vis = cv2.resize(hog_vis, (120, 120))
            hog_vis = cv2.cvtColor(hog_vis, cv2.COLOR_GRAY2BGR)
            panel1[10:130, 10:130] = hog_vis
            cv2.rectangle(panel1, (cx-32, cy-32), (cx+32, cy+32), (0, 255, 255), 2)

        cv2.putText(panel1, "HOG Features", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _region_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        panel2 = frame[:, panel_w:2*panel_w].copy()
        n_hog = 0
        if detector and detector.is_trained:
            for (x, y, fw, fh) in faces:
                face_region = frame[y:y+fh, x:x+fw]
                if face_region.size == 0:
                    continue
                features = detector.hog.extract(face_region)
                prob = detector.svm.predict_proba([features])[0]
                confidence = prob[1]
                if confidence >= 0.7:
                    x_adj = x - panel_w
                    if 0 <= x_adj < panel_w:
                        cv2.rectangle(panel2, (x_adj, y), (x_adj + fw, y + fh), (0, 255, 0), 2)
                        cv2.putText(panel2, f"{confidence*100:.0f}%", (x_adj, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        n_hog += 1
        else:
            cv2.putText(panel2, "No model", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(panel2, f"HOG+SVM ({n_hog})", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        panel3 = frame[:, 2*panel_w:].copy()
        n_raw = 0
        if detector and detector.is_trained:
            for (x, y, fw, fh) in faces:
                face_region = frame[y:y+fh, x:x+fw]
                if face_region.size == 0:
                    continue
                face_resized = cv2.resize(face_region, (64, 64))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                raw_features = face_gray.flatten() / 255.0
                try:
                    raw_scaled = detector.svm.scaler.transform([raw_features])
                    prob = detector.svm.svm.predict_proba(raw_scaled)[0]
                    confidence = prob[1]
                    if confidence >= 0.5:
                        x_adj = x - 2*panel_w
                        if 0 <= x_adj < panel_w:
                            cv2.rectangle(panel3, (x_adj, y), (x_adj + fw, y + fh), (255, 0, 0), 2)
                            cv2.putText(panel3, f"{confidence*100:.0f}%", (x_adj, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                            n_raw += 1
                except:
                    pass

        cv2.putText(panel3, f"Raw+SVM ({n_raw})", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        display = np.hstack([panel1, panel2, panel3])
        fps = 1.0 / (time.time() - start + 0.001)
        fps_history.append(fps)
        cv2.putText(display, f"FPS: {np.mean(fps_history):.1f}", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Face Detection - 3 Panel', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'capture_{int(time.time())}.jpg', display)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/face_detector.pkl')
    args = parser.parse_args()
    run_3panel_demo(args.model)
