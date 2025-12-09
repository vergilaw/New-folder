import cv2
import numpy as np
import joblib
from .hog_extractor import HOGExtractor
from .svm_classifier import SVMClassifier


class FaceDetector:
    def __init__(self, window_size=(64, 64), cell_size=8):
        self.hog = HOGExtractor(window_size=window_size, cell_size=cell_size)
        self.svm = SVMClassifier(kernel='rbf', C=1.0)
        self.window_size = window_size
        self.is_trained = False

    def train(self, positive_samples, negative_samples, test_size=0.2):
        from sklearn.model_selection import train_test_split
        X, y = [], []
        for img in positive_samples:
            X.append(self.hog.extract(img))
            y.append(1)
        for img in negative_samples:
            X.append(self.hog.extract(img))
            y.append(0)
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        self.svm.fit(X_train, y_train)
        accuracy, report = self.svm.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy*100:.2f}%")
        self.is_trained = True
        return accuracy

    def detect(self, image, confidence_threshold=0.6, scales=[1.0, 0.75, 0.5], stride=16, nms_threshold=0.3):
        if not self.is_trained:
            raise ValueError("Model not trained!")
        boxes = []
        h, w = image.shape[:2]
        win_w, win_h = self.window_size
        for scale in scales:
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w < win_w or new_h < win_h:
                continue
            resized = cv2.resize(image, (new_w, new_h))
            for y in range(0, new_h - win_h + 1, stride):
                for x in range(0, new_w - win_w + 1, stride):
                    window = resized[y:y+win_h, x:x+win_w]
                    features = self.hog.extract(window)
                    prob = self.svm.predict_proba([features])[0]
                    confidence = prob[1]
                    if confidence >= confidence_threshold:
                        boxes.append({'x': int(x / scale), 'y': int(y / scale), 'w': int(win_w / scale), 'h': int(win_h / scale), 'confidence': float(confidence)})
        return self._nms(boxes, nms_threshold)

    def _nms(self, boxes, threshold):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        kept = []
        while boxes:
            best = boxes.pop(0)
            kept.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < threshold]
        return kept

    def _iou(self, box1, box2):
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
        y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = box1['w'] * box1['h'] + box2['w'] * box2['h'] - intersection
        return intersection / union if union > 0 else 0

    def save(self, path):
        data = {'hog_config': self.hog.get_info(), 'svm': {'scaler': self.svm.scaler, 'model': self.svm.svm}, 'is_trained': self.is_trained}
        joblib.dump(data, path)

    def load(self, path):
        data = joblib.load(path)
        config = data['hog_config']
        self.hog = HOGExtractor(window_size=config['window_size'], cell_size=config['cell_size'])
        self.window_size = config['window_size']
        self.svm.scaler = data['svm']['scaler']
        self.svm.svm = data['svm']['model']
        self.svm.is_trained = True
        self.is_trained = data['is_trained']


def draw_detections(image, boxes, color=(0, 255, 0), thickness=2):
    result = image.copy()
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        conf = box.get('confidence', 0)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        label = f"{conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(result, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return result
