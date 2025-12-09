import os
import cv2
import numpy as np
from glob import glob
from src.detector import FaceDetector


def download_lfw_kaggle():
    try:
        import kagglehub
        path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
        return path
    except:
        return None


def load_lfw_faces(data_path=None, max_samples=2000):
    possible_paths = [data_path, "data/lfw", "data/lfw-dataset", os.path.expanduser("~/.cache/kagglehub/datasets/jessicali9530/lfw-dataset")]
    lfw_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            lfw_path = p
            break
    if not lfw_path:
        downloaded = download_lfw_kaggle()
        if downloaded:
            return load_lfw_faces(downloaded, max_samples)
        return []
    faces = []
    for root, dirs, files in os.walk(lfw_path):
        for f in files:
            if f.lower().endswith('.jpg'):
                img_path = os.path.join(root, f)
                img = cv2.imread(img_path)
                if img is not None:
                    faces.append(img)
                    if len(faces) >= max_samples:
                        break
        if len(faces) >= max_samples:
            break
    return faces


def load_custom_faces(data_path='data/faces'):
    faces = []
    if not os.path.exists(data_path):
        return faces
    for img_path in glob(os.path.join(data_path, '*.*')):
        img = cv2.imread(img_path)
        if img is not None:
            faces.append(img)
    return faces


def generate_negative_samples(n_samples=2000, size=64):
    negative = []
    for i in range(n_samples):
        strategy = i % 4
        if strategy == 0:
            patch = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        elif strategy == 1:
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            for c in range(3):
                patch[:, :, c] = np.tile(np.linspace(0, 255, size), (size, 1))
        elif strategy == 2:
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            for c in range(3):
                patch[:, :, c] = np.tile(np.linspace(0, 255, size).reshape(-1, 1), (1, size))
        else:
            patch = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
            for _ in range(np.random.randint(2, 6)):
                x1, y1 = np.random.randint(0, size-10, 2)
                x2, y2 = x1 + np.random.randint(5, 20), y1 + np.random.randint(5, 20)
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.rectangle(patch, (x1, y1), (min(x2, size-1), min(y2, size-1)), color, -1)
        negative.append(patch)
    return negative


def load_negative_from_images(images_path='data/negative', n_patches=2000, patch_size=64):
    negative = []
    if not os.path.exists(images_path):
        return negative
    image_files = glob(os.path.join(images_path, '*.*'))
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        n_per_image = min(20, n_patches // len(image_files) + 1)
        for _ in range(n_per_image):
            if h > patch_size and w > patch_size:
                x = np.random.randint(0, w - patch_size)
                y = np.random.randint(0, h - patch_size)
                patch = img[y:y+patch_size, x:x+patch_size]
                negative.append(patch)
        if len(negative) >= n_patches:
            break
    return negative[:n_patches]


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    positive = load_lfw_faces(max_samples=2000)
    custom_faces = load_custom_faces('data/faces')
    positive.extend(custom_faces)

    if len(positive) < 100:
        print("Not enough face samples!")
        return

    negative = load_negative_from_images('data/negative', n_patches=len(positive))
    if len(negative) < len(positive):
        n_need = len(positive) - len(negative)
        negative.extend(generate_negative_samples(n_need))

    n_samples = min(len(positive), len(negative))
    positive = positive[:n_samples]
    negative = negative[:n_samples]

    detector = FaceDetector(window_size=(64, 64), cell_size=8)
    accuracy = detector.train(positive, negative, test_size=0.2)

    model_path = 'models/face_detector.pkl'
    detector.save(model_path)

    print(f"\nTraining complete! Accuracy: {accuracy*100:.2f}%")
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
