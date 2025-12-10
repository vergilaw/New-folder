import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from src.detector import FaceDetector


def download_lfw_kaggle():
    try:
        import kagglehub
        path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
        return path
    except:
        return None


def load_lfw_faces(data_path=None, max_samples=2000):
    possible_paths = [data_path, "data/lfw", "data/lfw-dataset", 
                      os.path.expanduser("~/.cache/kagglehub/datasets/jessicali9530/lfw-dataset")]
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


def plot_training_results(results, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(results['y_test'], results['y_pred'])
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_title(f'Confusion Matrix (Accuracy: {results["accuracy"]*100:.2f}%)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Non-Face', 'Face'])
    ax1.set_yticklabels(['Non-Face', 'Face'])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=ax1)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/confusion_matrix.png")
    
    # 2. ROC Curve
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_prob'])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/roc_curve.png")
    
    # 3. Precision-Recall Curve
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(results['y_test'], results['y_prob'])
    ax3.plot(recall, precision, 'g-', linewidth=2)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/precision_recall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/precision_recall.png")

    # 4. PCA Explained Variance (all 500 components)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    if results['pca_variance'] is not None:
        n_components = len(results['pca_variance'])
        cumsum = np.cumsum(results['pca_variance'])
        
        # Plot all 500 components
        ax4.bar(range(n_components), results['pca_variance'], alpha=0.6, label='Individual', width=1.0)
        ax4.plot(range(n_components), cumsum, 'r-', linewidth=2, label='Cumulative')
        ax4.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Explained Variance Ratio')
        ax4.set_title(f'PCA: {results["reduced_dim"]} dims ({sum(results["pca_variance"])*100:.1f}%)')
        ax4.set_xlim(0, n_components)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No PCA Applied', ha='center', va='center', fontsize=14)
        ax4.set_title(f'Feature Dimension: {results["original_dim"]}')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pca_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/pca_variance.png")


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("Loading face samples...")
    positive = load_lfw_faces(max_samples=2000)
    
    if len(positive) < 100:
        print("Not enough face samples!")
        return

    print(f"Loaded {len(positive)} face samples")
    
    print("Generating negative samples...")
    negative = generate_negative_samples(len(positive))
    
    n_samples = min(len(positive), len(negative))
    positive = positive[:n_samples]
    negative = negative[:n_samples]
    
    print(f"Training with {n_samples} positive and {n_samples} negative samples...")
    
    # Train with PCA 500
    detector = FaceDetector(window_size=(64, 64), cell_size=8, pca_components=500)
    results = detector.train(positive, negative, test_size=0.2, return_details=True)
    
    # Save model
    model_path = 'models/face_detector.pkl'
    detector.save(model_path)
    print(f"\nModel saved: {model_path}")
    
    # Plot results
    plot_training_results(results, 'models')


if __name__ == "__main__":
    main()
