import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def compute_hog_fullsize(image, cell_size=8, orientations=9):
    """Compute HOG on full image without resizing"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64) / 255.0
    h, w = gray.shape
    
    # Compute gradients
    gx = np.zeros((h, w), dtype=np.float64)
    gy = np.zeros((h, w), dtype=np.float64)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180
    
    # Build cell histograms
    cells_y = h // cell_size
    cells_x = w // cell_size
    bin_width = 180.0 / orientations
    
    histograms = np.zeros((cells_y, cells_x, orientations))
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            y1, y2 = cy * cell_size, (cy + 1) * cell_size
            x1, x2 = cx * cell_size, (cx + 1) * cell_size
            
            cell_mag = magnitude[y1:y2, x1:x2].flatten()
            cell_ori = orientation[y1:y2, x1:x2].flatten()
            
            for m, o in zip(cell_mag, cell_ori):
                bin_idx = int(o / bin_width) % orientations
                histograms[cy, cx, bin_idx] += m
    
    # Visualize
    vis_image = np.zeros((h, w), dtype=np.float64)
    bin_angles = np.arange(orientations) * bin_width + bin_width / 2
    bin_angles_rad = bin_angles * np.pi / 180
    max_mag = np.max(histograms) + 1e-6
    
    for cy in range(cells_y):
        for cx in range(cells_x):
            center_y = cy * cell_size + cell_size // 2
            center_x = cx * cell_size + cell_size // 2
            
            for bin_idx in range(orientations):
                mag = histograms[cy, cx, bin_idx]
                angle = bin_angles_rad[bin_idx]
                length = (mag / max_mag) * (cell_size // 2)
                
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                
                x1, y1 = int(center_x - dx), int(center_y - dy)
                x2, y2 = int(center_x + dx), int(center_y + dy)
                cv2.line(vis_image, (x1, y1), (x2, y2), 1.0, 1)
    
    return vis_image


def extract_hog_visualization(image_path, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read: {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Original scale HOG (full size)
    hog_vis1 = compute_hog_fullsize(image, cell_size=8)
    hog_vis1 = (hog_vis1 * 255).astype(np.uint8)
    
    # 1/2x scale HOG
    image_half = cv2.resize(image, (w // 2, h // 2))
    hog_vis2 = compute_hog_fullsize(image_half, cell_size=8)
    hog_vis2 = (hog_vis2 * 255).astype(np.uint8)
    hog_vis2 = cv2.resize(hog_vis2, (w, h))  # Resize back for display
    
    # Create dark background result
    padding = 60
    result_h = h * 2 + padding * 3
    result_w = w + padding * 2
    result = np.zeros((result_h, result_w, 3), dtype=np.uint8)
    result[:] = (30, 30, 30)  # Dark gray background
    
    # Place HOG images
    hog1_colored = cv2.cvtColor(hog_vis1, cv2.COLOR_GRAY2BGR)
    hog2_colored = cv2.cvtColor(hog_vis2, cv2.COLOR_GRAY2BGR)
    
    y1 = padding
    y2 = padding * 2 + h
    x = padding
    
    result[y1:y1+h, x:x+w] = hog1_colored
    result[y2:y2+h, x:x+w] = hog2_colored
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "Original Scale", (x, y1 - 20), font, 0.8, (255, 255, 255), 2)
    cv2.putText(result, "1/2x Scale", (x, y2 - 20), font, 0.8, (255, 255, 255), 2)
    
    # Save output
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_hog{ext}"
    
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")
    print(f"Image size: {w}x{h}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Input image path')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    args = parser.parse_args()
    
    extract_hog_visualization(args.image, args.output)
