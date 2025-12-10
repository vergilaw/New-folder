import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.hog_extractor import HOGExtractor


def main():
    # Try different camera indices (DroidCam usually uses 1 or 2)
    cap = None
    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Camera found at index {idx}")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        print("Cannot open camera. Make sure DroidCam is running.")
        return

    hog = HOGExtractor(window_size=(64, 64), cell_size=8, orientations=9)
    
    print("Press 'q' to quit")
    print("Press 's' to toggle split view")
    print("Press 'c' to change cell size (8/16)")
    
    split_view = True
    cell_size = 8

    box_size = 200  # Size of center box

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Calculate center box coordinates
        cx, cy = w // 2, h // 2
        x1 = cx - box_size // 2
        y1 = cy - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Extract center region for HOG
        center_region = frame[y1:y2, x1:x2]
        
        # Extract HOG with visualization from center region only
        features, hog_vis = hog.extract_with_visualization(center_region)
        
        # Convert HOG visualization to displayable format
        hog_vis = (hog_vis * 255).astype(np.uint8)
        hog_vis = cv2.resize(hog_vis, (box_size, box_size))
        hog_colored = cv2.applyColorMap(hog_vis, cv2.COLORMAP_JET)
        
        # Draw center box on frame
        result = frame.copy()
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, "Place face here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if split_view:
            # Show HOG visualization on the right side
            hog_display = cv2.resize(hog_colored, (200, 200))
            result[20:220, w-220:w-20] = hog_display
            cv2.rectangle(result, (w-220, 20), (w-20, 220), (255, 255, 255), 2)
            cv2.putText(result, f"HOG (cell={cell_size})", (w-220, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Overlay HOG on center box
            result[y1:y2, x1:x2] = cv2.addWeighted(center_region, 0.5, hog_colored, 0.5, 0)
        
        # Show feature dimension
        cv2.putText(result, f"Features: {len(features)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, f"Cell size: {cell_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('HOG Realtime', result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            split_view = not split_view
        elif key == ord('c'):
            cell_size = 16 if cell_size == 8 else 8
            hog = HOGExtractor(window_size=(64, 64), cell_size=cell_size, orientations=9)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
