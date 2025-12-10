# So sánh 3 Phương pháp Phát hiện Khuôn mặt

## 1. Tổng quan

| Phương pháp | Mô tả |
|-------------|-------|
| **HOG + SVM (Sliding Window)** | HOG + SVM với sliding window cơ bản |
| **HOG + SVM (Optimized)** | HOG + SVM với NMS và tối ưu hóa |
| **Raw Pixel + SVM** | Dùng pixel thô, không trích xuất đặc trưng |

---

## 2. Kết quả thực tế

### Trên ảnh nhóm người thật:

| Phương pháp | Kết quả | Đánh giá |
|-------------|---------|----------|
| HOG+SVM (Sliding) | 64+ boxes, detect loạn | ❌ Quá nhiều false positive |
| HOG+SVM (Optimized) | 10 boxes, đúng vị trí mặt | ✅ Chính xác |
| Raw Pixel + SVM | **0 boxes** | ❌ Hoàn toàn thất bại |

### Phân tích:

**HOG + SVM (Sliding Window cơ bản):**
- Sliding window quét toàn bộ ảnh
- Không có Non-Maximum Suppression (NMS)
- Kết quả: Hàng chục boxes chồng chéo, nhiều false positive
- Vấn đề: Cần thêm NMS để loại bỏ boxes trùng lặp

**HOG + SVM (Optimized):**
- Có NMS để gộp các boxes trùng
- Threshold phù hợp
- Kết quả: Detect chính xác các khuôn mặt
- Đây là phương pháp production-ready

**Raw Pixel + SVM:**
- Không detect được bất kỳ khuôn mặt nào
- Accuracy 98% trên test set là **ảo** vì:
  - Test set đã được crop sẵn, mặt ở giữa
  - Khi sliding window trên ảnh thật, không match được pattern
  - Pixel values thay đổi hoàn toàn khi vị trí khác

---

## 3. Tại sao Raw Pixel thất bại hoàn toàn?

### 3.1. Vấn đề cốt lõi

Raw Pixel học được: "Pixel ở vị trí (32, 32) có giá trị ~150 → là mặt"

Nhưng khi sliding window:
- Mặt có thể ở bất kỳ vị trí nào trong window
- Pixel (32, 32) có thể là mũi, mắt, tóc, hoặc background
- → Model không nhận ra

### 3.2. So sánh trực quan

```
Training (mặt ở giữa):          Sliding window (mặt lệch):
┌─────────────────┐             ┌─────────────────┐
│                 │             │        ┌───┐    │
│    ┌─────┐      │             │        │ 👤│    │
│    │ 👤  │      │      vs     │        └───┘    │
│    └─────┘      │             │                 │
│                 │             │                 │
└─────────────────┘             └─────────────────┘
   Raw Pixel: ✅                   Raw Pixel: ❌
   HOG: ✅                         HOG: ✅
```

HOG detect được vì nó tìm **pattern của edges** (mắt, mũi, miệng), không quan tâm vị trí tuyệt đối.

---

## 4. Tại sao HOG hoạt động?

### 4.1. HOG capture cấu trúc

HOG không nhìn pixel values, mà nhìn **hướng của edges**:

- Mắt: Có edge ngang (lông mày) + edge tròn (con ngươi)
- Mũi: Có edge dọc ở giữa
- Miệng: Có edge ngang

Dù mặt ở đâu trong window, pattern edges này vẫn tồn tại.

### 4.2. Bất biến với ánh sáng

```
Ảnh sáng:                    Ảnh tối:
Pixel: [200, 210, 205]       Pixel: [50, 60, 55]
       ↓                            ↓
HOG: [→, →, ↗]               HOG: [→, →, ↗]  (giống nhau!)
```

---

## 5. Kết luận

### Ranking thực tế:

| # | Phương pháp | Thực tế |
|---|-------------|---------|
| 🥇 | HOG + SVM (Optimized) | Hoạt động tốt, production-ready |
| 🥈 | HOG + SVM (Sliding) | Cần thêm NMS, nhiều false positive |
| 🥉 | Raw Pixel + SVM | **Không hoạt động**, chỉ tốt trên test set ảo |

### Bài học:

1. **Accuracy trên test set ≠ Hiệu quả thực tế**
   - Test set cùng distribution với training → accuracy cao
   - Ảnh thực tế khác distribution → fail

2. **Feature engineering quan trọng**
   - Raw pixel không có tính bất biến
   - HOG có bất biến với ánh sáng, vị trí (trong cell)

3. **Post-processing cần thiết**
   - NMS để loại bỏ duplicate boxes
   - Threshold tuning cho từng use case

---

## 6. Code

```python
# Phương pháp được khuyến nghị: HOG + SVM với NMS
from src.detector import FaceDetector

detector = FaceDetector(window_size=(64, 64), cell_size=8)
detector.load('models/face_detector.pkl')

# Detect với NMS
faces = detector.detect(image, 
                       scale_factor=1.2, 
                       min_neighbors=3,  # NMS threshold
                       confidence_threshold=0.5)
```

---

*Kết quả dựa trên test thực tế với ảnh nhóm người*
