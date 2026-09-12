# Model Card - Rice Leaf Disease Detection

## 1. Task & Problem Formulation
- **Nhiệm vụ**: Object Detection / Symptom Localization (Phát hiện và định vị vùng tổn thương bệnh trên phiến lá lúa).
- **Mục tiêu**: Xác định vị trí bounding box và phân loại vùng triệu chứng, phục vụ công tác trinh sát thực địa ban đầu (Field Scouting Decision Support).
- **Không thực hiện**: Không làm bài toán Image Classification toàn ảnh (một ảnh lá có thể có nhiều tổn thương hoặc nhiều bệnh cùng lúc).

---

## 2. Classes (Lớp bệnh mục tiêu)
Mô hình tập trung vào 2 bệnh phổ biến và gây hại nghiêm trọng:

| Class ID | Tên chuẩn | Tên tiếng Việt | Tác nhân gây bệnh |
|---|---|---|---|
| `0` | `Bacterial_Leaf_Blight` | Bạc lá lúa | Vi khuẩn *Xanthomonas oryzae* |
| `1` | `Brown_Spot` | Đốm nâu | Nấm *Bipolaris oryzae* |

---

## 3. Dataset & Data Quality
- **Nguồn dữ liệu**: Dữ liệu ảnh gán nhãn thực tế từ các bộ dữ liệu lá lúa công khai (`RiceLeafAnnotatedDataset`, `dataset1`).
- **Làm sạch & Chống rò rỉ (Zero Data Leakage)**:
  - Lọc trùng lặp tuyệt đối bằng mã băm SHA-256 (loại bỏ trường hợp ảnh giống hệt nhau ở cả Train và Test).
  - Gom nhóm các biến thể augmentation / crop cùng ảnh gốc bằng `original_key` và perceptual hash (pHash).
  - Phân chia Train (70%), Validation (15%), Test (15%) theo nhóm (Group-aware Split), đảm bảo các biến thể không bao giờ bị phân tách chéo giữa các tập.
  - Phân biệt rõ ràng giữa ảnh negative hợp lệ (không chứa bệnh) và ảnh thiếu nhãn/lỗi nhãn (bị loại bỏ, không tự động xem là negative).

---

## 4. Model Architecture & Training
- **Kiến trúc**: YOLOv8s (Ultralytics).
- **Kích thước ảnh đầu vào**: 640 × 640 pixels.
- **Kỹ thuật tối ưu**:
  - Optimizer: AdamW, Learning Rate 0.001, Weight Decay 0.0005.
  - Early Stopping: Patience 25 epochs.
  - Train-only Augmentation: HSV color space, rotation, translation, scaling, horizontal flip, mosaic.
  - Validation/Test: Tiền xử lý tất định, không áp dụng augmentation.

---

## 5. Metrics & Evaluation
Mô hình được đánh giá trên các độ đo chuẩn trong Object Detection:
- **Precision (P)**: Độ chính xác của các bounding box được dự đoán.
- **Recall (R)**: Tỷ lệ phát hiện được các tổn thương thực tế.
- **mAP@0.5**: Mean Average Precision tại ngưỡng IoU 0.5.
- **mAP@0.5:0.95**: Mean Average Precision trên dải IoU từ 0.50 đến 0.95 (bước 0.05).
- **Per-class AP**: Đánh giá chi tiết riêng biệt cho từng loại bệnh.

---

## 6. Intended Use (Mục đích & Bối cảnh sử dụng)
### Phù hợp:
- Ứng dụng hỗ trợ trinh sát thực địa cho kỹ sư nông nghiệp, khuyến nông viên, sinh viên và nhà nghiên cứu.
- Ảnh chụp cận cảnh từng phiến lá hoặc cụm lá lúa trong điều kiện ánh sáng tự nhiên rõ nét.

### Không phù hợp:
- Ảnh chụp toàn cảnh cánh đồng từ xa hoặc ảnh chụp từ máy bay không người lái (drone/UAV).
- Ảnh chụp thiếu sáng nghiêm trọng, rung mờ hoặc cháy sáng mạnh.
- Không dùng để chẩn đoán các bệnh ngoài 2 lớp mục tiêu đã nêu.

---

## 7. Limitations & Ethical Considerations (Giới hạn & Rủi ro)
- **Domain Shift**: Mô hình có thể giảm độ chính xác khi gặp điều kiện thời tiết, giống lúa hoặc môi trường canh tác khác biệt so với dữ liệu huấn luyện.
- **Tổn thương nhỏ (Small Lesions)**: Các đốm bệnh rất nhỏ ở giai đoạn đầu có thể dễ bị nhầm lẫn với bụi bẩn hoặc nếp gập tự nhiên của lá.
- **Detection Score $\neq$ Xác suất bệnh lý**: Điểm tin cậy (confidence score) thể hiện mức độ tương đồng đặc trưng thị giác, không phải tỷ lệ xác suất sinh học tuyệt đối.
- **Miễn trừ trách nhiệm**: Kết quả chỉ mang tính hỗ trợ tham khảo. Tuyệt đối không tự ý phun thuốc bảo vệ thực vật hoặc hóa chất khi chưa có chỉ dẫn của chuyên gia nông nghiệp có chuyên môn.
