# Model Card - Rice Leaf Disease Detection Platform

## 1. Phạm Vi Bài Toán & Định Vị (Scope & Positioning)

Mô hình là **công cụ hỗ trợ trinh sát thực địa và sàng lọc ban đầu (Field Scouting Decision Support)** thực hiện nhiệm vụ **Object Detection / Symptom Localization** nhằm phát hiện và khoanh vùng tổn thương trên ảnh lá lúa đơn lẻ bằng kiến trúc YOLOv8:
- **Bacterial Leaf Blight (Bạc lá lúa)** - Vi khuẩn *Xanthomonas oryzae*
- **Brown Spot (Đốm nâu)** - Nấm *Bipolaris oryzae*

> [!IMPORTANT]
> - Mô hình **KHÔNG** chẩn đoán tự động toàn bộ bệnh trên cây lúa.
> - Kết quả **KHÔNG** khẳng định lá khỏe mạnh (các tổn thương có thể thuộc bệnh ngoài phạm vi hỗ trợ).
> - Mô hình **TUYỆT ĐỐI KHÔNG** đưa ra khuyến cáo hoặc kê đơn thuốc/hóa chất bảo vệ thực vật tự động. Mọi quyết định xử lý thực địa phải có ý kiến trực tiếp của chuyên gia/kỹ sư nông nghiệp.

---

## 2. Hợp Đồng Dữ Liệu Ngữ Nghĩa Nhãn (Data Contract)

- **Annotation Unit**: Vùng tổn thương đại diện quan sát được trên phiến lá (`visible symptomatic region associated with target disease`).
- **Negative Samples**: Ảnh không chứa nhãn (0 detections) biểu thị lá không có triệu chứng mục tiêu hoặc lá khỏe mạnh; không có class riêng biệt cho lá khỏe.
- **Tọa độ chuẩn hóa**: $x_{center}, y_{center}, w, h \in [0, 1]$. Các nhãn dạng polygon từ dữ liệu nguồn được chuyển đổi thành bounding envelope bao quanh.

---

## 3. Đối Tượng & Bối Cảnh Sử Dụng Phù Hợp

### **Người dùng phù hợp**
- Sinh viên, nghiên cứu sinh AI / Computer Vision.
- Kỹ sư nông nghiệp thử nghiệm ứng dụng trinh sát và khoanh vùng triệu chứng bệnh hại.
- Nhà phát triển tích hợp công cụ hỗ trợ thị giác máy tính vào quy trình kiểm tra thực địa.

### **Bối cảnh hoạt động phù hợp**
- Ảnh chụp cận cảnh từng lá lúa đơn lẻ hoặc cụm lá tương đối rõ ràng.
- Đủ ánh sáng tự nhiên, góc chụp vuông góc hoặc hơi nghiêng.

### **Bối cảnh KHÔNG phù hợp**
- Ảnh toàn cánh đồng hoặc ảnh chụp từ máy bay không người lái (drone/UAV).
- Ảnh chụp trong điều kiện ánh sáng cực kém, chói sáng mạnh hoặc ảnh nhiều lá chồng lấn phức tạp chưa qua kiểm chứng.
- Suy luận các loại bệnh nông nghiệp ngoài 2 lớp đã định nghĩa.

---

## 4. Kiến Trúc & Cấu Hình Triển Khai

| Trường | Giá trị |
|---|---|
| Kiến trúc mô hình | YOLOv8n (Baseline) / YOLOv8s (Candidate) |
| Lineage & Checksum | Đã khóa bằng `manifest.csv` SHA-256 và `audit_report.json` SHA-256 |
| Random Seed | 42 (Cố định trong toàn bộ pipeline) |
| Ngưỡng tin cậy suy luận | Detection Score $\ge 0.25$, NMS IoU $= 0.45$ |
| Tầng quyết định (Decision Layer) | `detections` kèm `image_summary` và cờ `requires_human_review` |
| Định dạng phục vụ | PyTorch (`.pt`), ONNX (`.onnx`), OpenVINO, TorchScript |

---

## 5. Protocol Lựa Chọn & Khóa Đánh Giá

1. **Huấn luyện chuẩn tắc**: Huấn luyện đồng thời `YOLOv8n` và `YOLOv8s` trên cùng tập Train đã lọc trùng bằng SHA-256 và gom nhóm pHash BK-Tree + Union-Find.
2. **Xếp hạng Champion duy nhất bằng Validation mAP50-95**: Áp dụng các rào cản an toàn (Recall từng lớp $\ge 0.75$, latency p95 $\le 300\text{ ms}$).
3. **Phân tích lỗi đa chiều**: Đo lường theo Error Taxonomy (Missed lesion, Background FP, Localization error, Classification confusion, Duplicate detection) và phân tầng kích thước tổn thương (Small $< 0.05$, Medium $0.05 - 0.2$, Large $> 0.2$).
4. **Khóa tập Test (`--confirm-final-test`)**: Tập Test chỉ mở khóa duy nhất 1 lần khi chốt mô hình; không sử dụng kết quả test để tinh chỉnh siêu tham số.
5. **Quality Gate xuất mô hình**: Kiểm định tương đương suy luận (Prediction Parity) giữa PyTorch và ONNX trên tập ảnh mẫu trước khi đóng gói triển khai.

---

## 6. Rủi Ro & Giới Hạn (Risks & Limitations)

- **Domain Shift**: Sai biệt giữa ảnh công khai sạch và ảnh chụp thực tế ngoài đồng ruộng (ánh sáng, bùn đất, tạp chất).
- **Class Imbalance & Small Lesions**: Tổn thương đốm nhỏ giai đoạn đầu dễ bị nhầm lẫn với bụi bẩn hoặc nếp gập lá tự nhiên.
- **Uncalibrated Detection Score**: Điểm số YOLO thể hiện độ khớp đặc trưng hình ảnh, không phải xác suất bệnh lý tuyệt đối. Hệ thống sử dụng cờ `requires_human_review` đối với các trường hợp ranh giới hoặc chồng lấn triệu chứng khác lớp.
