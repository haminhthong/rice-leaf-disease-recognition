# Data Card - Rice Leaf Disease Detection Dataset

## 1. Cấu Trúc Thư Mục Dữ Liệu

Dữ liệu của hệ thống được quản lý tách biệt theo nguyên tắc không thay đổi dữ liệu gốc:

```text
data/
├── README.md            # Data Card (tài liệu này)
├── raw/                 # Lưu trữ file ZIP dữ liệu gốc (Không commit git)
├── extracted/           # Lưu trữ dữ liệu sau giải nén tạm (Không commit git)
├── processed/           # Tập dữ liệu YOLO đã xử lý, lọc trùng và chia split (Không commit git)
└── samples/             # Ảnh mẫu kích thước nhỏ phục vụ thử nghiệm smoke test
```

---

## 2. Nguồn Dữ Liệu & Giấy Phép (Data Provenance & License)

- **Dataset Nguồn 1**: `RiceLeafAnnotatedDataset.zip`
  - Nguồn: Bộ dữ liệu gán nhãn phát hiện triệu chứng bệnh lá lúa công khai.
  - License: CC BY 4.0 / Public Domain.
  - Mục đích: Cung cấp các ảnh tổn thương Bạc lá lúa và Đốm nâu.
- **Dataset Nguồn 2**: `dataset1.zip`
  - Nguồn: Bộ dữ liệu gán nhãn bổ sung với các điều kiện góc chụp và ánh sáng thực tế.
  - License: CC BY 4.0 / Public Domain.

---

## 3. Hợp Đồng Ngữ Nghĩa Nhãn (Annotation Semantics Data Contract)

> **Data Contract**:
> Mỗi Bounding Box biểu diễn **vùng triệu chứng bệnh quan sát được trên phiến lá (visible symptomatic region associated with target disease)** theo tọa độ chuẩn hóa $[x_{center}, y_{center}, w, h] \in [0, 1]$.

- **Đơn vị gán nhãn**: Vùng tổn thương đại diện (representative symptomatic region).
- **Quy ước Negative Samples (Ảnh khỏe mạnh / Không bệnh mục tiêu)**:
  Trong bài toán Object Detection, một lá không có triệu chứng mục tiêu hoặc lá khỏe mạnh được đại diện bằng một **tệp nhãn rỗng (0 bounding box)**. Không có class riêng cho lá khỏe mạnh; mô hình sẽ đưa ra trạng thái `no_detection` khi không tìm thấy vùng tổn thương vượt ngưỡng tin cậy.

---

## 4. Quy Trình Kiểm Toán & Làm Sạch Dữ Liệu (Audit & Cleaning Pipeline)

1. **Polygon to Bounding Envelope**: Chuyển đổi nhãn Polygon dạng chuỗi điểm thành Bounding Box chữ nhật nhỏ nhất bao quanh `[x_center, y_center, width, height]`, tự động xén viền (clipping) về $[0, 1]$ nếu tọa độ ngoài biên.
2. **Exact Deduplication (SHA-256)**: Loại bỏ các ảnh trùng tuyệt đối dựa trên giá trị băm nhị phân SHA-256.
3. **Annotation Conflict Quarantine**: Các ảnh trùng SHA-256 nhưng có nhãn gán khác nhau sẽ được **cách ly** vào `reports/data_conflicts/conflicts.csv` để kiểm duyệt thủ công, không đưa vào huấn luyện.
4. **Near-Duplicate Grouping (pHash + BK-Tree + Union-Find)**:
   - Gom các ảnh biến thể (augmentation/crop) vào cùng nhóm `group_id` với khoảng cách pHash Hamming $\le 2$.
   - Cấu trúc cây **BK-Tree** giúp thu hẹp không gian tìm kiếm các chuỗi pHash so với việc so sánh toàn bộ từng cặp ($O(N^2)$).
   - Tự động gom các ảnh có chung `original_key` (Roboflow parent key) vào cùng nhóm.
5. **Group-aware Stratified Split & Zero Leakage**:
   - Phân chia tập Train (70%), Val (15%), Test (15%) theo `group_id`.
   - Đảm bảo các ảnh cùng `group_id`, cùng `sha256` hoặc cùng `original_key` **tuyệt đối không bị phân cắt vào các tập dữ liệu khác nhau**.
6. **Split Diagnostics & Audit Report**:
   - Ghi nhận ma trận `Split x Source` (tỷ lệ ảnh từng nguồn qua các tập dữ liệu).
   - Thống kê phân bố diện tích tổn thương (Small $< 0.05$, Medium $0.05 - 0.2$, Large $> 0.2$).
   - Ghi nhận số lượng `bbox_count`, `polygon_count`, và `clipped_boxes`.
7. **Split Size Validation**: Đảm bảo số lượng nhóm tối thiểu (`train` $\ge 10$, `val` $\ge 5$, `test` $\ge 5$) và số lượng instance từng lớp tối thiểu trong `val` và `test` ($\ge 20$ instances/lớp).

---

## 5. Phân Bố Lớp & Nhãn Dữ Liệu (Taxonomy)

| Class ID | Tên Lớp Gốc | Tên Tiếng Việt | Mô Tả Tổn Thương Bệnh |
|---|---|---|---|
| 0 | `Bacterial_Leaf_Blight` | Bạc lá lúa | Vết sọc mọng nước dọc mép lá, màu vàng nhạt đến xám trắng do vi khuẩn *Xanthomonas oryzae* |
| 1 | `Brown_Spot` | Đốm nâu | Các đốm hình tròn hoặc elip màu nâu thẫm viền vàng nhạt trên phiến lá do nấm *Bipolaris oryzae* |
