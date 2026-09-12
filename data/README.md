# Data Card - Rice Leaf Disease Detection Dataset

## 1. Cấu Trúc Thư Mục Dữ Liệu

Dữ liệu được tổ chức tách biệt nhằm đảm bảo tính nguyên vẹn:

```text
data/
├── README.md            # Tài liệu mô tả dữ liệu (Data Card)
├── raw/                 # Lưu trữ file nén dữ liệu gốc (Không commit Git)
├── extracted/           # Thư mục giải nén tạm thời (Không commit Git)
├── processed/           # Dữ liệu sạch chuẩn YOLO đã lọc trùng và chia split (Không commit Git)
│   ├── train/           # images/ và labels/
│   ├── val/             # images/ và labels/
│   ├── test/            # images/ và labels/
│   ├── data.yaml        # Cấu hình dataset cho YOLOv8
│   ├── manifest.csv     # Bảng theo dõi metadata từng ảnh và split
│   └── data_report.json # Báo cáo thống kê làm sạch dữ liệu
└── sample/              # Một số ảnh mẫu thử nghiệm nhanh (Commit trong Git)
```

---

## 2. Nguồn Dữ Liệu (Data Sources)

Dữ liệu được tổng hợp từ các bộ ảnh gán nhãn thực địa công khai:
- `RiceLeafAnnotatedDataset.zip`
- `dataset1.zip`

---

## 3. Quy Ước Nhãn & Chất Lượng Dữ Liệu

### 3.1. Bounding Box Format
- Mỗi Bounding Box khoanh vùng một **triệu chứng bệnh quan sát được trên phiến lá** theo định dạng YOLO chuẩn hóa:
  `class_id x_center y_center width height` với các tọa độ thuộc $[0, 1]$.
- Các nhãn dạng Polygon từ dữ liệu nguồn được tự động chuyển đổi thành Bounding Box chữ nhật bao quanh (Bounding Envelope).

### 3.2. Phân Loại Trạng Thái Ảnh
- **valid**: Ảnh chứa ít nhất một bounding box thuộc 2 lớp bệnh mục tiêu.
- **negative**: Ảnh lá không chứa bệnh mục tiêu (ảnh nền/ảnh lá lành).
- **invalid**: Ảnh thiếu file nhãn hoặc tọa độ nhãn lỗi -> **bị loại bỏ (quarantined)**. Tuyệt đối không để ảnh lỗi/thiếu nhãn biến thành negative âm thầm, tránh đưa ảnh bệnh chưa gán nhãn vào tập làm ảnh nền.

---

## 4. Quy Trình Làm Sạch & Chia Tập Chống Rò Rỉ (Data Cleaning & Leakage-Aware Split)

1. **Lọc trùng tuyệt đối (SHA-256)**: Loại bỏ các ảnh trùng lặp hoàn toàn dựa trên mã băm SHA-256. Nếu 2 ảnh cùng nội dung nhưng nhãn gán mâu thuẫn, cả 2 sẽ được cách ly.
2. **Gom nhóm biến thể (original_key + pHash)**:
   - Các ảnh sinh ra từ cùng một ảnh gốc qua augmentation (crop, rotate, resize) có chung `original_key`.
   - Các ảnh gần giống nhau có khoảng cách Hamming perceptual hash (pHash) $\le 2$ được gom vào cùng nhóm `group_id`.
3. **Phân chia Train/Val/Test theo nhóm (Group-aware Split)**:
   - Tỷ lệ: Train 70%, Validation 15%, Test 15%.
   - **Bất biến quan trọng**: Toàn bộ ảnh trong cùng một `group_id` luôn thuộc về cùng một phân tập (Train, Val hoặc Test). Tuyệt đối không để ảnh gốc ở tập Train trong khi biến thể của nó rơi vào tập Test.

---

## 5. Lớp Bệnh Mục Tiêu (Target Classes)

| Class ID | Tên Lớp | Tên Tiếng Việt | Mô Tả Tổn Thương |
|---|---|---|---|
| `0` | `Bacterial_Leaf_Blight` | Bạc lá lúa | Vệt sọc vàng nhạt đến trắng xám dọc mép lá do vi khuẩn *Xanthomonas oryzae* |
| `1` | `Brown_Spot` | Đốm nâu | Đốm tròn/elip màu nâu thẫm viền vàng nhạt trên phiến lá do nấm *Bipolaris oryzae* |
