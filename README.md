# 🌾 Rice Leaf Disease Detection Platform

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8.3.220-green?logo=ultralytics)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **A leakage-aware computer-vision pipeline for detecting and localizing bacterial leaf blight and brown spot symptoms, with multi-source annotation harmonization, near-duplicate grouping, validation-only model selection, per-class error analysis, and production-oriented serving.**

---

## 1. Bài Toán & Các Lớp Bệnh Mục Tiêu (Problem & Supported Diseases)

Dự án tập trung vào bài toán **Object Detection / Symptom Localization** nhằm phát hiện và khoanh vùng tổn thương triệu chứng bệnh trên ảnh chụp cận cảnh từng lá lúa đơn lẻ:

- 🌾 **Bacterial Leaf Blight (Bạc lá lúa)** - Mã lớp `0`: Tổn thương sọc mọng nước vàng xám do vi khuẩn *Xanthomonas oryzae* gây ra.
- 🍂 **Brown Spot (Đốm nâu)** - Mã lớp `1`: Tổn thương dạng đốm tròn hoặc elip màu nâu thẫm viền vàng do nấm *Bipolaris oryzae* gây ra.

> [!IMPORTANT]
> **Tuyên Bố Phạm Vi & Miễn Trừ Trách Nhiệm Nông Nghiệp**:
> - Hệ thống là **công cụ hỗ trợ trinh sát thực địa và sàng lọc ban đầu (Field Scouting Decision Support)**.
> - Hệ thống **KHÔNG chẩn đoán tự động toàn bộ bệnh trên cây lúa**, **KHÔNG tự động khẳng định lá cây hoàn toàn khỏe mạnh** và **TUYỆT ĐỐI KHÔNG tự động kê đơn hay khuyến cáo liều lượng thuốc BVTV/hóa chất**.
> - Mọi kết quả cần được cán bộ hoặc chuyên gia bảo vệ thực vật thẩm định trực tiếp trước khi đưa ra quyết định canh tác.

---

## 2. Đối Tượng Phát Hiện: Triệu Chứng Bệnh Thay Vì Phân Loại Ảnh

Hệ thống hoạt động theo cơ chế **định vị triệu chứng (Symptom Localization)**, hoàn toàn khác biệt với phân loại ảnh toàn thể (Whole-image Classification):
- Mỗi hình ảnh có thể chứa **0, 1 hoặc nhiều vùng tổn thương** của cùng một hoặc cả hai loại bệnh.
- **Không có lớp "Healthy"**: Trong bài toán Object Detection, một lá khỏe mạnh hoặc không có triệu chứng mục tiêu được định nghĩa chuẩn tắc là **Negative Sample (ảnh không chứa Bounding Box)**. Trạng thái `NO_SUPPORTED_SYMPTOM_DETECTED` chỉ nghĩa là chưa tìm thấy triệu chứng thuộc phạm vi hỗ trợ, không khẳng định lá khỏe.

---

## 3. Hợp Đồng Dữ Liệu Ngữ Nghĩa Nhãn (Annotation Semantics Data Contract)

Để đảm bảo tính nhất quán giữa các nguồn dữ liệu công khai, pipeline áp dụng hợp đồng dữ liệu chuẩn:

> **Data Contract**:
> Mỗi Bounding Box biểu diễn **vùng triệu chứng bệnh quan sát thấy trên phiến lá (visible symptomatic region associated with target disease)**, với tọa độ chuẩn hóa $[x_{center}, y_{center}, w, h] \in [0, 1]$.

- **Đơn vị gán nhãn**: Vùng tổn thương đại diện (representative lesion / symptomatic envelope).
- **Tránh nhầm lẫn ngữ nghĩa**: Không gán nhãn toàn bộ lá thành 1 box nếu lá chỉ có đốm rải rác.

---

## 4. Kiến Trúc Pipeline Canonical & Bộ Điều Phối (Pipeline Orchestration)

> 📘 **Tài liệu kiến trúc chuyên sâu**: Xem phân tích chi tiết mọi khía cạnh tại [docs/DATA_AND_LOGIC_FLOW.md](docs/DATA_AND_LOGIC_FLOW.md) (bao gồm sơ đồ Mermaid về Data Flow, Logic Flow, Data Contracts giữa các tầng và cây quyết định hỗ trợ trinh sát thực địa).

Toàn bộ quy trình từ dữ liệu thô đến triển khai phục vụ được tự động hóa bằng **Pipeline Orchestrator** trung tâm (`rice-pipeline` hoặc `python scripts/run_pipeline.py`):

```text
1. MULTI-SOURCE DATA INGESTION
   Roboflow / Kaggle / YOLO datasets
        ↓
2. TAXONOMY & ANNOTATION NORMALIZATION
   Read source data.yaml ──► Map aliases ──► BLB / Brown Spot ──► Polygon → Bounding Box
        ↓
3. DATA QUALITY & LEAKAGE CONTROL
   Image validation ──► Label validation ──► SHA-256 exact dedup ──► pHash near-duplicate ──► BK-Tree + Union-Find
        ↓
4. GROUP-AWARE DATA SPLIT
   Image groups ──► Train 70% / Val 15% / Test 15% ──► Zero group leakage ──► Source & size distribution audit
        ↓
5. MODEL DEVELOPMENT
   YOLOv8s @ 640 ──► Validation metrics ──► Decision policy ──► Locked test ──► Versioned release
        ↓
6. ERROR & ROBUSTNESS ANALYSIS
   Per-class AP / Recall ──► FP / FN / Low IoU / Confusion ──► Lesion size slices (S/M/L) ──► Negative false alarm
        ↓
7. LOCKED FINAL TEST
   Open test once (--confirm-final-test) ──► mAP50-95, mAP50, Recall, Per-class metrics
        ↓
8. SERVING & DECISION LAYER
   Rice Leaf Image ──► Input Validation ──► Shared RiceLeafDetector ──► Detection Score + Image Summary ──► Human Review Flag ──► FastAPI / Streamlit / ONNX
```

### Các Lệnh Điều Phối Pipeline Chính:
```bash
# Xem trước DAG các giai đoạn và kiểm tra điều kiện tiên quyết (Dry Run)
python scripts/run_pipeline.py --dry-run

# Chạy riêng công đoạn chuẩn hóa và chia dữ liệu sạch
python scripts/run_pipeline.py --stage data

# Chạy huấn luyện baseline (ví dụ 10 epochs)
python scripts/run_pipeline.py --stage train --epochs 10

# Chạy tự động hóa toàn bộ Pipeline (End-to-End Run)
python scripts/run_pipeline.py --stage all
```

### Kiến Trúc Suy Luận Trực Tuyến (Online Serving Architecture)

```text
Rice Leaf Image
      │
      ▼
Input Validation (Magic bytes, resolution, pixel limit)
      │
      ▼
YOLOv8 Detector (RiceLeafDetector)
      │
      ▼
Disease Bounding Boxes (Coordinates + Class ID + Detection Score)
      │
      ▼
Decision & Image-Level Summary (Presence flags + Human Review Flag)
      │
      ▼
FastAPI REST API / Streamlit Dashboard / ONNX Runtime
```

---

## 5. Nguồn Dữ Liệu & Data Card (Data Provenance)

- **Source 1 (`RiceLeafAnnotatedDataset.zip`)**: Bộ dữ liệu ảnh lá lúa công khai gán nhãn đa dạng theo định dạng YOLO.
- **Source 2 (`dataset1.zip`)**: Bộ dữ liệu bổ sung với nhiều góc chụp và độ chiếu sáng khác nhau.
- Chi tiết về phân bố nhãn, provenance và kiểm toán dữ liệu được lưu tại [data/README.md](data/README.md). License chỉ ghi `unknown/pending verification` nếu chưa có bằng chứng theo từng nguồn.

---

## 6. Chuẩn Hóa Taxonomy & Chuyển Đổi Polygon → BBox

1. **Chuẩn hóa Alias tên lớp**: Tự động ánh xạ các biến thể như `bacterial leaf blight`, `bacterial leafblight` $\rightarrow$ `Bacterial_Leaf_Blight`; `brown spot`, `Brown-Spot` $\rightarrow$ `Brown_Spot`.
2. **Chuyển đổi Polygon $\rightarrow$ Bounding Envelope**: Tính toán hộp chữ nhật bao nhỏ nhất `[x_center, y_center, width, height]` từ chuỗi tọa độ đa giác, đồng thời xén viền (clipping) về $[0, 1]$ nếu tọa độ vượt khung ảnh do làm tròn.
3. **Báo cáo kiểm toán chuyển đổi**: Ghi nhận số lượng `bbox_count`, `polygon_count`, và `clipped_boxes` vào `audit_report.json`.

---

## 7. Kiểm Soát Chất Lượng & Chống Rò Rỉ Dữ Liệu (Leakage Control)

1. **Lọc trùng SHA-256 tuyệt đối**: Loại bỏ ảnh trùng lặp nhị phân chính xác.
2. **Cách ly xung đột nhãn (Annotation Conflict)**: Nếu 2 ảnh trùng SHA-256 nhưng nhãn gán khác nhau, pipeline tự động **cách ly vào `reports/data_conflicts/conflicts.csv`** để chuyên gia kiểm duyệt, không tự ý chọn ngẫu nhiên.
3. **Gom nhóm biến thể pHash (BK-Tree + Union-Find)**:
   - Các ảnh công khai thường chứa biến thể crop/zoom/flip từ cùng một ảnh gốc.
   - Trích xuất perceptual hash 64-bit (`pHash`).
   - Cấu trúc cây **BK-Tree** (Burkhard-Keller Tree) giúp thu hẹp không gian tìm kiếm các chuỗi pHash theo khoảng cách Hamming so với so sánh toàn bộ từng cặp ($O(N^2)$), kết hợp **Union-Find** gom tất cả biến thể vào một `group_id`.
   - Gom đồng thời biến thể có chung `original_key` (Roboflow augmentation parent key).

---

## 8. Group-Aware Split & Báo Cáo Phân Bố (Stratification Diagnostics)

- **Tỷ lệ phân chia**: Train 70%, Val 15%, Test 15% theo từng nhóm độc lập `group_id`.
- **Zero Group Leakage Enforcement**: Mọi ảnh cùng nhóm `group_id`, cùng `sha256` hoặc cùng `original_key` tuyệt đối không bao giờ xuất hiện ở hai tập dữ liệu khác nhau. Hàm `validate_dataset()` kiểm tra nghiêm ngặt điều kiện này trước khi xuất dữ liệu.
- **Source-Aware Diagnostics**: Báo cáo kiểm toán `audit_report.json` lưu vết ma trận `Split x Source` và phân bố số lượng tổn thương theo nguồn dữ liệu nhằm phát hiện sớm nguy cơ mô hình học theo đặc thù nguồn chụp (source-specific bias).

---

## 9. Chính Sách Augmentation & Đặc Thù Nông Học (Augmentation Policy)

- **Nguyên tắc bất biến**: Augmentation chỉ áp dụng trên tập Train; tập Validation và Test chỉ áp dụng tiền xử lý xác định (deterministic letterbox/resize).
- **Cẩn trọng với màu sắc (HSV)**: Chẩn đoán bệnh cây trồng phụ thuộc chặt chẽ vào sắc độ tổn thương (vàng úa, chlorosis, hoại tử nâu). Việc tăng cường màu sắc (Hue/Saturation) quá mức có thể làm sai lệch tín hiệu bệnh học thực tế. Pipeline giới hạn biên độ HSV nhẹ nhàng.
- **Cẩn trọng với hình học**: Cho phép lật ngang (Horizontal Flip), hạn chế lật dọc (Vertical Flip) thiếu tự nhiên và kiểm soát Mosaic để không tạo ngữ cảnh lá gãy khúc phi thực tế.

---

## 10. Mô Hình Canonical & Chính Sách Quyết Định

Đường chạy production chỉ dùng `YOLOv8s @ 640` với hai lớp mục tiêu. `YOLOv8n` chỉ là baseline thí nghiệm riêng, không được tự động chọn làm model phục vụ.

- **Metric chính**: Validation `mAP50-95`.
- **Metric theo use case**: Image-Level Target Recall, recall từng lớp, false-alarm rate của `TRUE_NEGATIVE` và `OUT_OF_SCOPE_NEGATIVE`.
- **Decision policy**: `review_threshold` và `accept_threshold` nằm trong YAML, phải được chọn trên Validation rồi đóng gói cùng model. Test không được dùng để chỉnh ngưỡng.

---

## 11. Bảng Đánh Đổi Pareto (Pareto Trade-off Table)

| Kiến Trúc | Số Tham Số | Kích Thước File | Val mAP50-95 | Val Recall | p95 Latency (CPU) | Trường Hợp Triển Khai |
|---|---|---|---|---|---|---|
| **YOLOv8s** (Canonical) | ~11.2M | ~22.5 MB | *Đo đạc thực tế* | *Đo đạc thực tế* | *Đo đạc thực tế* | Server Cloud, trinh sát tự động |

> [!NOTE]
> Bảng trên thể hiện khung đo đạc chính thức. Số liệu mAP sẽ được ghi nhận sau khi huấn luyện hoàn tất trên toàn bộ tập dữ liệu thực nghiệm đã kiểm toán.

---

## 12. Bộ Tiêu Trí Đánh Giá (Detection Metrics)

- **mAP50-95**: Diện tích dưới đường cong PR trung bình tại các ngưỡng IoU từ 0.50 đến 0.95 (bước 0.05). Đây là chỉ số trọng tâm đánh giá khả năng định vị chính xác.
- **mAP50**: mAP tại IoU 0.50.
- **Per-class AP & Recall**: Bắt buộc báo cáo chi tiết cho từng lớp `Bacterial_Leaf_Blight` và `Brown_Spot`.
- **Image-level Disease Recall**:
  $$\text{ImageRecall} = \frac{\text{Số ảnh có bệnh chứa ít nhất 1 hộp phát hiện đúng}}{\text{Tổng số ảnh có bệnh thật}}$$
  Chỉ số này phản ánh sát nhất nghiệp vụ trinh sát đồng ruộng ban đầu.

---

## 13. Phân Loại Lỗi & Phân Tầng Kích Thước Tổn Thương (Error Slicing)

Module `error_analysis.py` phân tích chi tiết theo cấu trúc:

```text
Detection Errors
├── FN: Bỏ sót tổn thương thật (Missed lesion)
├── FP: Báo nhầm nền / lá khỏe (Background False Positive)
├── Localization: Dự đoán đúng lớp nhưng IoU thấp [0.1, 0.5)
├── Classification Confusion: Nhầm giữa 2 lớp bệnh (BLB ↔ Brown Spot)
└── Duplicate Detection: Nhiều box dự đoán đè lên cùng 1 nhãn thật
```

### Phân Tầng Theo Kích Thước Tổn Thương (Lesion Size Slices)
- **Small Lesions** (Diện tích box $< 5\%$ diện tích ảnh): Đốm nâu giai đoạn đầu.
- **Medium Lesions** ($5\% \le \text{diện tích} \le 20\%$): Tổn thương đốm trung bình.
- **Large Lesions** (Diện tích $> 20\%$): Vệt bạc lá lan rộng dọc phiến lá.
- Báo cáo Recall riêng cho từng nhóm kích thước giúp tránh trường hợp tổng mAP cao nhưng bỏ sót toàn bộ tổn thương nhỏ.

---

## 14. Giao Thức Khóa Tập Test (Locked Final Test Protocol)

- **Không Peeking**: Tập Test hoàn toàn bị cô lập trong quá trình huấn luyện, tinh chỉnh siêu tham số và phân tích lỗi (Val-only error tuning).
- **Khóa kỹ thuật**: Lệnh `rice-evaluate --split test` bắt buộc cờ `--confirm-final-test`. Sau lần đầu, `final_test_report.json` tồn tại sẽ chặn chạy lại; chỉ `--force-reopen-test` mới mở lại và đánh dấu `test_compromised=true`.

---

## 15. Benchmark Ảnh Negative & Điểm Tin Cậy (Confidence Policy)

- **Benchmark ảnh negative**: Báo cáo riêng tỷ lệ báo động giả trên `TRUE_NEGATIVE` và `OUT_OF_SCOPE_NEGATIVE`, không gộp hai loại thành một negative mơ hồ.
- **Chính sách điểm số (Detection Score)**: YOLO confidence score thể hiện điểm số khớp đặc trưng hình ảnh của mô hình, không phải xác suất bệnh lý tuyệt đối đã cân bằng (uncalibrated score).
- **Cờ thẩm định trực quan (Human Review Flag)**:
  - Tự động kích hoạt khi xuất hiện candidate trong vùng `[review_threshold, accept_threshold)` lấy từ cấu hình.
  - Tự động kích hoạt khi có các hộp phát hiện đè nhau thuộc hai lớp bệnh khác nhau.

---

## 16. Phục Vụ Suy Luận: FastAPI & Streamlit Dashboard

### 1. RESTful Web API (FastAPI)
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```
- Endpoint `POST /predict`: Trả về kết quả phát hiện chi tiết (`detections`) kèm khối tóm tắt quyết định (`image_summary`) và cảnh báo chuyên môn.
- Endpoints `GET /health/live`, `GET /health/ready`, `GET /info`.

### 2. Streamlit Dashboard Tương Tác
```bash
streamlit run app/dashboard.py
```
- Giao diện trực quan cho phép upload ảnh, điều chỉnh ngưỡng tin cậy, xem hộp bounding box, thống kê `Image Summary` và thông báo khi cần thẩm định thủ công.

---

## 17. Xuất Mô Hình & Đảm Bảo Tính Tương Đương (Export Parity)

Mô hình được hỗ trợ xuất sang **ONNX**, **OpenVINO** và **TorchScript** qua lệnh `rice-export`.

Module `export.py` tích hợp hàm kiểm định chất lượng **Prediction Parity Quality Gate**:
- So sánh kết quả suy luận giữa mô hình PyTorch gốc và bản xuất ONNX trên tập ảnh kiểm thử.
- Điều kiện đạt: Trùng khớp toàn bộ Class IDs, sai khác điểm tin cậy $\le 0.05$ và Bounding Box IoU $\ge 0.85$.

---

## 18. Kiểm Thử Hệ Thống (Software Correctness) vs Bằng Chứng Mô Hình

Repository tách bạch hoàn toàn giữa việc kiểm thử tính đúng đắn của phần mềm và đánh giá khả năng tổng quát hóa của mô hình:

- **Software Correctness Suite (41 Automated Tests)**:
  - `test_exact_duplicate_removed`: Xác minh lọc trùng SHA-256.
  - `test_near_duplicate_group_not_split`: Xác minh gom nhóm pHash BK-Tree.
  - `test_same_original_key_not_cross_split`: Đảm bảo biến thể augmentation cùng parent key không rò rỉ split.
  - `test_polygon_to_bbox_bounds`: Kiểm tra tọa độ BBox bao quanh nằm trong $[0, 1]$.
  - `test_invalid_annotation_rejected`: Từ chối tọa độ lỗi, NaN, sai định dạng.
  - `test_negative_image_kept`: Bảo tồn ảnh negative không chứa nhãn.
  - `test_taxonomy_alias_mapping`: Ánh xạ chính xác các alias nhãn nguồn.
  - `test_manifest_group_integrity`: Kiểm toán không rò rỉ group hay SHA qua các tập.
  - `test_test_split_not_used_for_selection`: Tập test hoàn toàn bị khóa khỏi bước xếp hạng Champion.
  - `test_export_prediction_parity`: Xác minh tính tương đương suy luận giữa PyTorch và ONNX.

---

## 19. Hướng Dẫn Nhanh (Quick Start)

### 1. Cài Đặt Môi Trường
```bash
git clone <repository-url>
cd rice-leaf-disease-recognition

# Cài đặt package ở chế độ Editable
python -m pip install --upgrade pip
pip install -e ".[app,dev]"
```

### 2. Tạo Dữ Liệu Demo Smoke Test & Chuẩn Bị Dữ Liệu
```bash
python scripts/create_demo_assets.py
rice-prepare --overwrite
```

### 3. Chạy Toàn Bộ Test Suite & Linter
```bash
# Kiểm tra định dạng & linter
ruff format --check src app scripts tests
ruff check src app scripts tests

# Chạy toàn bộ test suite (47+ tests)
pytest -v
```

### 4. Chạy Toàn Bộ Pipeline & Khởi Chạy Web App
```bash
# Thực thi canonical: Chuẩn hóa -> Train YOLOv8s -> Val -> Error Analysis -> Test khóa -> Export
python scripts/run_pipeline.py --stage all --epochs 10

# Khởi chạy REST API
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Hoặc khởi chạy Web Dashboard
streamlit run app/dashboard.py
```

---

## 20. Lộ Trình Phát Triển (Prioritized Roadmap)

| Mức Độ | Nhiệm Vụ | Trạng Thái |
|---|---|---|
| 🔴 **P0** | Chuẩn hóa định vị: Detection/Localization thay cho classification | ✅ Hoàn thành |
| 🔴 **P0** | Hợp đồng dữ liệu ngữ nghĩa BBox & Giải thích không có class Healthy | ✅ Hoàn thành |
| 🔴 **P0** | Tinh chỉnh thuật ngữ sang production-oriented portfolio pipeline | ✅ Hoàn thành |
| 🔴 **P0** | Tách bạch software test suite khỏi bằng chứng mô hình | ✅ Hoàn thành |
| 🟠 **P1** | Ràng buộc kỹ thuật chống rò rỉ `original_key` qua các split | ✅ Hoàn thành |
| 🟠 **P1** | Báo cáo kiểm toán phân bố `Split x Source` & phân nhóm diện tích tổn thương | ✅ Hoàn thành |
| 🟠 **P1** | Module phân tích lỗi đa chiều (Error Taxonomy) & Lesion Size Slicing | ✅ Hoàn thành |
| 🟠 **P1** | Tách tầng Detection và Image Summary kèm cờ Human Review | ✅ Hoàn thành |
| 🟠 **P1** | Quality Gate kiểm định tương đương suy luận PyTorch vs ONNX | ✅ Hoàn thành |
| 🟡 **P2** | Thí nghiệm Domain Generalization: Source-holdout cross-dataset | 📋 Dự kiến |
| 🟡 **P2** | Thu thập tập kiểm thử ảnh chụp điện thoại thực tế tại đồng ruộng | 📋 Dự kiến |
| 🟡 **P2** | Chuyển đổi sang YOLOv8-seg (Segmentation) để ước lượng % diện tích tổn thương | 📋 Dự kiến |
| 🟡 **P3** | Tối ưu hóa triển khai Edge (NVIDIA Jetson, OpenVINO Raspberry Pi) | 📋 Dự kiến |
| 🟡 **P3** | Theo dõi độ trôi dạt dữ liệu thực địa (Field Drift Monitoring) | 📋 Dự kiến |

---

## 📄 Giấy Phép (License)

Mã nguồn được phát hành theo [MIT License](LICENSE).
