# 🌾 Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8.3.220-green?logo=ultralytics)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Hệ thống Computer Vision & MLOps end-to-end phát hiện và định vị hai loại bệnh hại nguy hiểm trên lá lúa bằng mô hình **YOLOv8**:

- 🌾 **Bacterial Leaf Blight (Bạc lá lúa)** - Bệnh do vi khuẩn *Xanthomonas oryzae* gây nên.
- 🍂 **Brown Spot (Đốm nâu)** - Bệnh do nấm *Bipolaris oryzae* gây ra.

Dự án được xây dựng theo tiêu chuẩn MLOps doanh nghiệp: kiểm toán dữ liệu đa nguồn, loại bỏ trùng lặp bằng **pHash + BK-Tree**, phân chia dữ liệu theo **Group-aware Split** triệt tiêu data leakage, đánh giá mô hình khách quan với **Validation-only Model Selection**, tích hợp REST API **FastAPI**, giao diện Web **Streamlit Dashboard**, container **Docker** và xuất mô hình đa nền tảng (**ONNX, OpenVINO, TorchScript**).

---

## 🎯 Vì Sao Dự Án Này Đáng Chú Ý Trên Portfolio CV?

Trong thực tế xây dựng ứng dụng Computer Vision từ dữ liệu công khai, thách thức lớn nhất không nằm ở câu lệnh `model.train()` mà nằm ở **chất lượng dữ liệu và giao thức kiểm thử**:

1. **Rò rỉ dữ liệu (Data Leakage) nghiêm trọng**: Các bộ dữ liệu công khai (như Roboflow, Kaggle) thường chứa ảnh augment hoặc ảnh được cắt từ cùng một bức ảnh gốc. Nếu chia ngẫu nhiên từng file, ảnh gốc sẽ nằm ở tập Train còn ảnh augment nằm ở tập Test, khiến mAP cao giả tạo (~99%) nhưng mô hình thất bại khi triển khai thực tế.
2. **Nhiễu gán nhãn & đa định dạng (Taxonomy & Annotation Noise)**: Mỗi nguồn dữ liệu định nghĩa tên lớp khác nhau và sử dụng định dạng nhãn khác nhau (Bounding Box vs. Polygon).

### 💡 Giải Pháp Triển Khai Trong Pipeline

- **Taxonomy Normalization**: Đọc trực tiếp `data.yaml` của từng nguồn, chuẩn hóa alias tên lớp về hai lớp mục tiêu chuẩn.
- **Polygon to Box Conversion**: Tự động chuyển đổi các nét vẽ Polygon thành Bounding Box (Bounding Envelope) chuẩn YOLO `[x_center, y_center, width, height]`.
- **Exact Deduplication (SHA-256)**: Loại bỏ các file ảnh trùng lặp tuyệt đối bằng thuật toán SHA-256.
- **Near-Duplicate Grouping (pHash + BK-Tree + Union-Find)**: Trích xuất Perceptual Hash 64-bit qua biến đổi Cosin rời rạc (DCT-II), xây dựng cây **BK-Tree** để truy vấn khoảng cách Hamming trong thời gian $O(\log N)$, và hợp nhất các ảnh cùng nguồn gốc thành một `group_id` bằng **Union-Find**.
- **Group-aware Stratified Split**: Phân chia tập dữ liệu Train (70%), Val (15%), Test (15%) theo `group_id`, đảm bảo ảnh gốc và biến thể của nó nằm gọn trong cùng một tập dữ liệu.
- **Locked Test Set Protocol**: Khóa tập Test bằng cờ `--confirm-final-test`. Mô hình chỉ được chọn dựa trên Validation mAP50-95.

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

```text
Hai dataset ZIP nén nguồn
      │
      ▼
Giải nén an toàn ──► Đọc taxonomy ──► Chuẩn hóa Polygon/BBox
      │
      ▼
Kiểm tra ảnh/nhãn ──► SHA-256 dedup ──► pHash BK-Tree grouping
      │
      ▼
Group-aware split ──► Dataset YOLO sạch + manifest + audit report
      │
      ├──► YOLOv8n baseline ──┐
      └──► YOLOv8s candidate ─┴──► Chọn mô hình bằng Validation mAP50-95
                                      │
                                      ▼
                     Error analysis ──► Final Test một lần duy nhất
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
             CLI/FastAPI API      Streamlit       ONNX/OpenVINO
```

---

## 📐 Công Thức Toán Học & Metrics Đánh Giá

Pipeline tính toán chính xác các độ đo chuẩn mực của Object Detection:

### 1. Intersection over Union (IoU)
$$\text{IoU}(B_{pred}, B_{gt}) = \frac{\text{Area}(B_{pred} \cap B_{gt})}{\text{Area}(B_{pred} \cup B_{gt})}$$

### 2. Precision & Recall
$$P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}$$

### 3. Mean Average Precision (mAP)
$$\text{AP} = \int_{0}^{1} P(R) \, dR, \quad \text{mAP50-95} = \frac{1}{N_{classes}} \sum_{c=1}^{N_{classes}} \text{mAP}_{c} \ @ \ \text{IoU} \in [0.50:0.05:0.95]$$

---

## 📁 Cấu Trúc Repository

```text
.
├── .github/workflows/ci.yml       # Automated CI Gate (Ruff & Pytest)
├── app/
│   ├── api.py                     # RESTful FastAPI (Health, Info, Predict)
│   └── dashboard.py               # Streamlit Multi-Tab Dashboard
├── configs/
│   ├── default.yaml               # Cấu hình mặc định
│   ├── yolov8n_baseline.yaml      # Cấu hình YOLOv8n baseline
│   └── yolov8s_champion.yaml      # Cấu hình YOLOv8s candidate
├── data/
│   ├── README.md                  # Data Card chi tiết
│   └── sample/                    # Ảnh mẫu thử nghiệm nhanh
├── scripts/
│   └── create_demo_assets.py      # Script sinh dữ liệu mẫu giả định (Demo Quickstart)
├── src/rice_leaf_detection/
│   ├── annotations.py             # Chuẩn hóa nhãn & Polygon -> BBox
│   ├── compare.py                 # Xếp hạng mô hình theo Validation set
│   ├── config.py                  # Dataclass cấu hình bất biến & Validation
│   ├── constants.py               # Taxonomy & hằng số hệ thống
│   ├── deduplication.py           # Thuật toán BK-Tree & Union-Find pHash
│   ├── error_analysis.py          # Phân tích lỗi TP/FP/FN & Negative samples
│   ├── evaluate.py                # Đánh giá metric Val / Test
│   ├── export.py                  # Xuất mô hình ONNX/OpenVINO/TorchScript
│   ├── inference.py               # Class suy luận dùng chung RiceLeafDetector
│   ├── predict.py                 # CLI dự đoán ảnh, thư mục, video
│   ├── prepare.py                 # Pipeline xử lý dữ liệu end-to-end
│   ├── train.py                   # Pipeline huấn luyện YOLOv8
│   └── utils.py                   # Tiện ích mã hóa pHash, SHA-256, Zip an toàn
├── tests/                         # Test Suite bao phủ 22 test cases
├── Dockerfile                     # Docker container có Healthcheck
├── MODEL_CARD.md                  # Model Card chi tiết
├── pyproject.toml                 # Package setup & cấu hình tools
└── requirements.txt               # Thư viện phụ thuộc
```

---

## 🚀 Hướng Dẫn Nhanh (Quick Start)

### 1. Cài Đặt Môi Trường

```bash
# Clone repository
git clone <repository-url>
cd rice-leaf-disease-recognition

# Tạo môi trường ảo Python
python -m venv .venv

# Active môi trường ảo (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Active môi trường ảo (Linux/macOS)
# source .venv/bin/activate

# Cài đặt package ở chế độ Editable kèm phụ thuộc app & dev
python -m pip install --upgrade pip
pip install -e ".[app,dev]"
```

### 2. Sinh Dữ Liệu Demo Quickstart (Chỉ Mất 3 Giây)

Bạn có thể chạy thử **toàn bộ hệ thống** ngay lập tức mà không cần tải dữ liệu nặng hàng GB:

```bash
python scripts/create_demo_assets.py
```

Lệnh trên sẽ tự động sinh:
- Các ảnh mẫu lá lúa tại `data/sample/`.
- 2 file ZIP dữ liệu mẫu `RiceLeafAnnotatedDataset.zip` và `dataset1.zip`.

### 3. Chạy Pipeline Xử Lý & Kiểm Toán Dữ Liệu

```bash
rice-prepare --overwrite
```

Đầu ra tạo tại `data/processed/rice_leaf_detection/`:
- `train/`, `val/`, `test/` (Ảnh và nhãn YOLO sạch).
- `data.yaml` (Cấu hình bộ dữ liệu).
- `manifest.csv` (Lưu vết từng bức ảnh, split, sha256, phash).
- `audit_report.json` (Thống kê số lượng ảnh trùng bị loại, liên kết pHash).

### 4. Đánh Giá & Phân Tích Lỗi

```bash
# Chạy Unit Tests kiểm thử toàn bộ hệ thống
python -m pytest

# Kiểm tra Linter định dạng code
python -m ruff check src app tests scripts
```

### 5. Dự Đoán Ảnh Trực Tiếp Qua CLI

```bash
rice-predict --weights artifacts/best.pt --source data/sample/rice_leaf.jpg
```

---

## 🌐 Web Service & Dashboard UI

### 1. RESTful FastAPI Server

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Các Endpoints khả dụng:
- `GET /health`: Kiểm tra sức khỏe dịch vụ.
- `GET /info`: Trả về thông tin mô hình, phiên bản và các lớp hỗ trợ.
- `POST /predict`: Upload ảnh lá lúa nhận kết quả Bounding Boxes dạng JSON.

Ví dụ gọi API bằng `curl`:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@data/sample/rice_leaf.jpg"
```

### 2. Streamlit Interactive Dashboard

```bash
streamlit run app/dashboard.py
```

Dashboard hỗ trợ 3 Tabs chuyên nghiệp:
- 🎯 **Phân Tích Ảnh**: Upload ảnh, tùy chỉnh Confidence Threshold & IoU, xem Bounding Box trực quan.
- 📊 **Thống Kê Dữ Liệu & Audit**: Kiểm tra báo cáo audit_report.json và manifest dữ liệu.
- ℹ️ **Kiến Trúc & Protocol**: Xem sơ đồ pipeline MLOps và chiến lược chống rò rỉ dữ liệu.

---

## 🐳 Đóng Gói Docker

```bash
# Build Docker Image
docker build -t rice-leaf-detection .

# Chạy Docker Container
docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts" rice-leaf-detection
```

Container tích hợp sẵn Healthcheck tự động tại `http://localhost:8000/health`.

---

## 📦 Export Mô Hình Đa Nền Tảng

```bash
# Xuất mô hình sang ONNX
rice-export --weights artifacts/best.pt --format onnx --imgsz 640 --simplify

# Xuất mô hình sang OpenVINO hoặc TorchScript
rice-export --weights artifacts/best.pt --format openvino
```

File xuất được lưu tại `artifacts/export/` kèm `metadata.json` chứa SHA-256 Checksum.

---

## 💼 CV Highlights & Gợi Ý Trả Lời Phỏng Vấn

### Các dòng tóm tắt đưa vào CV:

> **Data & MLOps Engineer**: Xây dựng pipeline YOLOv8 phát hiện bạc lá và đốm nâu trên lá lúa; xử lý đa nguồn dữ liệu, chuẩn hóa Polygon -> BBox, thiết kế thuật toán deduplication (**SHA-256 + pHash BK-Tree**) và **Group-aware Stratified Split** triệt tiêu Data Leakage giữa Train và Test.

> **Production Deployment**: Đóng gói quy trình Computer Vision thành Python Package với CLI (`rice-prepare`, `rice-train`, `rice-predict`), REST API (**FastAPI**), Web Dashboard (**Streamlit**), **Docker** container có healthcheck và **CI Gate (GitHub Actions)** với 22 unit tests.

### Câu hỏi phỏng vấn chuyên sâu gợi ý:

1. **Q: Vì sao không chia train/test ngẫu nhiên từng file ảnh?**
   - *A: Ảnh lá lúa từ các nguồn công khai thường chứa các biến thể augment hoặc crop từ cùng ảnh gốc. Chia ngẫu nhiên làm ảnh gần giống nhau xuất hiện ở cả train và test, gây Data Leakage và đẩy mAP cao giả tạo. Pipeline của tôi gom ảnh theo `group_id` bằng pHash BK-Tree rồi mới chia tập dữ liệu.*
2. **Q: Cấu trúc BK-Tree giải quyết bài toán pHash như thế nào?**
   - *A: So sánh pHash giữa $N$ ảnh thông qua khoảng cách Hamming tốn $O(N^2)$. BK-Tree tận dụng bất đẳng thức tam giác metric space giúp giảm độ phức tạp truy vấn khoảng cách Hamming xuống $O(\log N)$.*
3. **Q: Vì sao tập Test bị khóa bằng cờ `--confirm-final-test`?**
   - *A: Nếu dùng kết quả trên tập Test để điều chỉnh hyperparameter hay ngưỡng confidence, tập Test đã bị "rò rỉ" thông tin. Tập Test trong dự án chỉ được đánh giá duy nhất một lần sau khi đã chốt mô hình bằng tập Validation.*

---

## 📄 Giấy Phép (License)

Mã nguồn dự án phát hành theo [MIT License](LICENSE).
