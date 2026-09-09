# 🌾 Rice Leaf Disease Detection
[![CI](https://github.com/haminhthong/Rice-Leaf-Disease-Decognition/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/Rice-Leaf-Disease-Decognition/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-111F68)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> Pipeline Object Detection và Decision Support cho ảnh lá lúa, tập trung vào hai vùng triệu chứng: Bacterial Leaf Blight và Brown Spot.

## 1. Bài toán và phạm vi ứng dụng

Dự án giải bài toán **phát hiện và định vị vùng triệu chứng** bằng YOLOv8, không phải phân loại toàn ảnh. Một ảnh có thể có nhiều bounding box, nhiều lớp hoặc không có bounding box thuộc phạm vi.

| Class ID | Tên chuẩn | Tên tiếng Việt |
|---|---|---|
| `0` | `Bacterial_Leaf_Blight` | Bạc lá lúa |
| `1` | `Brown_Spot` | Đốm nâu |

Phạm vi sử dụng là hỗ trợ trinh sát và sàng lọc ban đầu. `NO_SUPPORTED_SYMPTOM_DETECTED` chỉ có nghĩa là không tìm thấy triệu chứng thuộc hai lớp đã hỗ trợ; trạng thái này không khẳng định lá khỏe mạnh. Hệ thống không tự chẩn đoán toàn bộ bệnh trên cây lúa và không tự đưa ra liều lượng hay quyết định phun thuốc. Kết quả cần được cán bộ hoặc chuyên gia nông nghiệp kiểm tra.

### Hợp đồng nhãn

Mỗi bounding box biểu diễn vùng triệu chứng quan sát được trên phiến lá, theo định dạng YOLO chuẩn hóa `class x_center y_center width height`, trong đó các tọa độ thuộc `[0, 1]`. Polygon nguồn được chuyển thành bounding envelope.

Ảnh âm tính được phân biệt trong manifest:

- `TRUE_NEGATIVE`: file nhãn rỗng tồn tại và đã được xác minh.
- `OUT_OF_SCOPE_NEGATIVE`: chỉ có lớp bệnh ngoài phạm vi hai lớp mục tiêu.
- `INVALID_OR_MISSING`: thiếu file nhãn hoặc có dòng nhãn lỗi; luôn quarantine, không biến thành nhãn rỗng.

## 2. Quy trình kỹ thuật canonical

Đây là luồng duy nhất chi phối mã nguồn, cấu hình và báo cáo. Orchestrator nằm tại `src/rice_leaf_detection/pipeline.py`; cấu hình chuẩn duy nhất là `configs/default.yaml`.

```mermaid
flowchart TD
    A[ZIP nguồn<br/>RiceLeafAnnotatedDataset.zip<br/>dataset1.zip] --> B[Stage data: giải nén an toàn<br/>safe_extract_zip + phát hiện data.yaml]
    B --> C[Đọc data.yaml nguồn<br/>chuẩn hóa alias lớp về class 0/1]
    C --> D[Đọc và kiểm tra ảnh, nhãn<br/>BBox hoặc Polygon -> BBox<br/>tọa độ hữu hạn và trong [0,1]]
    D -->|nhãn lỗi hoặc thiếu| Q[(Quarantine<br/>audit_report.json)]
    D -->|ảnh hợp lệ| E[Phân loại annotation_status<br/>TARGET_POSITIVE / TRUE_NEGATIVE / OUT_OF_SCOPE_NEGATIVE]
    E --> F[Exact dedup SHA-256<br/>xung đột nhãn -> reports/data_conflicts]
    F --> G[Near-duplicate grouping<br/>pHash DCT-II 64-bit + BK-Tree<br/>Union-Find + original_key]
    G --> H[Greedy group/source/class-aware split<br/>Train 70% / Val 15% / Test 15%]
    H --> I{Quality gates}
    I -->|fail| Q
    I -->|pass| J[(Processed dataset<br/>data.yaml + manifest.csv<br/>data_manifest.json + audit_report.json)]
    J --> K[Stage train<br/>YOLOv8s @ 640<br/>seed, augmentation, metadata]
    K --> L[Stage evaluate<br/>Validation only<br/>mAP50, mAP50-95, precision, recall]
    L --> M[Stage tune_policy<br/>grid search review/accept<br/>macro image recall - false alarm]
    M --> N[(policy_tuning.json<br/>selected_policy)]
    N --> O[Stage errors<br/>TP / FP / FN / localization<br/>confusion / duplicate + lesion size]
    N --> P{Đã chốt model và policy?}
    P -->|chưa| K
    P -->|đã chốt + --confirm-final-test| R[Stage test một lần<br/>final_test_report.json<br/>hash model + hash split test]
    P -->|chưa mở khóa| S[Test bị khóa]
    R --> T[Stage export<br/>model.pt + ONNX/TorchScript<br/>parity quality gate]
    N --> T
    T --> U[(artifacts/<br/>model.pt<br/>detection_policy.json<br/>model_metadata.json<br/>metadata.json)]
    U --> V[Shared RiceLeafDetector<br/>artifact contract + policy]
    V --> W[FastAPI /predict<br/>magic bytes + pixel limit<br/>JSON Pydantic]
    V --> X[Streamlit dashboard]
    V --> Y[CLI rice-predict]
    W --> Z[DETECTED / REVIEW_REQUIRED /<br/>NO_SUPPORTED_SYMPTOM_DETECTED]
    X --> Z
    Y --> Z
```

### Ý nghĩa các stage

1. **data**: ingest nhiều ZIP, chuẩn hóa taxonomy, kiểm tra annotation, loại ảnh trùng, gom nhóm ảnh gần trùng, chia split và ghi manifest.
2. **train**: chỉ huấn luyện `YOLOv8s` theo `configs/default.yaml`; augmentation chỉ truyền vào train, validation/test dùng preprocessing tất định.
3. **evaluate**: đo mAP và metric theo lớp trên Validation; không dùng Test để chọn model.
4. **tune_policy**: chạy model một lần trên Validation, sau đó đánh giá nhiều cặp ngưỡng. Policy được chọn theo objective cân bằng image recall, accepted false alarm, review recall và review false alarm.
5. **errors**: dùng chính model và policy đã chốt để phân tích lỗi theo nguồn, loại negative, kích thước tổn thương và taxonomy.
6. **test**: chỉ mở bằng `--confirm-final-test`; báo cáo tồn tại sẽ khóa lần chạy tiếp theo. `--force-reopen-test` mở lại nhưng đánh dấu `test_compromised=true`.
7. **export**: đóng gói trọng số, policy, checksum và kiểm tra parity giữa PyTorch và artifact export trước khi phục vụ.

## 3. Luồng dữ liệu và hợp đồng artifact

| Artifact | Tạo ở stage | Được stage sau sử dụng | Nội dung |
|---|---|---|---|
| `data.yaml` | data | train/evaluate/test | đường dẫn train, val, test và 2 class |
| `manifest.csv` | data | train, tune_policy, errors | split, source, group, hash, trạng thái nhãn, số instance |
| `data_manifest.json` | data | test/report | dataset id, archive hash, split hash, provenance |
| `audit_report.json` | data | dashboard/report | lỗi nhãn, ảnh hỏng, dedup, phân bố source/split |
| `run_metadata.json` | train | truy vết run | seed, config, dataset hash, model hash, thiết bị |
| `val_metrics.json` và `experiments.csv` | evaluate | báo cáo Validation | precision, recall, mAP50, mAP50-95 |
| `policy_tuning.json` | tune_policy | errors/test/export/runtime | policy đã chọn và top candidate |
| `error_summary.json` | errors | model report | taxonomy lỗi, negative benchmark, source slices |
| `final_test_report.json` | test | báo cáo cuối | metric Test và hash model/dataset |
| `model.pt`, `detection_policy.json`, `model_metadata.json` | export | API, dashboard, CLI | artifact triển khai có checksum |

Runtime chỉ nạp `artifacts/model.pt` khi metadata và policy đi kèm khớp class, image size, checksum và ngưỡng quyết định.

## 4. Cấu trúc thư mục dự án

```text
rice-leaf-disease-recognition/
├── .github/workflows/ci.yml       # CI: format, lint, pip check, smoke test, pytest
├── app/
│   ├── api.py                     # FastAPI: /health, /info, /predict
│   ├── dashboard.py               # Streamlit dashboard
│   ├── dependencies.py            # detector singleton
│   ├── schemas.py                 # Pydantic response models
│   ├── settings.py                # biến môi trường runtime
│   └── validation.py              # magic bytes, pixel limit, quality gate
├── configs/
│   └── default.yaml               # cấu hình canonical duy nhất
├── data/
│   ├── README.md                  # Data Card
│   └── sample/                    # ảnh mẫu commit trong repo
├── docs/
│   ├── BAO_CAO_CAI_TIEN_DU_AN.docx
│   └── HUONG_DAN_CAI_THIEN_CHI_TIET.docx
├── scripts/
│   ├── create_demo_assets.py      # tạo ZIP/ảnh demo synthetic
│   └── run_pipeline.py            # entry point không cần cài console script
├── src/rice_leaf_detection/
│   ├── annotations.py             # parse, validate, normalize label
│   ├── config.py                  # dataclass + validate YAML
│   ├── constants.py               # class, split, status constants
│   ├── deduplication.py           # SHA, pHash, BK-Tree, Union-Find
│   ├── error_analysis.py          # matching và error taxonomy
│   ├── evaluate.py                # metric theo ảnh và CLI evaluate
│   ├── export.py                  # export và prediction parity
│   ├── inference.py               # detector + decision policy runtime
│   ├── pipeline.py                # orchestrator canonical
│   ├── policy.py                  # tuning policy Validation-only
│   ├── predict.py                 # CLI prediction
│   ├── prepare.py                 # data preparation
│   ├── train.py                   # CLI training
│   └── utils.py                   # IO, hash, seed, safe ZIP
├── tests/                         # unit/API/logic tests; không cần model thật
├── Dockerfile                     # FastAPI container
├── MODEL_CARD.md                 # giới hạn và protocol mô hình
├── pyproject.toml                # package, entry points, dev/app extras
├── requirements.txt               # dependency runtime tối thiểu
├── LICENSE
└── README.md
```

Các thư mục `data/processed`, `data/extracted`, `runs`, `reports`, `artifacts`, file ZIP và trọng số `.pt` bị ignore vì là dữ liệu sinh ra hoặc file lớn. Chỉ ảnh mẫu và tài liệu cần thiết được theo dõi trong Git.

## 5. Cài đặt

Yêu cầu Python 3.10 trở lên. Chạy từ thư mục gốc repository:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[app,dev]"
```

Nếu chỉ chạy xử lý dữ liệu và test lõi:

```bash
pip install -e ".[dev]"
```

Nếu dùng dữ liệu thật, đặt `RiceLeafAnnotatedDataset.zip` và `dataset1.zip` ở thư mục gốc hoặc truyền đường dẫn qua `--archives`. License và annotation unit của từng archive phải được xác minh trước khi phát hành model; pipeline ghi trạng thái chưa xác minh vào audit.

## 6. Chạy kiểm tra và thử nghiệm

### Kiểm tra code và CI local

```bash
ruff format --check src app scripts tests
ruff check src app scripts tests
pip check
pytest -v
```

Test parity export cần PyTorch/Ultralytics đầy đủ; nếu môi trường tối giản không có `torch`, module test parity sẽ được skip có chủ đích, còn các test logic/API vẫn phải pass.

### Tạo dữ liệu demo

```bash
python scripts/create_demo_assets.py
rice-prepare --overwrite
```

Lệnh trên tạo hai ảnh mẫu trong `data/sample/` và hai ZIP synthetic đủ cấu trúc YOLO. Nó không đại diện cho bằng chứng chất lượng mô hình thật.

### Xem DAG và chạy từng stage

```bash
python scripts/run_pipeline.py --dry-run

python scripts/run_pipeline.py --stage data
python scripts/run_pipeline.py --stage train --epochs 10
python scripts/run_pipeline.py --stage evaluate
python scripts/run_pipeline.py --stage tune_policy
python scripts/run_pipeline.py --stage errors

# Chỉ chạy sau khi model và policy đã chốt
python scripts/run_pipeline.py --stage test --confirm-final-test

# Đóng gói artifact phục vụ
python scripts/run_pipeline.py --stage export

# End-to-end; có test chính thức thì thêm --confirm-final-test
python scripts/run_pipeline.py --stage all --epochs 10
```

Dùng `--force` để chạy lại các stage có output cũ. Không dùng `--force-reopen-test` trong quy trình báo cáo bình thường.

### Suy luận CLI

```bash
rice-predict --weights artifacts/model.pt --source data/sample/bacterial_leaf_blight_sample.jpg
```

CLI hỗ trợ ảnh, thư mục hoặc video theo nguồn mà Ultralytics chấp nhận. Kết quả được ghi dưới `runs/predict/results`; policy được lấy từ config và được kiểm tra lại với policy đóng gói trong artifact.

### FastAPI

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health/live`: tiến trình đang chạy.
- `GET /health/ready`: model artifact đã nạp được chưa.
- `GET /info`: class, image size, candidate threshold và decision policy.
- `POST /predict`: upload JPEG/PNG/WebP; kiểm tra magic bytes, giới hạn 10 MB, giới hạn 25 triệu pixel và trả JSON Pydantic.

```bash
curl -X POST http://localhost:8000/predict -F "file=@data/sample/bacterial_leaf_blight_sample.jpg"
```

### Streamlit

```bash
streamlit run app/dashboard.py
```

Dashboard dùng cùng `RiceLeafDetector` với API, hiển thị bounding box, image summary và human-review flag.

### Docker

```bash
docker build -t rice-leaf-disease .
docker run --rm -p 8000:8000 ^
  -e RICE_MODEL_PATH=/app/artifacts/model.pt ^
  rice-leaf-disease
```

Docker image chứa code và dependency nhưng không chứa dataset hay trọng số bị ignore. Muốn readiness chuyển sang `ready`, cần mount/copy artifact model hợp lệ vào đường dẫn đã cấu hình.

## 7. Cấu hình và decision policy

`configs/default.yaml` là nguồn cấu hình cho train/evaluate/inference:

- `project.seed`: seed tái lập.
- `data.yaml`, `data.image_size`: dataset và kích thước ảnh.
- `model.architecture`, `model.weights`: kiến trúc/trọng số khởi tạo; canonical hiện là `yolov8s`.
- `training.*`: epoch, batch GPU/CPU, optimizer, learning rate, weight decay.
- `training.hsv_*`, hình học, mosaic/mixup: augmentation chỉ truyền khi train.
- `inference.candidate_confidence`, `inference.iou`: lấy candidate và NMS.
- `policy.review_threshold`, `policy.accept_threshold`: giá trị ưu tiên ban đầu; stage `tune_policy` chọn policy cuối trên Validation.

Runtime có thể override bằng biến môi trường:

```text
RICE_MODEL_PATH
RICE_MODEL_RELEASE_DIR
RICE_IMAGE_SIZE
RICE_CANDIDATE_CONFIDENCE
RICE_IOU
RICE_REVIEW_THRESHOLD
RICE_ACCEPT_THRESHOLD
RICE_CORS_ORIGINS
INFERENCE_CONCURRENCY
```

Artifact contract không cho phép runtime dùng policy khác policy đã đóng gói cùng model.

## 8. Báo cáo và tiêu chí đánh giá

Metric mô hình gồm precision, recall, mAP50, mAP50-95 và per-class AP/recall. Metric nghiệp vụ gồm image-level recall cho từng lớp và false-alarm rate tách riêng `TRUE_NEGATIVE` với `OUT_OF_SCOPE_NEGATIVE`.

Error taxonomy:

- true positive;
- false negative/missed lesion;
- false positive trên background;
- localization error khi IoU thấp;
- classification confusion giữa hai lớp;
- duplicate detection;
- recall theo small/medium/large lesion.

Test bị khóa để tránh dùng thông tin Test cho train, chọn policy hoặc sửa ngưỡng. Kết quả cuối chỉ có giá trị khi có dataset id, split hash, model hash và policy hash tương ứng.

## 9. CI và repo cleanliness

Workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) chạy trên push và pull request bằng Python 3.11:

1. cài package cùng app/dev extras;
2. kiểm tra format bằng Ruff;
3. lint bằng Ruff;
4. chạy `pip check`;
5. import smoke test package và FastAPI;
6. chạy toàn bộ pytest.

Các file legacy `compare.py`, hai config baseline/candidate và hai Markdown phân tán đã được loại bỏ để README này là tài liệu kỹ thuật chính; `MODEL_CARD.md` và `data/README.md` giữ vai trò tài liệu chuyên biệt cho model và data.

## 10. Giới hạn và trách nhiệm

Model phụ thuộc mạnh vào domain của archive huấn luyện. Ảnh ngoài điều kiện cận cảnh, ảnh quá tối/sáng, mờ, lá chồng lấn hoặc bệnh ngoài phạm vi có thể cho kết quả không ổn định. Detection score không phải xác suất bệnh lý đã hiệu chỉnh. Không sử dụng output để tự động quyết định hóa chất; luôn cần thẩm định chuyên môn.

## 11. License

Mã nguồn được phát hành theo [MIT License](LICENSE).
