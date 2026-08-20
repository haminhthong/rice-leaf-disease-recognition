# Phát hiện bệnh trên lá lúa bằng YOLOv8

Hệ thống computer vision phát hiện và định vị hai bệnh phổ biến trên lá lúa trong ảnh:

- **Bacterial Leaf Blight** — bạc lá lúa.
- **Brown Spot** — đốm nâu.

Dự án sử dụng **YOLOv8s Object Detection**, trả về đồng thời loại bệnh, độ tin cậy và bounding box của vùng bệnh. Source code được tái cấu trúc từ notebook nghiên cứu thành pipeline chạy độc lập, có CLI cho toàn bộ vòng đời dữ liệu và mô hình.

> Đây là dự án hỗ trợ sàng lọc hình ảnh, không thay thế chẩn đoán của chuyên gia nông nghiệp.

## Điểm nổi bật

- Hợp nhất hai bộ dữ liệu YOLO có taxonomy khác nhau về hai lớp mục tiêu thống nhất.
- Chuyển annotation polygon thành bounding box và kiểm tra tọa độ/định dạng nhãn.
- Giữ ảnh lá khỏe hoặc bệnh ngoài phạm vi làm **negative samples**, giúp giảm false positive.
- Loại ảnh trùng tuyệt đối bằng SHA-256; nhóm ảnh gần trùng bằng perceptual hash và BK-tree.
- Chia train/validation/test theo nhóm ảnh liên quan để hạn chế **data leakage**.
- Ghi `manifest.csv` và `audit_report.json` để truy vết nguồn và chất lượng dữ liệu.
- Cố định seed, phiên bản Ultralytics và cấu hình huấn luyện để tăng khả năng tái lập.
- Khóa đánh giá test bằng cờ xác nhận, tránh dùng test set trong quá trình tuning.
- Hỗ trợ dự đoán ảnh, thư mục ảnh và video bằng command line.

## Kiến trúc pipeline

```text
Hai dataset ZIP
      │
      ▼
Giải nén an toàn → Chuẩn hóa taxonomy/annotation → Kiểm tra ảnh và nhãn
      │
      ▼
SHA-256 dedup → pHash grouping → Group-aware train/val/test split
      │
      ▼
Dataset YOLO sạch → Train YOLOv8s → Validation → Final test → Inference
```

## Cấu trúc dự án

```text
.
├── src/rice_leaf_detection/
│   ├── annotations.py       # Chuẩn hóa class và annotation
│   ├── constants.py         # Taxonomy và cấu hình mặc định
│   ├── deduplication.py     # SHA-256, BK-tree và Union-Find
│   ├── prepare.py           # Xây dựng dataset sạch
│   ├── train.py             # Huấn luyện/resume YOLOv8
│   ├── evaluate.py          # Đánh giá val/test và xuất metrics
│   ├── predict.py           # Inference ảnh/video
│   └── utils.py
├── tests/                   # Unit tests cho logic annotation
├── requirements.txt
├── pyproject.toml
└── README.md
```

Hai tệp dữ liệu gốc đặt ở thư mục gốc nhưng được bỏ qua khi commit Git:

```text
RiceLeafAnnotatedDataset.zip
dataset1.zip
```

## Công nghệ sử dụng

- Python 3.10+
- PyTorch, Ultralytics YOLOv8
- OpenCV, Pillow, ImageHash
- Pandas, NumPy, scikit-learn
- PyYAML, tqdm

## Cài đặt

Khuyến nghị dùng GPU NVIDIA hỗ trợ CUDA để huấn luyện. Inference và kiểm thử pipeline vẫn chạy được trên CPU.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Phiên bản `ultralytics==8.3.220` được cố định theo notebook gốc. Khi cần CUDA, hãy cài bản PyTorch phù hợp với GPU theo tài liệu chính thức trước khi chạy `pip install -e .`.

## 1. Chuẩn bị dữ liệu

Đặt hai ZIP ở thư mục gốc rồi chạy:

```bash
rice-prepare
```

Hoặc dùng module Python:

```bash
python -m rice_leaf_detection.prepare \
  --archives RiceLeafAnnotatedDataset.zip dataset1.zip \
  --output data/processed/rice_leaf_detection
```

Các tùy chọn quan trọng:

| Tùy chọn | Mặc định | Ý nghĩa |
|---|---:|---|
| `--phash-distance` | `2` | Khoảng cách Hamming tối đa để nhóm ảnh gần trùng |
| `--drop-negatives` | tắt | Loại ảnh không có hai lớp mục tiêu; thường không khuyến nghị |
| `--overwrite` | tắt | Tạo lại dataset đầu ra đã tồn tại |

Đầu ra gồm:

```text
data/processed/rice_leaf_detection/
├── train/{images,labels}/
├── val/{images,labels}/
├── test/{images,labels}/
├── data.yaml
├── manifest.csv
└── audit_report.json
```

`manifest.csv` lưu split, nguồn, kích thước ảnh, hash, group và số instance mỗi lớp. `audit_report.json` lưu ảnh hỏng, nhãn lỗi, file label thiếu, ảnh trùng và xung đột annotation.

## 2. Huấn luyện

```bash
rice-train \
  --data data/processed/rice_leaf_detection/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640
```

Cấu hình chính kế thừa từ notebook:

- Optimizer: AdamW
- Learning rate ban đầu: `0.001`
- Weight decay: `0.0005`
- Early stopping patience: `25`
- Seed: `42`
- Input size: `640 × 640`
- Checkpoint mỗi 10 epoch

Nếu không truyền `--device` và `--batch`, chương trình tự chọn GPU với batch 16; nếu chỉ có CPU, dùng batch 4.

Resume một run bị gián đoạn:

```bash
rice-train --resume runs/train/<run-name>/weights/last.pt
```

Resume dùng `resume=True`, khôi phục cả epoch, optimizer và learning-rate scheduler thay vì chỉ nạp lại weights.

## 3. Đánh giá

Đánh giá trên validation để lựa chọn mô hình và ngưỡng:

```bash
rice-evaluate \
  --weights runs/train/<run-name>/weights/best.pt \
  --split val
```

Chỉ đánh giá test **một lần sau khi chốt mô hình**:

```bash
rice-evaluate \
  --weights runs/train/<run-name>/weights/best.pt \
  --split test \
  --confirm-final-test
```

Chương trình xuất `metrics.json`, `per_class_metrics.csv`, confusion matrix và các đường cong PR/F1. Các chỉ số cần báo cáo cho bài toán detection là:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Kết quả riêng cho từng lớp

Không dùng “accuracy theo ảnh” thay cho các chỉ số object detection trên.

## 4. Dự đoán

Ảnh đơn:

```bash
rice-predict \
  --weights runs/train/<run-name>/weights/best.pt \
  --source samples/rice_leaf.jpg \
  --conf 0.25
```

Thư mục ảnh hoặc video dùng cùng lệnh bằng cách đổi `--source`. Kết quả có bounding box được lưu trong `runs/predict/results/`.

```bash
rice-predict --weights path/to/best.pt --source path/to/images --save-txt
rice-predict --weights path/to/best.pt --source path/to/video.mp4
```

`--conf 0.25` chỉ là giá trị khởi đầu. Ngưỡng triển khai nên được chọn từ PR/F1 curve trên validation set.

## Kiểm thử

```bash
pip install pytest
pytest -q
```

## Dữ liệu

Pipeline sử dụng hai nguồn trong notebook gốc:

| Dataset | Quy mô mô tả trong metadata | Lớp nguồn | Ghi chú giấy phép |
|---|---:|---:|---|
| RiceLeafAnnotatedDataset | 3.567 ảnh | 8 | README nguồn không nêu giấy phép; cần xác minh trước khi phân phối |
| Rice-Leaf-Disease v6 | 2.304 ảnh | 4 | CC BY 4.0, xuất từ Roboflow |

Chỉ annotation của `Bacterial Leaf Blight` và `Brown Spot` được giữ làm positive labels. Ảnh chỉ chứa các lớp còn lại được giữ với label rỗng làm negative sample. Hai ZIP và dataset sinh ra không được commit vào repository.

## Kết quả thí nghiệm

Notebook nguồn bắt đầu một run `YOLOv8s` trên CPU nhưng dừng trong epoch đầu, vì vậy **chưa có kết quả validation/test hoàn chỉnh để công bố**. Không nên đưa số liệu chưa chạy xong vào CV.

Sau khi huấn luyện, cập nhật bảng sau từ `metrics.json`:

| Split | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Validation | TBD | TBD | TBD | TBD |
| Test cuối cùng | TBD | TBD | TBD | TBD |

Khi bổ sung kết quả, nên ghi thêm cấu hình GPU, thời gian huấn luyện, số ảnh sau dedup, số negative samples và metric từng lớp để người đọc có thể đánh giá thí nghiệm đầy đủ.

## Các quyết định kỹ thuật

### Vì sao đây là object detection?

Ảnh có thể chứa nhiều vùng bệnh. Classification chỉ cho biết ảnh thuộc lớp nào, còn detection xác định cả loại bệnh lẫn vị trí tổn thương để trực quan hóa và hỗ trợ kiểm tra kết quả.

### Vì sao chia dữ liệu lại theo group?

Hai dataset có thể chứa bản sao, ảnh đã augment hoặc nhiều export của cùng ảnh gốc. Chia ngẫu nhiên từng file có thể đưa các biến thể gần như giống nhau vào cả train và test, làm metric cao giả tạo. Pipeline nhóm các ảnh liên quan trước rồi mới chia split.

### Vì sao giữ negative samples?

Nếu chỉ huấn luyện trên ảnh có hai bệnh mục tiêu, mô hình dễ dự đoán nhầm lá khỏe hoặc bệnh khác thành hai lớp đã học. Label rỗng giúp mô hình học thêm tín hiệu nền và giảm false positive.

## Hạn chế và hướng phát triển

- Hiện chỉ hỗ trợ hai bệnh, chưa bao phủ đầy đủ bệnh và sâu hại trên lúa.
- Chất lượng phụ thuộc vào độ chính xác và tính đa dạng của annotation nguồn.
- Cần đánh giá thêm trên ảnh thực địa khác phân phối dữ liệu huấn luyện.
- pHash có thể nhóm nhầm một số ảnh có bố cục gần giống; cần audit thủ công các nhóm lớn.
- Có thể thử YOLO phiên bản mới hơn, augmentation phù hợp miền dữ liệu và model export ONNX/TensorRT.
- Có thể xây dựng API hoặc ứng dụng web/mobile sau khi hoàn tất benchmark và hiệu chỉnh confidence.

## Gợi ý mô tả trong CV

> Xây dựng pipeline phát hiện bạc lá và đốm nâu trên cây lúa bằng YOLOv8; hợp nhất hai bộ dữ liệu đa taxonomy, chuẩn hóa polygon/bounding box, loại ảnh trùng bằng SHA-256 và perceptual hashing, thiết kế group-aware split chống data leakage, đồng thời tự động hóa train/evaluate/inference bằng Python CLI.

Sau khi có kết quả thật, thêm một bullet định lượng, ví dụ: `Đạt mAP@0.5:0.95 = X trên test set gồm Y ảnh`.

## Nguồn gốc

Source code được tái cấu trúc từ notebook `Nhan_Dien_Benh_Cay_Lua.ipynb`. Notebook ban đầu tối ưu cho Google Colab/Drive; phiên bản này loại bỏ phụ thuộc Colab, thay đường dẫn cố định bằng đối số CLI và tách logic thành các module có thể kiểm thử, tái sử dụng và đưa lên GitHub.

