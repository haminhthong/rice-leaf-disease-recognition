# Thẻ dữ liệu

## Mục đích

Dataset hợp nhất phục vụ phát hiện hai bệnh trên lá lúa: bạc lá và đốm nâu. Dữ liệu chỉ phù hợp cho nghiên cứu, học tập và xây dựng bản thử nghiệm; không dùng thay thế chẩn đoán chuyên môn.

## Nguồn

- `RiceLeafAnnotatedDataset.zip`: 8 lớp nguồn; giấy phép chưa được nêu rõ trong metadata đi kèm.
- `dataset1.zip`: 4 lớp nguồn, xuất từ Roboflow với giấy phép CC BY 4.0.

Các file ZIP không được lưu trong Git. Người sử dụng phải tự kiểm tra quyền phân phối trước khi công bố dữ liệu hoặc ảnh mẫu.

## Chuẩn hóa

- Giữ hai lớp mục tiêu và ánh xạ class ID về cùng taxonomy.
- Chuyển polygon thành bounding box YOLO.
- Cắt bbox lệch biên về miền tọa độ `[0, 1]` và ghi số lượng vào báo cáo audit.
- Giữ ảnh không có lớp mục tiêu làm negative sample.
- Loại ảnh trùng tuyệt đối bằng SHA-256.
- Nhóm ảnh gần trùng bằng pHash trước khi chia dữ liệu.

## Chia dữ liệu

Tỷ lệ mặc định là 70% train, 15% validation và 15% test. Việc chia được thực hiện theo nhóm ảnh thay vì từng file, nhằm ngăn ảnh gốc và biến thể gần trùng xuất hiện ở nhiều split.

## Artifact kiểm toán

Sau khi chạy `rice-prepare`, kiểm tra:

- `manifest.csv`: nguồn, split, group, hash, kích thước và số bbox theo lớp.
- `audit_report.json`: ảnh lỗi, nhãn lỗi, nhãn thiếu, bbox bị cắt biên, ảnh trùng và xung đột annotation.

## Hạn chế

- Dữ liệu từ ít nguồn có thể gây background bias và domain shift.
- Một phần ảnh nguồn đã được resize hoặc augment trước khi phát hành.
- pHash chỉ là phép đo gần đúng; các nhóm lớn cần được kiểm tra thủ công.
- Chưa có external test set độc lập từ ruộng lúa ngoài các nguồn trên.

