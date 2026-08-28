# Model Card - Rice Leaf Disease Detection

## Mục tiêu

Phát hiện và định vị hai loại tổn thương trên lá lúa: bạc lá và đốm nâu. Mô hình là công cụ hỗ trợ quan sát ảnh, không phải hệ thống chẩn đoán nông nghiệp.

## Phiên bản được công bố

Chưa có champion model được xác nhận. Chỉ điền bảng dưới đây sau khi hoàn thành validation, khóa cấu hình và chạy final test đúng một lần.

| Trường | Giá trị |
|---|---|
| Kiến trúc | Chưa xác định |
| Dataset version/hash | Chưa xác định |
| Seed | 42 |
| Precision test | Chưa xác định |
| Recall test | Chưa xác định |
| mAP50 test | Chưa xác định |
| mAP50-95 test | Chưa xác định |
| Confidence triển khai | Chưa xác định từ validation |

## Protocol lựa chọn

1. Huấn luyện YOLOv8n làm baseline và YOLOv8s làm ứng viên champion trên cùng split/seed.
2. Xếp hạng chỉ bằng validation mAP50-95.
3. Phân tích false positive, false negative và negative samples.
4. Chốt confidence bằng validation.
5. Chạy test một lần và ghi metric cuối cùng.

## Intended use

- Demo computer vision và portfolio kỹ thuật.
- Hỗ trợ khoanh vùng triệu chứng trong ảnh lá lúa có chất lượng phù hợp.
- Nghiên cứu pipeline dữ liệu, leakage và object detection.

## Không nên sử dụng

- Quyết định dùng thuốc hoặc xử lý nông nghiệp tự động.
- Ảnh cây trồng khác, ảnh vệ tinh hoặc ảnh không thể hiện rõ lá lúa.
- Suy luận các bệnh ngoài hai lớp đã huấn luyện.

## Rủi ro và hạn chế

- Domain shift giữa dataset công khai và ảnh thực địa.
- Background bias, class imbalance và annotation noise.
- Không có detection không đồng nghĩa với lá khỏe; hệ thống phải trả trạng thái chưa xác định khi confidence thấp.
- Cần external test set độc lập trước khi tuyên bố khả năng khái quát.

