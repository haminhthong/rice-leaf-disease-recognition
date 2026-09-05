# 💼 Portfolio Highlights & Interview Notes

Tài liệu này tổng hợp các câu tóm tắt điểm nhấn cho CV và bộ câu hỏi phỏng vấn chuyên sâu liên quan tới dự án **Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition)**.

---

## 🎯 CV Highlights (Dành cho CV / Portfolio)

### **Computer Vision & MLOps Engineer**
- Thiết kế và triển khai pipeline Object Detection (YOLOv8) phát hiện 2 lớp tổn thương lá lúa: **Bạc lá lúa (Bacterial Leaf Blight)** và **Đốm nâu (Brown Spot)**.
- Xử lý dữ liệu đa nguồn: Chuẩn hóa nhãn Polygon thành Bounding Box, xây dựng thuật toán lọc trùng SHA-256 kết hợp trích xuất pHash qua **BK-Tree + Union-Find** để nhóm các ảnh gần trùng.
- Thiết kế chiến lược chia dữ liệu **Group-aware Stratified Split** giúp giảm nguy cơ Data Leakage giữa tập huấn luyện và đánh giá.
- Đóng gói ứng dụng thành Python Package tích hợp REST API (**FastAPI**), Web Dashboard (**Streamlit**), container **Docker** có healthcheck và quy trình kiểm thử tự động **GitHub Actions CI**.

---

## ❓ Câu Hỏi Phỏng Vấn Chuyên Sâu & Gợi Ý Trả Lời

### 1. Vì sao không chia train/test ngẫu nhiên theo từng file ảnh?
**Trả lời:**
Ảnh lá lúa thu thập từ các nguồn mở thường chứa nhiều ảnh biến thể (augmentation, crop, resize) cắt từ cùng một bức ảnh gốc. Nếu phân chia ngẫu nhiên, các ảnh cùng gốc sẽ bị phân tán vào cả tập Train và Test, tạo ra Data Leakage nghiêm trọng và khiến chỉ số mAP cao giả tạo (~99%). Pipeline của dự án gom các ảnh trùng và gần trùng vào cùng một `group_id` trước khi chia split, đảm bảo toàn bộ biến thể của cùng một ảnh gốc luôn thuộc về cùng một tập dữ liệu.

### 2. Thuật toán BK-Tree và pHash hoạt động như thế nào trong bài toán này?
**Trả lời:**
- **pHash (Perceptual Hash)**: Trích xuất chuỗi băm 64-bit dựa trên biến đổi Cosin rời rạc (DCT-II) phản ánh các đặc trưng thị giác chính của ảnh.
- **BK-Tree (Burkhard-Keller Tree)**: Khi tìm kiếm các ảnh gần trùng trong tập $N$ ảnh, so sánh cặp $O(N^2)$ quá đắt đỏ. BK-Tree tổ chức chuỗi băm theo metric space và bất đẳng thức tam giác ($d(x,c) \le d(x,y) + d(y,c)$), giúp giảm đáng kể số phép so sánh khoảng cách Hamming trong nhiều phân bố dữ liệu thực tế.

### 3. Tại sao lại khóa tập Test và chỉ chọn mô hình bằng Validation set?
**Trả lời:**
Tập Test phải giữ vai trò là tập đánh giá độc lập hoàn toàn. Nếu dùng kết quả trên tập Test để điều chỉnh hyperparameter, chọn checkpoint hoặc tinh chỉnh ngưỡng confidence, tập Test đã bị lọt thông tin (data leakage vào quyết định thiết kế). Pipeline áp dụng quy trình: Huấn luyện baseline và ứng dụng -> Đánh giá trên Validation -> Chọn mô hình -> Khóa cấu hình -> Đánh giá tập Test đúng 1 lần duy nhất để báo cáo kết quả thực tế.

### 4. `status: no_detection` có ý nghĩa gì đối với bài toán thực tế?
**Trả lời:**
`no_detection` chỉ có nghĩa là hệ thống không tìm thấy vùng tổn thương nào vượt ngưỡng confidence đối với 2 loại bệnh thuộc phạm vi mô hình (Bạc lá lúa & Đốm nâu). Nó **không đồng nghĩa với việc lá cây khỏe mạnh** (lá có thể bị bệnh khác như Đạo ôn, Vàng lá, hoặc ảnh thiếu sáng/nhiễu). Trong sản phẩm thực tế, kết quả này được trả về kèm cảnh báo rõ ràng để người dùng không bị nhầm lẫn.
