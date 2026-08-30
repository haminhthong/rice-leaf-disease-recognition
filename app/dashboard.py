"""Streamlit Web Dashboard ứng dụng Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition).

Giao diện tương tác sinh động bao gồm 3 Tabs chính:
1. 🎯 **Phân tích Ảnh (Inference)**: Upload ảnh, tùy chỉnh Confidence & IoU, xem Bounding Box.
2. 📊 **Audit & Thống kê Dữ liệu**: Xem báo cáo audit_report.json và manifest dữ liệu.
3. ℹ️ **Kiến trúc Pipeline**: Sơ đồ pipeline MLOps, giao thức chống rò rỉ và Model Card.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from rice_leaf_detection.inference import RiceLeafDetector

# Cấu hình trang Dashboard Streamlit
st.set_page_config(
    page_title="Nhận Diện Bệnh Lá Lúa | Rice Leaf Disease Recognition",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌾 Hệ Thống Nhận Diện Bệnh Trên Lá Lúa")
st.caption(
    "Ứng dụng Computer Vision & MLOps phát hiện Bạc lá lúa "
    "(Bacterial Leaf Blight) và Đốm nâu (Brown Spot)."
)


# Sidebar cấu hình tham số
st.sidebar.header("⚙️ Cấu Hình Mô Hình")
weights_path_env = os.getenv("RICE_MODEL_PATH", "artifacts/best.pt")
weights = Path(st.sidebar.text_input("Trọng số mô hình (.pt):", weights_path_env))

confidence = st.sidebar.slider("Ngưỡng tin cậy (Confidence Threshold):", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("Ngưỡng NMS IoU (IoU Threshold):", 0.10, 0.90, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Lưu ý:** Sản phẩm hỗ trợ sàng lọc hình ảnh và minh họa quy trình MLOps. "
    "Kết quả không thay thế chẩn đoán chuyên môn của chuyên gia nông nghiệp."
)

# Phân chia 3 Tab giao diện chính
tab_infer, tab_audit, tab_about = st.tabs([
    "🎯 Phân Tích Ảnh",
    "📊 Thống Kê Dữ Liệu & Audit",
    "ℹ️ Kiến Trúc & Protocol",
])

# ---------------------------------------------------------
# TAB 1: PHÂN TÍCH ẢNH (INFERENCE)
# ---------------------------------------------------------
with tab_infer:
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📤 Upload Ảnh Lá Lúa")
        uploaded_file = st.file_uploader(
            "Chọn file ảnh cần phân tích (hỗ trợ JPG, PNG, WebP):",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file is not None:
            content = uploaded_file.getvalue()
            image_np = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)

            if image_np is None:
                st.error("❌ Không đọc được nội dung ảnh đã chọn. Vui lòng thử lại file khác.")
            else:
                st.image(
                    cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB),
                    caption="Ảnh gốc tải lên",
                    use_container_width=True,
                )

    with col_result:
        st.subheader("🔍 Kết Quả Phân Tích")

        if uploaded_file is None:
            st.info("👈 Vui lòng upload file ảnh lá lúa ở cột bên trái để xem kết quả phân tích.")
        elif image_np is not None:
            if not weights.exists():
                st.warning(
                    f"⚠️ Chưa tìm thấy file trọng số mô hình tại `{weights}`.\n\n"
                    "Hãy chạy huấn luyện hoặc đặt file `best.pt` vào thư mục `artifacts/`."
                )
            else:
                with st.spinner("Đang chạy mô hình YOLOv8 phân tích..."):
                    try:
                        detector = RiceLeafDetector(weights, confidence=confidence, iou=iou)
                        prediction, result = detector.predict(image_np)

                        if prediction.rejected:
                            st.warning(f"⚠️ {prediction.reason}")
                        else:
                            # Hiển thị ảnh kèm Bounding Box
                            annotated_image = result.plot()
                            st.image(
                                cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                                caption="Ảnh phát hiện bệnh (YOLOv8 Bounding Boxes)",
                                use_container_width=True,
                            )

                            st.markdown("#### 📋 Danh sách vùng tổn thương phát hiện:")
                            detection_rows = []
                            for idx, det in enumerate(prediction.detections, 1):
                                detection_rows.append({
                                    "STT": idx,
                                    "Lớp Bệnh (Tiếng Việt)": det.class_name_vi,
                                    "Tên Tiếng Anh": det.class_name,
                                    "Độ Tin Cậy (Confidence)": f"{det.confidence:.1%}",
                                    "Tọa Độ (x1, y1, x2, y2)": (
                                        f"({det.box_xyxy[0]:.1f}, {det.box_xyxy[1]:.1f}, "
                                        f"{det.box_xyxy[2]:.1f}, {det.box_xyxy[3]:.1f})"
                                    ),
                                })
                            st.dataframe(pd.DataFrame(detection_rows), use_container_width=True)

                    except Exception as exc:
                        st.error(f"Lỗi trong quá trình xử lý ảnh: {exc}")

# ---------------------------------------------------------
# TAB 2: THỐNG KÊ DỮ LIỆU & AUDIT
# ---------------------------------------------------------
with tab_audit:
    st.subheader("📊 Kiểm Toán Bộ Dữ Liệu (Data Audit & Manifest)")
    
    processed_dir = Path("data/processed/rice_leaf_detection")
    manifest_path = processed_dir / "manifest.csv"
    audit_path = processed_dir / "audit_report.json"

    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Tổng Số Ảnh Sạch", len(manifest_df))
        col_m2.metric("Số Ảnh Tập Train", len(manifest_df[manifest_df["split"] == "train"]))
        col_m3.metric("Số Ảnh Tập Val", len(manifest_df[manifest_df["split"] == "val"]))
        col_m4.metric("Số Ảnh Tập Test", len(manifest_df[manifest_df["split"] == "test"]))

        st.markdown("#### 📌 Phân Phổi Dữ Liệu Theo Split & Nguồn Dữ Liệu")
        summary_df = manifest_df.groupby("split").agg(
            Tong_So_Anh=("output_image", "count"),
            Negative_Samples=("is_negative", "sum"),
            Bac_La_Lua_Boxes=("instances_class_0", "sum"),
            Dom_Nau_Boxes=("instances_class_1", "sum"),
        )
        st.dataframe(summary_df, use_container_width=True)

        with st.expander("📄 Xem Chi Tiết 20 Dòng Manifest Đầu Tiên"):
            st.dataframe(manifest_df.head(20), use_container_width=True)
    else:
        st.info(
            "Chưa tìm thấy dataset sạch đã xử lý tại "
            "`data/processed/rice_leaf_detection/manifest.csv`. "
            "Hãy chạy lệnh `rice-prepare`."
        )

    if audit_path.exists():
        st.markdown("#### 📄 Báo Cáo Kiểm Toán Dữ Liệu (Audit Report JSON)")
        audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
        
        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric("Ảnh Trùng SHA-256 Đã Loại", audit_data.get("exact_duplicates_removed", 0))
        col_a2.metric("Liên Kết pHash Gần Trùng", audit_data.get("near_duplicate_links", 0))
        col_a3.metric("Số Khung Bbox Đã Clip", audit_data.get("clipped_boxes", 0))

        with st.expander("🔍 Xem Toàn Bộ Audit Report JSON"):
            st.json(audit_data)

# ---------------------------------------------------------
# TAB 3: KIẾN TRÚC & PROTOCOL
# ---------------------------------------------------------
with tab_about:
    st.subheader("ℹ️ Kiến Trúc Hệ Thống & ML Protocol")

    st.markdown("""
    ### 🛡️ Giải Pháp Chống Rò Rỉ Dữ Liệu (Anti-Data Leakage)
    1. **Taxonomy Normalization**: Ánh xạ đa nguồn dữ liệu về 2 lớp mục tiêu chính:
       - **Bacterial Leaf Blight** (Bạc lá lúa)
       - **Brown Spot** (Đốm nâu)
    2. **pHash + BK-Tree Grouping**: Nhóm các ảnh biến thể gần giống nhau (do crop/augment)
       vào cùng một `group_id`.
    3. **Group-aware Stratified Split**: Phân chia tập dữ liệu Train (70%), Val (15%), Test (15%)
       theo `group_id` nhằm đảm bảo ảnh biến thể không xuất hiện ở cả tập Train và Test.
    4. **Val-only Selection & Test Lock**: Mô hình được lựa chọn bằng Validation set (`mAP50-95`).
       Tập Test chỉ được mở khóa đánh giá một lần duy nhất với cờ `--confirm-final-test`.


    ### 🏗️ Sơ Đồ Pipeline Trực Quan

    ```text
    Hai dataset ZIP nén nguồn
          │
          ▼
    Giải nén an toàn ──► Chuẩn hóa nhãn Polygon/BBox ──► Lọc nhãn lỗi
          │
          ▼
    SHA-256 Dedup ──► pHash BK-Tree Grouping ──► Group-aware Split
          │
          ▼
    YOLOv8 Training ──► Val-only Model Selection ──► Test Evaluation (1 Lần)
          │
          ├──► RESTful FastAPI (/predict, /health, /info)
          ├──► Streamlit Dashboard UI
          └──► Export ONNX / OpenVINO / TorchScript
    ```
    """)
