"""Streamlit Web Dashboard ứng dụng Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Detection).

Giao diện trực quan cao cấp dành cho bài toán Computer Vision / Precision Agriculture:
1. 🎯 Chẩn đoán & Định vị: Upload ảnh hoặc chọn ảnh mẫu 1-click, xem Bounding Box & bảng phân tích.
2. 🌾 Khuyến cáo nông học: Biện pháp xử lý thực địa tùy theo mầm bệnh phát hiện được.
3. 📊 Dữ liệu & Chống rò rỉ: Thống kê tập dữ liệu, SHA-256 exact dedup và group-aware split.
4. 🔬 Phương pháp luận CV: Kiến trúc YOLOv8s, so sánh Detection vs Classification.
"""

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.settings import get_settings
from rice_leaf_detection.inference import RiceLeafDetector

logger = logging.getLogger(__name__)

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="RiceGuard AI | Nhận Diện Bệnh Lá Lúa",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện nông nghiệp thông minh cao cấp
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .hero-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        border: 1px solid rgba(16, 185, 129, 0.25);
        box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.3);
        margin-bottom: 1.5rem;
        color: #f0fdf4;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.35rem;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #a7f3d0;
        margin-bottom: 0.8rem;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .pill-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #ecfdf5;
    }

    .diag-card {
        border-radius: 0.875rem;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }

    .diag-success {
        background: #f0fdf4;
        border-color: #86efac;
        color: #166534;
    }

    .diag-warning {
        background: #fffbeb;
        border-color: #fcd34d;
        color: #92400e;
    }

    .diag-danger {
        background: #fef2f2;
        border-color: #fca5a5;
        color: #991b1b;
    }

    .advice-box {
        background: #f8fafc;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        border-left: 4px solid #10b981;
        margin-top: 0.75rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(
    weights_path: str, image_size: int, confidence: float, iou: float
) -> RiceLeafDetector:
    """Tái sử dụng detector singleton giữa các lượt suy luận trên Streamlit."""
    return RiceLeafDetector(
        Path(weights_path),
        image_size=image_size,
        confidence=confidence,
        iou=iou,
    )


# Banner tiêu đề chính
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">🌾 RiceGuard AI — Nhận Diện & Định Vị Bệnh Lá Lúa</div>
    <div class="hero-subtitle">
        Hệ thống Computer Vision phát hiện và định vị vùng tổn thương Bạc lá lúa
        (Bacterial Leaf Blight) và Đốm nâu (Brown Spot) dựa trên YOLOv8s.
    </div>
    <div class="badge-container">
        <span class="pill-badge">🎯 Object Detection (Lesion Localization)</span>
        <span class="pill-badge">⚡ YOLOv8s Architecture</span>
        <span class="pill-badge">🛡️ Leakage-Aware Split</span>
        <span class="pill-badge">🌱 Precision Agriculture</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Cấu hình thanh bên (Sidebar)
st.sidebar.markdown("### ⚙️ Tham Số Suy Luận")
runtime_settings = get_settings()

default_weights = os.getenv("RICE_MODEL_PATH", runtime_settings.weights.as_posix())
if not Path(default_weights).exists():
    candidate_weights = list(Path("runs/train").glob("**/weights/best.pt"))
    if candidate_weights:
        default_weights = str(candidate_weights[0])

weights_input = st.sidebar.text_input("Đường dẫn file trọng số (.pt):", default_weights)
weights_path = Path(weights_input)

if weights_path.exists():
    st.sidebar.caption("✅ File trọng số sẵn sàng")
else:
    st.sidebar.caption("⚠️ File trọng số chưa sẵn sàng (cần train)")

confidence = st.sidebar.slider(
    "Ngưỡng tin cậy (Confidence):",
    min_value=0.10,
    max_value=0.95,
    value=float(os.getenv("RICE_CONFIDENCE", str(runtime_settings.confidence))),
    step=0.05,
    help="Chỉ giữ lại các bounding box có độ tin cậy lớn hơn hoặc bằng ngưỡng này.",
)

iou = st.sidebar.slider(
    "Ngưỡng Non-Maximum Suppression (IoU):",
    min_value=0.10,
    max_value=0.90,
    value=float(os.getenv("RICE_IOU", str(runtime_settings.iou))),
    step=0.05,
    help="Loại bỏ các bounding box trùng lặp trên cùng một vùng tổn thương.",
)

image_size = int(os.getenv("RICE_IMAGE_SIZE", str(runtime_settings.image_size)))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌾 2 Mầm Bệnh Hỗ Trợ")
st.sidebar.markdown(
    """
- 🟡 **Bacterial Leaf Blight** (*Xanthomonas oryzae*): Bạc lá lúa. Vết sọc dọc theo gân lá,
  ban đầu xanh xám úng nước, sau chuyển vàng rơm đến bạc trắng.
- 🟤 **Brown Spot** (*Bipolaris oryzae*): Đốm nâu. Vết đốm hình bầu dục hoặc mắt tròn màu nâu,
  tâm màu xám nhạt, thường xuất hiện ở đất phèn/thiếu dinh dưỡng.
"""
)

st.sidebar.markdown("---")
st.sidebar.caption("Rice Leaf Disease Recognition v1.1.0 • YOLOv8 Engine")

# Các Tabs chính
tab_infer, tab_data, tab_about = st.tabs(
    [
        "🎯 Chẩn Đoán & Định Vị",
        "📊 Dữ Liệu & Chống Rò Rỉ",
        "🔬 Phương Pháp Luận & Model Card",
    ]
)

# ==============================================================================
# TAB 1: CHẨN ĐOÁN & ĐỊNH VỊ
# ==============================================================================
with tab_infer:
    col_input, col_view = st.columns([1, 1], gap="large")

    image_bgr = None
    source_name = ""

    with col_input:
        st.markdown("#### 📥 Chọn Nguồn Ảnh Đầu Vào")

        input_choice = st.radio(
            "Phương thức nạp ảnh:",
            ["🖼️ Dùng ảnh mẫu có sẵn", "📁 Tải ảnh từ thiết bị"],
            horizontal=True,
        )

        if input_choice == "🖼️ Dùng ảnh mẫu có sẵn":
            sample_dir = Path("data/sample")
            sample_blb = sample_dir / "bacterial_leaf_blight_sample.jpg"
            sample_bs = sample_dir / "brown_spot_sample.jpg"

            sample_options = {}
            if sample_blb.exists():
                sample_options["Mẫu 1: Bạc lá lúa (BLB)"] = sample_blb
            if sample_bs.exists():
                sample_options["Mẫu 2: Đốm nâu (Brown Spot)"] = sample_bs

            if sample_options:
                chosen_sample = st.selectbox(
                    "Chọn ảnh mẫu thực địa:",
                    list(sample_options.keys()),
                )
                chosen_path = sample_options[chosen_sample]
                image_bgr = cv2.imread(str(chosen_path))
                source_name = chosen_path.name
            else:
                st.info("Chưa tìm thấy ảnh trong `data/sample/`. Vui lòng dùng tải ảnh từ máy.")

        else:
            uploaded_file = st.file_uploader(
                "Tải lên ảnh lá lúa (định dạng JPG, PNG, WebP):",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                source_name = uploaded_file.name

        if image_bgr is not None:
            st.markdown("##### 🖼️ Ảnh Nguyên Bản:")
            st.image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Ảnh: {source_name} ({image_bgr.shape[1]}x{image_bgr.shape[0]}px)",
                use_container_width=True,
            )

    with col_view:
        st.markdown("#### 🔍 Kết Quả Phân Tích & Định Vị")

        if image_bgr is None:
            st.info("👈 Hãy chọn ảnh mẫu hoặc tải ảnh ở cột bên trái để bắt đầu phân tích.")
        elif not weights_path.exists():
            st.warning(
                f"⚠️ Chưa tìm thấy file trọng số mô hình tại `{weights_path}`.\n\n"
                "Bạn có thể huấn luyện mô hình bằng lệnh:\n"
                "```bash\n"
                "python scripts/train.py --config configs/default.yaml\n"
                "```"
            )
        else:
            start_time = time.perf_counter()
            with st.spinner("Đang chạy mô hình YOLOv8s định vị tổn thương..."):
                try:
                    detector = load_detector(str(weights_path), image_size, confidence, iou)
                    prediction, result = detector.predict(
                        image_bgr,
                        confidence=confidence,
                        iou=iou,
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Thống kê số lượng tổn thương
                    blb_detections = [d for d in prediction.detections if d.class_id == 0]
                    bs_detections = [d for d in prediction.detections if d.class_id == 1]
                    total_count = len(prediction.detections)

                    # KPI Cards
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Bạc Lá Lúa", f"{len(blb_detections)} vùng")
                    kpi2.metric("Đốm Nâu", f"{len(bs_detections)} vùng")
                    max_conf = (
                        max((d.confidence for d in prediction.detections), default=0.0)
                        if total_count > 0
                        else 0.0
                    )
                    kpi3.metric("Tin Cậy Cao", f"{max_conf:.1%}")
                    kpi4.metric("Thời Gian", f"{latency_ms:.1f} ms")

                    # Banner chẩn đoán
                    if total_count == 0:
                        st.markdown(
                            """
                        <div class="diag-card diag-success">
                            <strong>✅ KHÔNG PHÁT HIỆN TRIỆU CHỨNG BỆNH</strong><br>
                            Không phát hiện vùng tổn thương nào vượt qua ngưỡng tin cậy.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif len(blb_detections) > 0 and len(bs_detections) > 0:
                        st.markdown(
                            f"""
                        <div class="diag-card diag-danger">
                            <strong>⚠️ NHIỄM ĐỒNG THỜI 2 LOẠI BỆNH</strong><br>
                            Phát hiện {len(blb_detections)} tổn thương Bạc lá và
                            {len(bs_detections)} tổn thương Đốm nâu.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif len(blb_detections) > 0:
                        st.markdown(
                            f"""
                        <div class="diag-card diag-warning">
                            <strong>🟡 PHÁT HIỆN BỆNH BẠC LÁ LÚA (BLB)</strong><br>
                            Định vị được {len(blb_detections)} vùng tổn thương do vi khuẩn
                            <i>Xanthomonas oryzae</i>.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                        <div class="diag-card diag-warning">
                            <strong>🟤 PHÁT HIỆN BỆNH ĐỐM NÂU (Brown Spot)</strong><br>
                            Định vị được {len(bs_detections)} vết đốm nâu do nấm
                            <i>Bipolaris oryzae</i>.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    # Hiển thị ảnh có bounding box
                    annotated_bgr = result.plot()
                    st.image(
                        cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                        caption="Ảnh định vị tổn thương với Bounding Box",
                        use_container_width=True,
                    )

                    # Bảng chi tiết từng vùng
                    if total_count > 0:
                        st.markdown("##### 📋 Chi Tiết Bounding Box:")
                        img_h, img_w = image_bgr.shape[:2]
                        total_pixels = img_h * img_w

                        rows = []
                        for idx, det in enumerate(prediction.detections, 1):
                            x1, y1, x2, y2 = det.box_xyxy
                            box_w = max(0.0, x2 - x1)
                            box_h = max(0.0, y2 - y1)
                            area_pct = (box_w * box_h) / total_pixels * 100

                            rows.append(
                                {
                                    "#": idx,
                                    "Bệnh": det.class_name_vi,
                                    "Tên Tiếng Anh": det.class_name,
                                    "Tin Cậy": f"{det.confidence:.2%}",
                                    "Tọa Độ [x1, y1, x2, y2]": (
                                        f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
                                    ),
                                    "Diện Tích (% Lá)": f"{area_pct:.2f}%",
                                }
                            )
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    # Khuyến cáo nông học thực tiễn
                    st.markdown("##### 💡 Khuyến Cáo Kỹ Thuật Nông Nghiệp:")
                    if len(blb_detections) > 0:
                        st.markdown(
                            """
                        <div class="advice-box">
                            <strong>🌾 Đối với Bệnh Bạc Lá Lúa:</strong>
                            <ul>
                                <li><strong>Phân bón:</strong> Ngưng ngay bón thừa đạm (N);
                                bổ sung Kali (K) và Silic để tăng độ dày vách tế bào lá.</li>
                                <li><strong>Quản lý nước:</strong> Tháo bớt nước trong ruộng,
                                giữ ruộng thông thoáng, tránh vi khuẩn lây lan qua giọt dịch.</li>
                                <li><strong>Tiếp xúc:</strong> Tránh lội ruộng khi lá còn ướt
                                sương sớm nhằm ngăn ngừa phát tán vi khuẩn cơ học.</li>
                            </ul>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    if len(bs_detections) > 0:
                        st.markdown(
                            """
                        <div class="advice-box">
                            <strong>🌾 Đối với Bệnh Đốm Nâu:</strong>
                            <ul>
                                <li><strong>Cải tạo đất:</strong> Đốm nâu chỉ thị đất nghèo dinh
                                dưỡng hoặc ngộ độc phèn; cần bón vôi hạ phèn, bổ sung lân, kẽm.</li>
                                <li><strong>Chế độ nước:</strong> Duy trì mực nước nông ổn định,
                                tránh để ruộng khô nứt trong giai đoạn đẻ nhánh và làm đòng.</li>
                                <li><strong>Vụ tới:</strong> Xử lý hạt giống bằng nước ấm hoặc
                                dung dịch sát khuẩn trước khi gieo xạ.</li>
                            </ul>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    if total_count == 0:
                        st.caption(
                            "Không có khuyến cáo đặc biệt. Duy trì chế độ canh tác tiêu chuẩn."
                        )

                except Exception as exc:
                    logger.exception("Lỗi khi suy luận trên Streamlit")
                    st.error(f"Đã xảy ra lỗi khi phân tích ảnh: {exc}")

# ==============================================================================
# TAB 2: DỮ LIỆU & CHỐNG RÒ RỈ
# ==============================================================================
with tab_data:
    st.markdown("### 📊 Quản Trị Dữ Liệu & Chống Rò Rỉ (Data Quality & Anti-Leakage)")
    st.markdown(
        """
    Trong bài toán Computer Vision cho nông nghiệp, việc **chống rò rỉ dữ liệu (data leakage)**
    giữa các tập Train, Validation và Test đóng vai trò sống còn để mô hình không bị overfit ảo.
    """
    )

    dataset_dir = Path("data/processed/rice_leaf_detection")
    manifest_path = dataset_dir / "manifest.csv"
    report_path = dataset_dir / "data_report.json"

    if manifest_path.exists() and report_path.exists():
        df_manifest = pd.read_csv(manifest_path)
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report_data.get("summary", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng Ảnh Hợp Lệ", len(df_manifest))
        m2.metric("Ảnh Tập Train", len(df_manifest[df_manifest["split"] == "train"]))
        m3.metric("Ảnh Tập Val", len(df_manifest[df_manifest["split"] == "val"]))
        m4.metric("Ảnh Tập Test", len(df_manifest[df_manifest["split"] == "test"]))

        st.markdown("#### 🛡️ Thống Kê Làm Sạch & Khử Trùng Lặp")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Ảnh Trùng Tuyệt Đối (SHA-256)",
            summary.get("exact_duplicates_removed", 0),
            help="Loại bỏ ảnh copy giống hệt nhau để không rơi vào 2 split khác nhau.",
        )
        c2.metric(
            "Nhóm Ảnh Tương Tự (pHash)",
            summary.get("near_duplicate_links", 0),
            help="Gom cụm bằng perceptual hash Hamming distance để giữ nguyên nhóm trong 1 split.",
        )
        c3.metric(
            "Nhãn Lỗi/Thiếu Đã Loại",
            summary.get("invalid_or_missing_annotations_discarded", 0),
            help="Loại bỏ nhãn rỗng/lỗi; tuân thủ: missing annotation ≠ negative image.",
        )

        st.markdown("#### 📌 Phân Bố Tổn Thương Theo Split:")
        split_summary = df_manifest.groupby("split").agg(
            So_Luong_Anh=("output_image", "count"),
            Anh_Negative_Nen=("is_negative", "sum"),
            Vung_Bac_La_Lua=("instances_class_0", "sum"),
            Vung_Dom_Nau=("instances_class_1", "sum"),
        )
        st.dataframe(split_summary, use_container_width=True)

        with st.expander("📄 Khám phá 20 dòng đầu tiên của manifest.csv"):
            st.dataframe(df_manifest.head(20), use_container_width=True)

    else:
        st.info(
            "Chưa tìm thấy dữ liệu đã xử lý tại `data/processed/rice_leaf_detection/`.\n\n"
            "Chạy lệnh sau để chuẩn bị dữ liệu và xuất báo cáo:\n"
            "```bash\n"
            "python scripts/prepare_data.py --raw-dir data/raw --output-dir data/processed\n"
            "```"
        )

# ==============================================================================
# TAB 3: PHƯƠNG PHÁP LUẬN & MODEL CARD
# ==============================================================================
with tab_about:
    st.markdown("### 🔬 Phương Pháp Luận Computer Vision & Giới Hạn Mô Hình")

    col_meth1, col_meth2 = st.columns([1, 1], gap="large")

    with col_meth1:
        st.markdown("#### 🎯 Object Detection vs. Image Classification")
        st.markdown(
            """
        - **Phân loại ảnh thông thường (Classification):** Chỉ trả về một nhãn cho cả bức ảnh.
          Điều này không đủ trong nông nghiệp vì một lá có thể vừa có đốm nâu vừa có bạc lá,
          hoặc tổn thương chỉ chiếm diện tích nhỏ ở chóp lá.
        - **Phát hiện tổn thương (Object Detection):** YOLOv8s định vị chính xác vị trí,
          diện tích và loại tổn thương trên từng vùng lá, giúp định lượng mức độ nghiêm trọng.
        """
        )

        st.markdown("#### ⚙️ Cấu Hình Huấn Luyện")
        st.markdown(
            """
        - **Mô hình gốc:** YOLOv8s pretrained COCO.
        - **Kích thước ảnh:** 640x640 pixels.
        - **Hàm tối ưu:** AdamW, learning rate 0.001.
        - **Data Augmentation (Train-only):** HSV, xoay nhẹ, dịch chuyển, tỷ lệ, lật ngang.
          **Val và Test tắt toàn bộ augmentation.**
        """
        )

    with col_meth2:
        st.markdown("#### 🛡️ Giới Hạn & Đạo Đức AI (Model Card)")
        st.markdown(
            """
        - **Domain Shift:** Mô hình được huấn luyện trên điều kiện ánh sáng tự nhiên.
          Nếu chụp ảnh trong phòng lab có đèn flash gắt hoặc chụp quá xa, độ chính xác có thể giảm.
        - **Tổn thương quá nhỏ:** Các vết chấm đốm nâu mới dưới 10px có thể bị bỏ sót.
        - **Khuyến cáo an toàn:** Ứng dụng mang tính chất hỗ trợ trinh sát thực địa.
          Mọi quyết định phun thuốc BVTV cần có hướng dẫn của cán bộ nông nghiệp.
        """
        )

    st.markdown("---")
    st.markdown("#### 📋 Luồng Xử Lý 6 Bước Tinh Gọn (Pipeline Architecture):")
    st.code(
        """
1. Ảnh lá lúa thô kèm nhãn (Bounding Box / Polygon)
         │
         ▼
2. Kiểm tra chất lượng nhãn (Valid / Negative / Loại bỏ nhãn lỗi)
         │
         ▼
3. Khử trùng tuyệt đối (SHA-256) & Gom nhóm biến thể (pHash Hamming)
         │
         ▼
4. Huấn luyện YOLOv8s (Augmentation bật trên Train, tắt trên Val/Test)
         │
         ▼
5. Đánh giá khách quan (Precision, Recall, mAP@0.5, mAP@0.5:0.95)
         │
         ▼
6. Phân tích lỗi (TP/FP/FN/Kích thước) & Phục vụ (FastAPI / Streamlit)
""",
        language="text",
    )
