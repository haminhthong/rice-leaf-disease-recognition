import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from rice_leaf_detection.inference import RiceLeafDetector

st.set_page_config(page_title="Nhận diện bệnh lá lúa", page_icon="🌾")
st.title("Nhận diện bệnh trên lá lúa")
st.caption("Mô hình phát hiện bạc lá và đốm nâu. Kết quả chỉ mang tính hỗ trợ.")

weights = Path(os.getenv("RICE_MODEL_PATH", "artifacts/best.pt"))
confidence = st.slider("Ngưỡng tin cậy", 0.05, 0.95, 0.25, 0.05)
uploaded = st.file_uploader("Chọn ảnh lá lúa", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    content = uploaded.getvalue()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Không đọc được ảnh đã chọn.")
    elif not weights.exists():
        st.error(f"Chưa có trọng số mô hình tại {weights}.")
    else:
        with st.spinner("Đang phân tích ảnh..."):
            detector = RiceLeafDetector(weights, confidence=confidence)
            prediction, result = detector.predict(image)
        if prediction.rejected:
            st.warning(prediction.reason)
        else:
            st.image(result.plot(), channels="BGR", caption="Kết quả phát hiện")
            for detection in prediction.detections:
                st.write(
                    f"{detection.class_name_vi}: {detection.confidence:.1%}"
                )

