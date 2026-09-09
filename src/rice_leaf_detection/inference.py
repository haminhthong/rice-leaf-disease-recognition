"""Detector và decision policy dùng chung cho API, CLI và Streamlit.

Detector chỉ lấy candidate từ YOLO. Policy mới quyết định candidate nào được
chấp nhận hoặc cần người kiểm tra; các ngưỡng không còn nằm rải rác trong
logic suy luận.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .constants import CLASS_NAMES, CLASS_NAMES_VI
from .utils import sha256_file

Decision = Literal["accepted", "review"]
PredictionStatus = Literal[
    "DETECTED",
    "REVIEW_REQUIRED",
    "NO_SUPPORTED_SYMPTOM_DETECTED",
]


@dataclass(frozen=True)
class RawDetection:
    """Candidate thô từ model, chưa áp dụng quyết định nghiệp vụ."""

    class_id: int
    class_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, init=False)
class Detection:
    """Detection sau policy, chỉ lưu một điểm số duy nhất là ``score``."""

    class_id: int
    class_name: str
    class_name_vi: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    decision: Decision

    def __init__(
        self,
        class_id: int,
        class_name: str,
        class_name_vi: str,
        box_xyxy: tuple[float, float, float, float],
        score: float | None = None,
        decision: Decision = "accepted",
        confidence: float | None = None,
    ) -> None:
        # confidence là alias constructor cho code cũ; object chỉ lưu score.
        resolved_score = score if score is not None else confidence
        if resolved_score is None:
            raise ValueError("Detection cần score")
        if not 0 <= resolved_score <= 1:
            raise ValueError("Detection score phải nằm trong khoảng [0, 1]")
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "class_name_vi", class_name_vi)
        object.__setattr__(self, "score", float(resolved_score))
        object.__setattr__(self, "box_xyxy", box_xyxy)
        object.__setattr__(self, "decision", decision)

    @property
    def confidence(self) -> float:
        """Alias đọc tương thích; schema mới dùng ``score``."""
        return self.score


@dataclass(frozen=True)
class DetectionPolicy:
    """Chính sách hai ngưỡng được chọn trên Validation."""

    review_threshold: float = 0.20
    accept_threshold: float = 0.45

    def __post_init__(self) -> None:
        if not 0 <= self.review_threshold <= 1:
            raise ValueError("Ngưỡng review phải nằm trong khoảng [0, 1]")
        if not 0 <= self.accept_threshold <= 1:
            raise ValueError("Ngưỡng accept phải nằm trong khoảng [0, 1]")
        if self.review_threshold > self.accept_threshold:
            raise ValueError("Ngưỡng review không được lớn hơn ngưỡng accept")

    def decision_for(self, score: float) -> Decision | None:
        """Trả về accepted/review hoặc None nếu candidate phải loại bỏ."""
        if score >= self.accept_threshold:
            return "accepted"
        if score >= self.review_threshold:
            return "review"
        return None

    def apply(self, candidates: list[RawDetection]) -> list[Detection]:
        """Áp policy và chỉ trả các candidate accepted/review."""
        detections: list[Detection] = []
        for candidate in candidates:
            decision = self.decision_for(candidate.score)
            if decision is None:
                continue
            detections.append(
                Detection(
                    class_id=candidate.class_id,
                    class_name=candidate.class_name,
                    class_name_vi=CLASS_NAMES_VI.get(candidate.class_id, candidate.class_name),
                    score=candidate.score,
                    box_xyxy=candidate.box_xyxy,
                    decision=decision,
                )
            )
        return detections


def load_detection_policy(
    policy_path: Path | str,
    fallback: DetectionPolicy,
) -> DetectionPolicy:
    """Đọc policy cạnh artifact; dùng fallback khi model chưa được export."""
    path = Path(policy_path)
    if not path.exists():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DetectionPolicy(
            review_threshold=float(payload["review_threshold"]),
            accept_threshold=float(payload["accept_threshold"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Policy artifact không hợp lệ: {path}") from exc


@dataclass(frozen=True)
class ImageSummary:
    """Tóm tắt phân tích mức ảnh phục vụ hỗ trợ trinh sát đồng ruộng (Decision Support)."""

    bacterial_leaf_blight_detected: bool = False
    brown_spot_detected: bool = False
    total_detections: int = 0
    requires_human_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    accepted_detections: int = 0
    review_detections: int = 0


@dataclass(frozen=True)
class Prediction:
    """Kết quả dự đoán tổng hợp cho một hình ảnh."""

    detections: list[Detection]
    status: PredictionStatus
    message: str
    warnings: list[str] = field(default_factory=list)
    image_summary: ImageSummary | None = None
    image_quality: dict[str, Any] = field(default_factory=dict)


class RiceLeafDetector:
    """YOLO detector độc lập với policy quyết định nghiệp vụ."""

    def __init__(
        self,
        weights: Path | str,
        image_size: int = 640,
        confidence: float = 0.20,
        iou: float = 0.45,
        policy: DetectionPolicy | None = None,
        review_threshold: float | None = None,
        accept_threshold: float | None = None,
    ):
        if image_size <= 0:
            raise ValueError("Kích thước ảnh phải lớn hơn 0")
        if not 0 <= confidence <= 1:
            raise ValueError("Ngưỡng tin cậy phải nằm trong khoảng [0, 1]")
        if not 0 <= iou <= 1:
            raise ValueError("Ngưỡng IoU phải nằm trong khoảng [0, 1]")
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Không tìm thấy trọng số: {weights_path}")
        self._verify_artifact_contract(weights_path, image_size)

        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.image_size = image_size
        self.candidate_confidence = confidence
        self.iou = iou
        self.policy = policy or DetectionPolicy(
            review_threshold=confidence if review_threshold is None else review_threshold,
            accept_threshold=(
                max(confidence, 0.45) if accept_threshold is None else accept_threshold
            ),
        )
        self._verify_policy_contract(weights_path)

    def _verify_artifact_contract(self, weights_path: Path, image_size: int) -> None:
        """Fail fast nếu release có metadata nhưng model không khớp metadata."""
        metadata_path = weights_path.parent / "model_metadata.json"
        policy_path = weights_path.parent / "detection_policy.json"
        if not metadata_path.exists() and not policy_path.exists():
            return
        if not metadata_path.exists() or not policy_path.exists():
            raise ValueError(
                "Artifact release thiếu model_metadata.json hoặc detection_policy.json"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_classes = metadata.get("classes")
            if expected_classes != CLASS_NAMES:
                raise ValueError("Artifact không khớp danh sách lớp mục tiêu")
            if int(metadata.get("image_size")) != image_size:
                raise ValueError("Artifact không khớp image_size của runtime")
            expected_hash = metadata.get("weights_sha256")
            if expected_hash and sha256_file(weights_path) != expected_hash:
                raise ValueError("Checksum trọng số không khớp model_metadata.json")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Artifact contract không hợp lệ: {exc}") from exc

    def _verify_policy_contract(self, weights_path: Path) -> None:
        """Đảm bảo policy runtime không lệch policy đã đóng gói cùng model."""
        policy_path = weights_path.parent / "detection_policy.json"
        if not policy_path.exists():
            return
        policy_data = json.loads(policy_path.read_text(encoding="utf-8"))
        expected = DetectionPolicy(
            review_threshold=float(policy_data["review_threshold"]),
            accept_threshold=float(policy_data["accept_threshold"]),
        )
        if expected != self.policy:
            raise ValueError("Policy runtime không khớp detection_policy.json của model")

    @property
    def confidence(self) -> float:
        """Tên cũ cho ngưỡng candidate thấp nhất."""
        return self.candidate_confidence

    def predict(
        self,
        source: object,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> tuple[Prediction, Any]:
        """Thực hiện phát hiện bệnh trên nguồn ảnh đầu vào (Path, numpy ndarray hoặc PIL Image).

        Args:
            source: Nguồn ảnh đầu vào.
            confidence: Ngưỡng tin cậy riêng cho lượt gọi hiện tại.
            iou: Ngưỡng IoU riêng cho lượt gọi hiện tại.

        Returns:
            tuple[Prediction, Any]:
                - Object `Prediction` đã được cấu trúc hóa kèm `ImageSummary`.
                - Đối tượng `Results` gốc của Ultralytics (phục vụ vẽ bounding box `.plot()`).
        """
        candidate_confidence = self.candidate_confidence if confidence is None else confidence
        nms_iou = self.iou if iou is None else iou
        if not 0 <= candidate_confidence <= 1:
            raise ValueError("Ngưỡng tin cậy phải nằm trong khoảng [0, 1]")
        if not 0 <= nms_iou <= 1:
            raise ValueError("Ngưỡng IoU phải nằm trong khoảng [0, 1]")

        # Lấy từ review threshold để không làm mất candidate borderline trước policy.
        model_confidence = min(candidate_confidence, self.policy.review_threshold)
        result = self.model.predict(
            source=source,
            imgsz=self.image_size,
            conf=model_confidence,
            iou=nms_iou,
            verbose=False,
        )[0]
        candidates: list[RawDetection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(self.model.names[class_id])
            candidates.append(
                RawDetection(
                    class_id=class_id,
                    class_name=name,
                    score=float(box.conf[0]),
                    box_xyxy=tuple(float(value) for value in box.xyxy[0].tolist()),
                )
            )

        detections = self.policy.apply(candidates)
        accepted = [d for d in detections if d.decision == "accepted"]
        reviews = [d for d in detections if d.decision == "review"]
        review_reasons: list[str] = []
        if reviews:
            review_reasons.append(
                f"Có {len(reviews)} candidate trong vùng review "
                f"[{self.policy.review_threshold:.2f}, {self.policy.accept_threshold:.2f}), "
                "cần chuyên gia kiểm tra trực quan."
            )

        # Hai box khác lớp chồng lấn mạnh thường là dấu hiệu nhầm triệu chứng.
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if detections[i].class_id != detections[j].class_id:
                    bi, bj = detections[i].box_xyxy, detections[j].box_xyxy
                    if _box_iou(bi, bj) >= 0.30:
                        review_reasons.append(
                            "Các hộp tổn thương khác lớp chồng lấn mạnh; "
                            "cần kiểm tra lại triệu chứng."
                        )
                        break

        image_summary = ImageSummary(
            bacterial_leaf_blight_detected=any(d.class_id == 0 for d in accepted),
            brown_spot_detected=any(d.class_id == 1 for d in accepted),
            total_detections=len(detections),
            requires_human_review=bool(review_reasons),
            review_reasons=review_reasons,
            accepted_detections=len(accepted),
            review_detections=len(reviews),
        )

        domain_warnings = [
            "Hệ thống chỉ hỗ trợ trinh sát đồng ruộng, không phải chẩn đoán cuối cùng.",
            "Không tự động phun thuốc/hóa chất khi chưa có chỉ dẫn từ kỹ sư nông nghiệp.",
        ]

        if accepted:
            status: PredictionStatus = "DETECTED"
            message = f"Phát hiện {len(accepted)} vùng tổn thương đạt ngưỡng chấp nhận."
            warnings = domain_warnings
        elif reviews:
            status = "REVIEW_REQUIRED"
            message = "Có candidate ở vùng ranh giới; cần người kiểm tra trước khi kết luận."
            warnings = [*review_reasons, *domain_warnings]
        else:
            status = "NO_SUPPORTED_SYMPTOM_DETECTED"
            message = "Không phát hiện vùng tổn thương thuộc 2 lớp hỗ trợ ở ngưỡng đã chọn."
            warnings = [
                "Kết quả không khẳng định lá cây khỏe mạnh.",
                "Ảnh có thể thuộc bệnh ngoài phạm vi hỗ trợ hoặc không đạt chất lượng.",
                *domain_warnings,
            ]

        prediction = Prediction(
            detections=detections,
            status=status,
            message=message,
            warnings=warnings,
            image_summary=image_summary,
        )
        return prediction, result


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Tính IoU nội bộ cho kiểm tra chồng lấn giữa candidate khác lớp."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
