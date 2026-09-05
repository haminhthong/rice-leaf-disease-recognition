from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from rice_leaf_detection.export import verify_prediction_parity


class DummyBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = torch.tensor(cls)
        self.conf = torch.tensor(conf)
        self.xyxy = torch.tensor(xyxy)

    def __len__(self):
        return len(self.cls)


class DummyResult:
    def __init__(self, boxes):
        self.boxes = boxes


@patch("ultralytics.YOLO")
def test_export_prediction_parity_dat_chuan(mock_yolo: MagicMock, tmp_path: Path) -> None:
    """Kiểm tra parity check đạt khi kết quả suy luận PyTorch và Exported khớp trong dung sai."""
    mock_pt = MagicMock()
    mock_exp = MagicMock()

    # Cả 2 model trả về box giống nhau, sai khác confidence cực nhỏ (0.01 < 0.05)
    boxes_pt = DummyBoxes(cls=[0], conf=[0.90], xyxy=[[10.0, 10.0, 50.0, 50.0]])
    boxes_exp = DummyBoxes(cls=[0], conf=[0.89], xyxy=[[10.1, 10.1, 50.1, 50.1]])

    mock_pt.predict.return_value = [DummyResult(boxes_pt)]
    mock_exp.predict.return_value = [DummyResult(boxes_exp)]

    mock_yolo.side_effect = [mock_pt, mock_exp]

    dummy_img = tmp_path / "leaf.jpg"
    dummy_img.touch()

    summary = verify_prediction_parity(
        pytorch_model_path="best.pt",
        exported_model_path="best.onnx",
        sample_images=[dummy_img],
        conf_tolerance=0.05,
        min_box_iou=0.85,
    )

    assert summary["parity_passed"] is True
    assert summary["max_conf_diff"] <= 0.05
    assert summary["min_box_iou"] >= 0.85
    assert len(summary["mismatches"]) == 0


@patch("ultralytics.YOLO")
def test_export_prediction_parity_that_bai_khi_sai_class(
    mock_yolo: MagicMock, tmp_path: Path
) -> None:
    """Kiểm tra parity check phát hiện lỗi khi class id hoặc số lượng box không khớp."""
    mock_pt = MagicMock()
    mock_exp = MagicMock()

    boxes_pt = DummyBoxes(cls=[0], conf=[0.90], xyxy=[[10.0, 10.0, 50.0, 50.0]])
    boxes_exp = DummyBoxes(cls=[1], conf=[0.90], xyxy=[[10.0, 10.0, 50.0, 50.0]])  # Khác class

    mock_pt.predict.return_value = [DummyResult(boxes_pt)]
    mock_exp.predict.return_value = [DummyResult(boxes_exp)]

    mock_yolo.side_effect = [mock_pt, mock_exp]

    dummy_img = tmp_path / "leaf.jpg"
    dummy_img.touch()

    summary = verify_prediction_parity(
        pytorch_model_path="best.pt",
        exported_model_path="best.onnx",
        sample_images=[dummy_img],
    )

    assert summary["parity_passed"] is False
    assert len(summary["mismatches"]) > 0
