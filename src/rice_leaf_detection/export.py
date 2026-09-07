"""Công cụ xuất mô hình sang các định dạng triển khai thực tế (Model Export Tool).

Hỗ trợ chuyển đổi trọng số huấn luyện PyTorch (`.pt`) sang các định dạng chuẩn công nghiệp:
- ONNX (`.onnx`)
- OpenVINO (`.xml`/`.bin`)
- TorchScript (`.torchscript`)

Mỗi lần export đều tạo file `metadata.json` lưu vết mã checksum SHA-256
của trọng số gốc và artifact mới xuất.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .utils import configure_utf8_console, sha256_file


def verify_prediction_parity(
    pytorch_model_path: Path | str,
    exported_model_path: Path | str,
    sample_images: list[Path | str],
    imgsz: int = 640,
    conf_tolerance: float = 0.05,
    min_box_iou: float = 0.85,
) -> dict[str, Any]:
    """Kiểm tra độ tương đương suy luận giữa mô hình PyTorch và mô hình đã export (ONNX/OpenVINO).

    Tiêu chí chất lượng MLOps (Quality Gate):
    - Trùng khớp class_id giữa các hộp tương ứng.
    - Sai lệch độ tin cậy (confidence) <= conf_tolerance (mặc định 0.05).
    - IoU giữa bounding box PyTorch và Exported >= min_box_iou (mặc định 0.85).
    """
    from ultralytics import YOLO

    from .error_analysis import box_iou

    pt_model = YOLO(str(pytorch_model_path))
    exp_model = YOLO(str(exported_model_path))

    results_summary: dict[str, Any] = {
        "images_tested": len(sample_images),
        "parity_passed": True,
        "max_conf_diff": 0.0,
        "min_box_iou": 1.0,
        "mismatches": [],
    }

    for img in sample_images:
        # Parity so sánh cùng raw candidate ở ngưỡng thấp; decision policy
        # được kiểm tra riêng trong detector/predictor.
        pt_res = pt_model.predict(source=str(img), imgsz=imgsz, conf=0.001, verbose=False)[0]
        exp_res = exp_model.predict(source=str(img), imgsz=imgsz, conf=0.001, verbose=False)[0]

        pt_boxes = pt_res.boxes
        exp_boxes = exp_res.boxes

        if len(pt_boxes) != len(exp_boxes):
            results_summary["parity_passed"] = False
            results_summary["mismatches"].append(
                {
                    "image": str(img),
                    "reason": (
                        f"Khác biệt số box: PyTorch={len(pt_boxes)}, Exported={len(exp_boxes)}"
                    ),
                }
            )
            continue

        for i in range(len(pt_boxes)):
            pt_cls = int(pt_boxes.cls[i])
            exp_cls = int(exp_boxes.cls[i])
            pt_conf = float(pt_boxes.conf[i])
            exp_conf = float(exp_boxes.conf[i])
            pt_xyxy = tuple(float(v) for v in pt_boxes.xyxy[i].tolist())
            exp_xyxy = tuple(float(v) for v in exp_boxes.xyxy[i].tolist())

            conf_diff = abs(pt_conf - exp_conf)
            results_summary["max_conf_diff"] = max(results_summary["max_conf_diff"], conf_diff)

            iou = box_iou(pt_xyxy, exp_xyxy)
            results_summary["min_box_iou"] = min(results_summary["min_box_iou"], iou)

            if pt_cls != exp_cls or conf_diff > conf_tolerance or iou < min_box_iou:
                results_summary["parity_passed"] = False
                results_summary["mismatches"].append(
                    {
                        "image": str(img),
                        "box_index": i,
                        "pt_cls": pt_cls,
                        "exp_cls": exp_cls,
                        "conf_diff": conf_diff,
                        "iou": iou,
                    }
                )

    return results_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xuất mô hình sang định dạng triển khai")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "torchscript", "openvino"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("artifacts/export"))
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--check-parity", action="store_true", help="Kiểm tra tương đương suy luận")
    parser.add_argument(
        "--sample-images",
        nargs="*",
        type=Path,
        default=None,
        help="Danh sách ảnh mẫu kiểm tra parity",
    )
    return parser.parse_args()


def export_model(
    weights_path: Path | str,
    target_format: str = "onnx",
    imgsz: int = 640,
    output_dir: Path | str = Path("artifacts/export"),
    dynamic: bool = False,
    simplify: bool = False,
    check_parity: bool = False,
    sample_images: list[Path | str] | None = None,
) -> dict[str, Any]:
    """Xuất mô hình YOLOv8 sang định dạng triển khai thực tế và ghi metadata kiểm toán."""
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    output_dir = Path(output_dir)

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if imgsz <= 0:
        raise ValueError("imgsz phải lớn hơn 0")

    import logging
    logger = logging.getLogger("rice_leaf_export")

    model = YOLO(str(weights_path))

    try:
        exported_path_str = model.export(
            format=target_format,
            imgsz=imgsz,
            dynamic=dynamic,
            simplify=simplify,
        )
    except Exception as exc:
        if target_format == "onnx":
            logger.warning(
                "Xuất ONNX gặp vấn đề phụ thuộc (%s), chuyển sang định dạng TorchScript.", exc
            )
            target_format = "torchscript"
            exported_path_str = model.export(
                format="torchscript",
                imgsz=imgsz,
                dynamic=dynamic,
                simplify=simplify,
            )
        else:
            raise

    exported = Path(exported_path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / exported.name
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)

    parity_info = None
    if check_parity:
        sample_imgs = sample_images or list(Path("data/sample").glob("*.jpg"))
        if sample_imgs:
            parity_info = verify_prediction_parity(
                pytorch_model_path=weights_path,
                exported_model_path=destination,
                sample_images=sample_imgs,
                imgsz=imgsz,
            )

    metadata = {
        "source_weights": str(weights_path.resolve()),
        "source_sha256": sha256_file(weights_path),
        "exported_file": str(destination.resolve()),
        "exported_model": destination.name,
        "exported_sha256": sha256_file(destination),
        "format": target_format,
        "image_size": imgsz,
        "dynamic": dynamic,
        "simplified": simplify,
        "class_names": model.names,
        "parity_check": parity_info,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    configure_utf8_console()
    args = parse_args()

    meta = export_model(
        weights_path=args.weights,
        target_format=args.format,
        imgsz=args.imgsz,
        output_dir=args.output,
        dynamic=args.dynamic,
        simplify=args.simplify,
        check_parity=args.check_parity,
        sample_images=args.sample_images,
    )
    if meta.get("parity_check"):
        parity_info = meta["parity_check"]
        status_txt = "ĐẠT" if parity_info["parity_passed"] else "KHÔNG ĐẠT"
        print(
            f"Kiểm tra Parity: {status_txt} "
            f"(Max Diff: {parity_info['max_conf_diff']:.4f}, "
            f"Min IoU: {parity_info['min_box_iou']:.4f})"
        )
    print(f"Mô hình đã xuất: {meta['exported_file']}")


if __name__ == "__main__":
    main()
