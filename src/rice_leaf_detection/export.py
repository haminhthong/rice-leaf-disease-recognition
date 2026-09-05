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
        pt_res = pt_model.predict(source=str(img), imgsz=imgsz, conf=0.25, verbose=False)[0]
        exp_res = exp_model.predict(source=str(img), imgsz=imgsz, conf=0.25, verbose=False)[0]

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


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if args.imgsz <= 0:
        raise ValueError("--imgsz phải lớn hơn 0")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    exported = Path(
        model.export(
            format=args.format,
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            simplify=args.simplify,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / exported.name
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)

    parity_info = None
    if args.check_parity:
        sample_imgs = args.sample_images or list(Path("data/samples").glob("*.jpg"))
        if sample_imgs:
            parity_info = verify_prediction_parity(
                pytorch_model_path=args.weights,
                exported_model_path=destination,
                sample_images=sample_imgs,
                imgsz=args.imgsz,
            )
            status_txt = "ĐẠT" if parity_info["parity_passed"] else "KHÔNG ĐẠT"
            print(
                f"Kiểm tra Parity: {status_txt} "
                f"(Max Diff: {parity_info['max_conf_diff']:.4f}, "
                f"Min IoU: {parity_info['min_box_iou']:.4f})"
            )

    metadata = {
        "source_weights": str(args.weights.resolve()),
        "source_sha256": sha256_file(args.weights),
        "exported_model": destination.name,
        "exported_sha256": sha256_file(destination),
        "format": args.format,
        "image_size": args.imgsz,
        "dynamic": args.dynamic,
        "simplified": args.simplify,
        "class_names": model.names,
        "parity_check": parity_info,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Mô hình đã xuất: {destination.resolve()}")


if __name__ == "__main__":
    main()
