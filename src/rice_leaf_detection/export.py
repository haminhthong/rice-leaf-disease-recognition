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

from .utils import configure_utf8_console, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xuất mô hình sang định dạng triển khai")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "torchscript", "openvino"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("artifacts/export"))
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
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
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Mô hình đã xuất: {destination.resolve()}")


if __name__ == "__main__":
    main()
