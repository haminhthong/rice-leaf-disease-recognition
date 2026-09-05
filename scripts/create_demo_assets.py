"""Script tự động sinh dữ liệu mẫu giả định (Synthetic Demo Assets Generator).

Script này tạo ra:
1. Hai ảnh mẫu lá lúa (bacterial_leaf_blight và brown_spot) trong `data/sample/`
   để chạy thử CLI `rice-predict`, Streamlit Dashboard và FastAPI lập tức.
2. Các file nén ZIP bộ dữ liệu mẫu giả định (`RiceLeafAnnotatedDataset.zip` và `dataset1.zip`)
   để chạy thử pipeline `rice-prepare` trôi chảy mà không cần tải dataset thật nặng hàng GB.
"""

import io
import random
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

# Đảm bảo import được rice_leaf_detection khi chạy script trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rice_leaf_detection.utils import configure_utf8_console


def create_synthetic_leaf_image(
    disease_type: str, seed_index: int = 0, width: int = 640, height: int = 640
) -> Image.Image:
    """Tạo ảnh tổng hợp giả lập lá lúa kèm đốm bệnh có phash đa dạng theo seed_index."""
    rng = random.Random(seed_index)
    r = rng.randint(40, 120)
    g = rng.randint(120, 220)
    b = rng.randint(10, 60)
    image = Image.new("RGB", (width, height), color=(r, g, b))
    draw = ImageDraw.Draw(image)

    # Đường gân lá đa dạng
    for _ in range(8):
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = rng.randint(0, width), rng.randint(0, height)
        draw.line(
            [(x1, y1), (x2, y2)],
            fill=(rng.randint(50, 150), rng.randint(150, 255), rng.randint(50, 150)),
            width=rng.randint(2, 6),
        )

    # Đốm bệnh ngẫu nhiên
    for _ in range(4):
        x0 = rng.randint(50, 400)
        y0 = rng.randint(50, 400)
        x1 = x0 + rng.randint(60, 200)
        y1 = y0 + rng.randint(60, 200)
        box = [x0, y0, min(x1, width - 10), min(y1, height - 10)]

        if disease_type == "Bacterial_Leaf_Blight":
            draw.rectangle(box, fill=(rng.randint(180, 255), rng.randint(120, 200), 0))
        elif disease_type == "Brown_Spot":
            draw.ellipse(box, fill=(rng.randint(100, 160), rng.randint(40, 90), 0))
        else:
            draw.rectangle(box, fill=(rng.randint(160, 240), rng.randint(100, 180), 0))

    return image


def create_synthetic_zip(zip_path: Path, dataset_name: str, num_images: int = 50) -> None:
    """Tạo file ZIP chứa cấu trúc dataset YOLOv8 hợp lệ với các ảnh độc lập."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        yaml_content = """path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['Bacterial Leaf Blight', 'Brown Spot']
"""
        zip_file.writestr("data.yaml", yaml_content)

        splits = ["train", "valid", "test"]
        counter = 0
        for split in splits:
            for i in range(num_images):
                counter += 1
                img_name = f"{dataset_name}_{split}_{i + 1:02d}.jpg"
                lbl_name = f"{dataset_name}_{split}_{i + 1:02d}.txt"

                img = create_synthetic_leaf_image("Mixed", seed_index=counter * 100 + i)

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")

                zip_file.writestr(f"{split}/images/{img_name}", buffer.getvalue())

                # Mỗi ảnh chứa 2 instance: 1 Bacterial Leaf Blight (0) và 1 Brown Spot (1)
                label_text = (
                    "0 0.350000 0.450000 0.300000 0.500000\n1 0.650000 0.550000 0.200000 0.200000\n"
                )
                zip_file.writestr(f"{split}/labels/{lbl_name}", label_text)


def main() -> None:
    configure_utf8_console()
    print("🌾 Đang khởi tạo dữ liệu demo mẫu...")

    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)

    img_blight = create_synthetic_leaf_image("Bacterial_Leaf_Blight", seed_index=1)
    img_blight.save(sample_dir / "bacterial_leaf_blight_sample.jpg")

    img_brown = create_synthetic_leaf_image("Brown_Spot", seed_index=2)
    img_brown.save(sample_dir / "brown_spot_sample.jpg")

    print(f"✅ Đã tạo ảnh sample tại: {sample_dir.resolve()}")

    zip1 = Path("RiceLeafAnnotatedDataset.zip")
    zip2 = Path("dataset1.zip")

    create_synthetic_zip(zip1, "ds1", num_images=50)
    create_synthetic_zip(zip2, "ds2", num_images=50)

    print(f"✅ Đã tạo 2 bộ dữ liệu ZIP mẫu: {zip1.name}, {zip2.name}")
    print("🎉 Đã sẵn sàng chạy thử `rice-prepare` hoặc `pytest`!")


if __name__ == "__main__":
    main()
