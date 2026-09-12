"""Test fixture: Sinh dữ liệu mẫu giả định phục vụ kiểm thử và chạy thử nhanh."""

import io
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


def create_synthetic_leaf_image(
    disease_type: str, seed_index: int = 0, width: int = 640, height: int = 640
) -> Image.Image:
    """Tạo ảnh tổng hợp giả lập lá lúa kèm đốm bệnh theo seed_index."""
    rng = random.Random(seed_index)
    r = rng.randint(40, 120)
    g = rng.randint(120, 220)
    b = rng.randint(10, 60)
    image = Image.new("RGB", (width, height), color=(r, g, b))
    draw = ImageDraw.Draw(image)

    # Gân lá
    for _ in range(8):
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = rng.randint(0, width), rng.randint(0, height)
        draw.line(
            [(x1, y1), (x2, y2)],
            fill=(rng.randint(50, 150), rng.randint(150, 255), rng.randint(50, 150)),
            width=rng.randint(2, 6),
        )

    # Đốm bệnh
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
    """Tạo file ZIP chứa cấu trúc dataset YOLOv8 hợp lệ."""
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
