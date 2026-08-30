"""Script tự động sinh dữ liệu mẫu giả định (Synthetic Demo Assets Generator).

Script này tạo ra:
1. Các ảnh mẫu lá lúa (bacterial_leaf_blight và brown_spot) trong `data/sample/`
   để chạy thử CLI `rice-predict`, Streamlit Dashboard và FastAPI lập tức.
2. Các file nén ZIP bộ dữ liệu mẫu giả định (`RiceLeafAnnotatedDataset.zip` và `dataset1.zip`)
   để chạy thử pipeline `rice-prepare` trôi chảy mà không cần tải dataset thật nặng hàng GB.
"""

import io
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
    """Tạo ảnh tổng hợp giả lập lá lúa kèm đốm bệnh đa dạng theo seed_index."""
    # Nền lá lúa xanh lục lá mạ với thay đổi nhẹ tông màu
    bg_green = min(255, 140 + (seed_index * 7) % 30)
    image = Image.new("RGB", (width, height), color=(76, bg_green, 0))
    draw = ImageDraw.Draw(image)

    # Gân lá lúa (dọc)
    for x in range(20, width, 40):
        draw.line([(x, 0), (x, height)], fill=(102, 178, 255 if (x + seed_index * 5) % 80 == 0 else 120), width=2)

    offset = (seed_index * 13) % 100

    # Vẽ đốm bệnh theo loại
    if disease_type == "Bacterial_Leaf_Blight":
        # Bạc lá lúa: Vệt sọc đốm vàng/nâu kéo dài theo chiều lá
        draw.polygon([(200 + offset, 100), (280 + offset, 120), (260 + offset, 500), (180 + offset, 480)], fill=(204, 153, 0))
        draw.polygon([(400 - offset, 200), (460 - offset, 210), (450 - offset, 450), (390 - offset, 440)], fill=(153, 102, 0))
    elif disease_type == "Brown_Spot":
        # Đốm nâu: Đốm tròn/oval nhiều vị trí
        draw.ellipse([150 + offset, 150, 230 + offset, 230], fill=(102, 51, 0), outline=(204, 102, 0), width=3)
        draw.ellipse([350 - offset, 300, 430 - offset, 380], fill=(102, 51, 0), outline=(204, 102, 0), width=3)
        draw.ellipse([250, 450 - offset, 310, 510 - offset], fill=(102, 51, 0), outline=(204, 102, 0), width=3)

    return image


def create_synthetic_zip(zip_path: Path, dataset_name: str, num_images: int = 12) -> None:
    """Tạo file ZIP chứa cấu trúc dataset YOLOv8 hợp lệ với các ảnh độc lập."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # data.yaml
        yaml_content = f"""path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['Bacterial Leaf Blight', 'Brown Spot']
"""
        zip_file.writestr("data.yaml", yaml_content)

        # Tạo ảnh & nhãn cho các tập train, valid, test (mỗi tập 12 ảnh)
        splits = ["train", "valid", "test"]
        counter = 0
        for split in splits:
            for i in range(num_images):
                counter += 1
                img_name = f"{dataset_name}_{split}_{i+1:02d}.jpg"
                lbl_name = f"{dataset_name}_{split}_{i+1:02d}.txt"

                disease = "Bacterial_Leaf_Blight" if i % 2 == 0 else "Brown_Spot"
                img = create_synthetic_leaf_image(disease, seed_index=counter)

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")

                zip_file.writestr(f"{split}/images/{img_name}", buffer.getvalue())

                # YOLO format label: class_id x_center y_center width height
                class_id = 0 if disease == "Bacterial_Leaf_Blight" else 1
                label_text = f"{class_id} 0.350000 0.450000 0.300000 0.500000\n"
                zip_file.writestr(f"{split}/labels/{lbl_name}", label_text)


def main() -> None:
    configure_utf8_console()
    print("🌾 Đang khởi tạo dữ liệu demo mẫu...")

    # 1. Tạo ảnh sample cho rice-predict / API / Streamlit
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)

    img_blight = create_synthetic_leaf_image("Bacterial_Leaf_Blight", seed_index=1)
    img_blight.save(sample_dir / "bacterial_leaf_blight_sample.jpg")

    img_brown = create_synthetic_leaf_image("Brown_Spot", seed_index=2)
    img_brown.save(sample_dir / "brown_spot_sample.jpg")

    # Giữ 1 ảnh mặc định tên rice_leaf.jpg cho quickstart
    img_blight.save(sample_dir / "rice_leaf.jpg")

    print(f"✅ Đã tạo ảnh sample tại: {sample_dir.resolve()}")

    # 2. Tạo 2 file ZIP mẫu với các ảnh độc lập
    zip1 = Path("RiceLeafAnnotatedDataset.zip")
    zip2 = Path("dataset1.zip")

    create_synthetic_zip(zip1, "ds1", num_images=12)
    create_synthetic_zip(zip2, "ds2", num_images=12)

    print(f"✅ Đã tạo 2 bộ dữ liệu ZIP mẫu: {zip1.name}, {zip2.name}")
    print("🎉 Đã sẵn sàng chạy thử `rice-prepare` hoặc `pytest`!")


if __name__ == "__main__":
    main()
