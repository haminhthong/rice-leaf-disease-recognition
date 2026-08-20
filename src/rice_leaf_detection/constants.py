from pathlib import Path

SEED = 42
CLASS_NAMES = ["Bacterial_Leaf_Blight", "Brown_Spot"]
CLASS_NAMES_VI = {0: "Bạc lá lúa", 1: "Đốm nâu"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_ARCHIVES = (Path("RiceLeafAnnotatedDataset.zip"), Path("dataset1.zip"))
