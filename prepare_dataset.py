import shutil
import random
from pathlib import Path

SRC = Path("data/raw/mvtec/bottle")
DST = Path("data/processed")

classes = ["good", "broken_large", "broken_small", "contamination"]


def copy_images(src_dir, dst_dir, split=0.8):
    images = list(Path(src_dir).glob("*.png"))
    if not images:
        print(f"⚠️ No images found in {src_dir}")
        return

    random.shuffle(images)

    split_idx = int(len(images) * split)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img in train_imgs:
        dst = dst_dir / "train" / img.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dst / img.name)

    for img in val_imgs:
        dst = dst_dir / "val" / img.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dst / img.name)


print("🚀 Preparing dataset...")

# good from train
copy_images(SRC / "train/good", DST)

# defects from test
for cls in ["broken_large", "broken_small", "contamination"]:
    copy_images(SRC / f"test/{cls}", DST)

# good from test
copy_images(SRC / "test/good", DST)

print("✅ Dataset prepared successfully!")

