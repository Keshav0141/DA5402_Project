# src/pipeline/transform.py

import os
import yaml
import logging
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def resize_image(img: Image.Image, size: int) -> Image.Image:
    """Resize image to square."""
    return img.resize((size, size))


def augment_image(img: Image.Image) -> Image.Image:
    """
    Apply random augmentations to a single image.
    Augmentations: horizontal flip, brightness, contrast, slight blur.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Occasional slight blur
    if random.random() > 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img


def create_v1_resized(params: dict) -> None:
    """Resize all V1 images to target size (no augmentation)."""
    v1_dir  = Path(params["data"]["v1_dir"])
    classes = params["data"]["classes"]
    size    = params["data"]["img_size"]

    logger.info(f"Resizing V1 images to {size}x{size}...")

    for split in ["Training", "Testing"]:
        for cls in classes:
            src_dir = v1_dir / split / cls
            if not src_dir.exists():
                continue
            count = 0
            for img_path in src_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = resize_image(img, size)
                    img.save(img_path)
                    count += 1
                except Exception as e:
                    logger.warning(f"Could not process {img_path}: {e}")
            logger.info(f"[V1 resize] {split}/{cls}: {count} images resized")


def create_v2_augmented(params: dict) -> None:
    """
    Copy V1 Training data into V2 and apply augmentations.
    Testing data is copied as-is (never augment test set).
    """
    v1_dir  = Path(params["data"]["v1_dir"])
    v2_dir  = Path(params["data"]["v2_dir"])
    classes = params["data"]["classes"]
    size    = params["data"]["img_size"]

    random.seed(params["data"]["random_seed"])

    logger.info("Creating V2 augmented dataset...")

    for split in ["Training", "Testing"]:
        for cls in classes:
            src_dir = v1_dir / split / cls
            dst_dir = v2_dir / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            if not src_dir.exists():
                logger.warning(f"Source not found: {src_dir}")
                continue

            count = 0
            aug_count = 0

            for img_path in src_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = resize_image(img, size)

                    # Save original
                    img.save(dst_dir / img_path.name)
                    count += 1

                    # For training only: save one augmented copy
                    if split == "Training":
                        aug_img = augment_image(img.copy())
                        aug_name = f"aug_{img_path.name}"
                        aug_img.save(dst_dir / aug_name)
                        aug_count += 1

                except Exception as e:
                    logger.warning(f"Could not process {img_path}: {e}")

            if split == "Training":
                logger.info(
                    f"[V2] {split}/{cls}: "
                    f"{count} originals + {aug_count} augmented = {count + aug_count} total"
                )
            else:
                logger.info(f"[V2] {split}/{cls}: {count} images copied")


def log_v2_stats(params: dict) -> None:
    """Log final V2 dataset statistics."""
    v2_dir  = Path(params["data"]["v2_dir"])
    classes = params["data"]["classes"]

    logger.info("─── V2 Dataset Statistics ───")
    total = 0
    for split in ["Training", "Testing"]:
        for cls in classes:
            path  = v2_dir / split / cls
            count = len(list(path.glob("*.jpg"))) if path.exists() else 0
            total += count
            logger.info(f"  {split}/{cls}: {count} images")
    logger.info(f"  Total: {total} images")


if __name__ == "__main__":
    try:
        params = load_params()
        create_v1_resized(params)
        create_v2_augmented(params)
        log_v2_stats(params)
        logger.info("transform.py finished successfully.")
    except Exception as e:
        logger.error(f"transform.py failed: {e}")
        raise