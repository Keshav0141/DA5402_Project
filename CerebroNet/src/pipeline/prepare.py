
import os
import shutil
import yaml
import logging
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def prepare_v1(params: dict) -> None:
    """
    Version 1 — Copy raw data into v1_resized folder.
    Maintains the same folder structure (class subfolders).
    Just copies images — resizing happens in transform.py
    """
    raw_dir = Path(params["data"]["raw_dir"])
    v1_dir  = Path(params["data"]["v1_dir"])
    classes = params["data"]["classes"]

    logger.info("Starting V1 data preparation...")

    for split in ["Training", "Testing"]:
        for cls in classes:
            src = raw_dir / split / cls
            dst = v1_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                logger.warning(f"Source not found: {src}")
                continue

            count = 0
            for img_file in src.glob("*.jpg"):
                shutil.copy2(img_file, dst / img_file.name)
                count += 1

            logger.info(f"[V1] {split}/{cls}: copied {count} images")

    logger.info("V1 data preparation complete.")


def log_dataset_stats(params: dict) -> None:
    """Log class distribution for reproducibility."""
    v1_dir  = Path(params["data"]["v1_dir"])
    classes = params["data"]["classes"]

    logger.info("─── Dataset Statistics ───")
    total = 0
    for split in ["Training", "Testing"]:
        for cls in classes:
            path = v1_dir / split / cls
            count = len(list(path.glob("*.jpg"))) if path.exists() else 0
            total += count
            logger.info(f"  {split}/{cls}: {count} images")
    logger.info(f"  Total: {total} images")


if __name__ == "__main__":
    try:
        params = load_params()
        prepare_v1(params)
        log_dataset_stats(params)
        logger.info("prepare.py finished successfully.")
    except Exception as e:
        logger.error(f"prepare.py failed: {e}")
        raise