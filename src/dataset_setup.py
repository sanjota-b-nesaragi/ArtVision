"""
ARTVISION - Dataset Setup and Management Module
Inspects dataset structure, checks data health, validates image formats,
removes corrupted files, splits raw artwork into train/val/test partitions (70/15/15),
and provides comprehensive guides for downloading the WikiArt dataset.
"""

import os
import sys
import shutil
import random
import hashlib
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
CLASS_NAMES = ["Abstract", "Cubism", "Impressionism", "Surrealism"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_dataset_breakdown(dataset_root: str = DATASET_ROOT) -> Dict[str, Any]:
    """
    Scans the dataset folder and tallies the number of verified images in each split and class.
    """
    summary = {
        "splits": {},
        "total_images": 0,
        "is_ready_for_training": False,
        "classes": CLASS_NAMES
    }

    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(dataset_root, split)
        summary["splits"][split] = {}
        split_count = 0
        for cls in CLASS_NAMES:
            cls_folder = os.path.join(split_dir, cls)
            if os.path.exists(cls_folder):
                images = [
                    f for f in os.listdir(cls_folder)
                    if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
                ]
                count = len(images)
            else:
                count = 0
            summary["splits"][split][cls] = count
            split_count += count
        summary["splits"][split]["total"] = split_count
        summary["total_images"] += split_count

    # Check readiness (needs at least some images in every class of train & validation)
    train_has_all = all(summary["splits"]["train"].get(c, 0) > 0 for c in CLASS_NAMES)
    val_has_all = all(summary["splits"]["validation"].get(c, 0) > 0 for c in CLASS_NAMES)
    summary["is_ready_for_training"] = train_has_all and val_has_all

    return summary


def clean_and_verify_images(folder_path: str) -> Tuple[int, int]:
    """
    Walks a directory, verifies image integrity using PIL, and removes corrupted files.
    Returns (verified_count, removed_count).
    """
    try:
        from PIL import Image
    except ImportError:
        print("[Notice] Pillow not installed; skipping deep byte decoding checks.")
        return 0, 0

    verified = 0
    removed = 0

    for root, _, files in os.walk(folder_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                fpath = os.path.join(root, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    # Reopen and check dimensions
                    with Image.open(fpath) as img:
                        w, h = img.size
                        if w < 32 or h < 32:
                            print(f"Removing undersized image: {fpath} ({w}x{h})")
                            os.remove(fpath)
                            removed += 1
                            continue
                    verified += 1
                except Exception as e:
                    print(f"Removing corrupted image file: {fpath} ({e})")
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
                    removed += 1

    return verified, removed


def deduplicate_folder(folder_path: str) -> int:
    """
    Scans a folder and deletes exact duplicate images based on MD5 hashes.
    """
    hashes = set()
    duplicates_removed = 0

    for root, _, files in os.walk(folder_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                fpath = os.path.join(root, fname)
                try:
                    hasher = hashlib.md5()
                    with open(fpath, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            hasher.update(chunk)
                    digest = hasher.hexdigest()
                    if digest in hashes:
                        os.remove(fpath)
                        duplicates_removed += 1
                    else:
                        hashes.add(digest)
                except Exception:
                    pass

    return duplicates_removed


def organize_and_split_dataset(
    source_dir: str,
    target_dataset_dir: str = DATASET_ROOT,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Takes a source directory containing raw folders for each class (e.g. source_dir/Impressionism)
    and splits the images into train/validation/test sets according to specified ratios.
    """
    random.seed(seed)

    for split in ["train", "validation", "test"]:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(target_dataset_dir, split, cls), exist_ok=True)

    print(f"Splitting raw dataset from '{source_dir}' -> {train_ratio*100}% Train, {val_ratio*100}% Val, {test_ratio*100}% Test...")

    for cls in CLASS_NAMES:
        source_cls_dir = os.path.join(source_dir, cls)
        if not os.path.exists(source_cls_dir):
            print(f"[Warning] Class directory not found: {source_cls_dir}")
            continue

        images = [
            f for f in os.listdir(source_cls_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split_name, img_subset in [("train", train_imgs), ("validation", val_imgs), ("test", test_imgs)]:
            dest_dir = os.path.join(target_dataset_dir, split_name, cls)
            for img_name in img_subset:
                src_path = os.path.join(source_cls_dir, img_name)
                dst_path = os.path.join(dest_dir, img_name)
                shutil.copy2(src_path, dst_path)

        print(f"  {cls}: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

    print("Dataset splitting complete.")


def populate_demo_dataset():
    """
    Populates sample images from sample_data into train/validation/test splits
    so that model training scripts and data loaders can be tested immediately in setup state.
    """
    sample_dir = os.path.join(DATASET_ROOT, "sample_data")
    if not os.path.exists(sample_dir) or len(os.listdir(sample_dir)) == 0:
        from src.sample_images import create_all_sample_images
        create_all_sample_images(PROJECT_ROOT)

    for split in ["train", "validation", "test"]:
        for cls in CLASS_NAMES:
            dest = os.path.join(DATASET_ROOT, split, cls)
            os.makedirs(dest, exist_ok=True)
            sample_file = os.path.join(sample_dir, f"sample_{cls.lower()}.bmp")
            if os.path.exists(sample_file):
                # Copy multiple indexed variations for testing loaders
                for i in range(1, 4):
                    dst = os.path.join(dest, f"{cls.lower()}_demo_{i}.bmp")
                    shutil.copy2(sample_file, dst)

    print("Sample demo dataset populated across train/validation/test splits.")


def print_wikiart_instructions():
    """Display comprehensive instructions for obtaining real WikiArt and Kaggle datasets."""
    instructions = """
================================================================================
          WIKIART / KAGGLE ARTWORK DATASET ACQUISITION GUIDE
================================================================================

ARTVISION requires artwork images categorized into the following 4 styles:
  1. Impressionism
  2. Cubism
  3. Abstract (Abstract Art / Abstract Expressionism)
  4. Surrealism

--------------------------------------------------------------------------------
OPTION A: Kaggle WikiArt Dataset (Recommended)
--------------------------------------------------------------------------------
1. Ensure the Kaggle CLI is installed:
     pip install kaggle

2. Download the WikiArt Painting Dataset:
     kaggle datasets download -d steubk/wikiart

3. Unzip the downloaded file:
     tar -xf wikiart.zip  (or use 7-Zip / WinRAR on Windows)

4. Move or link the target style folders into your project:
     Source:                      Destination:
     wikiart/Impressionism/  -->  artwork_style_classification/dataset/raw/Impressionism/
     wikiart/Cubism/         -->  artwork_style_classification/dataset/raw/Cubism/
     wikiart/Abstract_Art/   -->  artwork_style_classification/dataset/raw/Abstract/
     wikiart/Surrealism/     -->  artwork_style_classification/dataset/raw/Surrealism/

5. Run the automatic splitter:
     python src/dataset_setup.py --split-raw dataset/raw

--------------------------------------------------------------------------------
OPTION B: Manual Download from WikiArt.org / Kaggle Web UI
--------------------------------------------------------------------------------
1. Visit: https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data
   or https://www.kaggle.com/datasets/steubk/wikiart
2. Download the archives for the 4 art styles.
3. Extract images directly into the train, validation, and test folders:
     dataset/train/Impressionism/
     dataset/train/Cubism/
     dataset/train/Abstract/
     dataset/train/Surrealism/
     (repeat for validation/ and test/ using an 70/15/15 ratio)

--------------------------------------------------------------------------------
OPTION C: Populate Sample Verification Set (Instant Local Testing)
--------------------------------------------------------------------------------
To test the pipeline and UI immediately without an internet download:
     python src/dataset_setup.py --populate-demo
================================================================================
"""
    print(instructions)
    return instructions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARTVISION Dataset Setup & Verification")
    parser.add_argument("--status", action="store_true", default=False, help="Display current dataset metrics")
    parser.add_argument("--guide", action="store_true", default=False, help="Print WikiArt download guide")
    parser.add_argument("--populate-demo", action="store_true", default=False, help="Seed train/val/test with local sample data")
    parser.add_argument("--clean", type=str, default=None, help="Clean and verify images in target directory")
    parser.add_argument("--split-raw", type=str, default=None, help="Split a raw directory containing style folders into train/val/test")

    args = parser.parse_args()

    if args.populate_demo:
        populate_demo_dataset()
        summary = get_dataset_breakdown()
        print(f"Status after demo population: {summary['total_images']} images ready.")
    elif args.split_raw:
        organize_and_split_dataset(args.split_raw)
    elif args.clean:
        v, r = clean_and_verify_images(args.clean)
        print(f"Cleaned folder '{args.clean}': {v} verified, {r} corrupted/undersized removed.")
    elif args.guide:
        print_wikiart_instructions()
    else:
        # Default action: print status & guide
        summary = get_dataset_breakdown()
        print("\n--- Current Dataset Status ---")
        print(f"Total Images: {summary['total_images']}")
        print(f"Ready for Training: {summary['is_ready_for_training']}")
        for split, counts in summary["splits"].items():
            print(f"  {split.upper()}: {counts.get('total', 0)} images ({counts})")
        if not summary["is_ready_for_training"]:
            print_wikiart_instructions()
