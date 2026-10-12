"""
Automated Dataset Setup for Bone Cancer Detection
Downloads and organizes the Bone Tumor X-Ray dataset into train/val splits.

USAGE (pick ONE method):

Method 1 - Kaggle CLI (recommended):
    pip install kaggle
    # Place your kaggle.json in ~/.kaggle/ (get it from kaggle.com/settings/api)
    python setup_dataset.py --method kaggle

Method 2 - Manual download:
    1. Download from: https://www.kaggle.com/datasets/nasirkhalid24/bone-tumor-detection-xray-images
    2. Extract the zip to ./raw_data/
    3. Run: python setup_dataset.py --method manual --raw-dir ./raw_data

Method 3 - Use MURA instead:
    python setup_dataset.py --method mura
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def organize_into_splits(source_dir, output_dir, val_ratio=0.15, test_ratio=0.10, seed=42):
    """
    Takes a flat or nested directory of images organized by class
    and splits into train/val/test with proper structure.
    """
    random.seed(seed)
    output_dir = Path(output_dir)

    # Find all class directories
    class_dirs = {}
    source = Path(source_dir)

    # Handle various dataset layouts
    for item in source.iterdir():
        if item.is_dir():
            # Check if this dir contains images directly
            images = list(item.glob("*.jpg")) + list(item.glob("*.png")) + list(item.glob("*.jpeg"))
            if images:
                class_dirs[item.name] = images
            else:
                # Check subdirectories (some datasets nest further)
                for sub in item.iterdir():
                    if sub.is_dir():
                        images = list(sub.glob("*.jpg")) + list(sub.glob("*.png")) + list(sub.glob("*.jpeg"))
                        if images:
                            class_dirs[sub.name] = images

    if not class_dirs:
        print("❌ No class directories with images found!")
        print(f"   Looked in: {source}")
        print("   Expected structure: source_dir/class_name/*.jpg")
        sys.exit(1)

    print(f"Found {len(class_dirs)} classes:")
    for cls, imgs in class_dirs.items():
        print(f"  • {cls}: {len(imgs)} images")

    # Split and copy
    for split_name in ["train", "val", "test"]:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for class_name, images in class_dirs.items():
        random.shuffle(images)
        n = len(images)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "test": images[:n_test],
            "val": images[n_test:n_test + n_val],
            "train": images[n_test + n_val:],
        }

        for split_name, split_images in splits.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(str(img_path), str(dest_dir / img_path.name))
                total_copied += 1

    print(f"\n✅ Organized {total_copied} images into {output_dir}/")
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        counts = {d.name: len(list(d.glob("*"))) for d in split_dir.iterdir() if d.is_dir()}
        print(f"  {split}: {counts}")


def download_kaggle(output_dir="./raw_data"):
    """Download using Kaggle CLI."""
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle CLI not installed. Run: pip install kaggle")
        print("   Then get API key from: https://www.kaggle.com/settings/api")
        sys.exit(1)

    dataset = "nasirkhalid24/bone-tumor-detection-xray-images"
    print(f"Downloading {dataset}...")
    os.system(f'kaggle datasets download -d {dataset} -p "{output_dir}" --unzip')
    print("Download complete.")
    return output_dir


def download_mura(output_dir="./raw_data"):
    """Download MURA dataset (Stanford)."""
    url = "https://stanfordaimi.azurewebsites.net/api/datasets/download/mura-v1.1"
    print("⚠️  MURA is ~40GB. This will take a while.")
    print(f"Downloading from {url}...")
    os.makedirs(output_dir, exist_ok=True)
    os.system(f'curl -L "{url}" -o "{output_dir}/mura.zip"')
    os.system(f'unzip "{output_dir}/mura.zip" -d "{output_dir}"')
    print("Download complete.")
    return output_dir


def main(args):
    raw_dir = args.raw_dir

    if args.method == "kaggle":
        raw_dir = download_kaggle(raw_dir)
    elif args.method == "mura":
        raw_dir = download_mura(raw_dir)
    elif args.method == "manual":
        if not Path(raw_dir).exists():
            print(f"❌ Directory not found: {raw_dir}")
            print("   Download manually and extract to that path first.")
            sys.exit(1)
    else:
        print(f"Unknown method: {args.method}")
        sys.exit(1)

    organize_into_splits(
        source_dir=raw_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print("\n🎉 Dataset ready! Train with:")
    print(f"   python train.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Bone Cancer Dataset")
    parser.add_argument("--method", choices=["kaggle", "manual", "mura"], default="kaggle",
                        help="How to obtain the dataset")
    parser.add_argument("--raw-dir", type=str, default="./raw_data",
                        help="Where raw/downloaded data lives (or will be downloaded to)")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Where to put organized train/val/test splits")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    args = parser.parse_args()
    main(args)
