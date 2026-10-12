"""Convert YOLO detection dataset -> classification folders (normal / benign / malignant)"""
from pathlib import Path
import shutil

SRC = Path("raw_data/dataset")
DST = Path("data")

CLASSES = {0: "benign", 1: "malignant"}

for split in ["train", "val", "test"]:
    for cls_name in ["normal", "benign", "malignant"]:
        (DST / split / cls_name).mkdir(parents=True, exist_ok=True)

counts = {s: {} for s in ["train","val","test"]}

for split in ["train", "val", "test"]:
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"
    if not img_dir.exists():
        continue
    for img in img_dir.iterdir():
        if img.suffix.lower() not in [".jpg",".jpeg",".png"]:
            continue
        # label file: same stem + .txt (YOLO replaces .jpeg with .txt)
        # dataset uses names like btxrd_IMG000001.jpeg -> btxrd_IMG000001.txt
        label = lbl_dir / (img.stem + ".txt")
        # fallback: try replacing .jpeg
        if not label.exists():
            # handle double extension case
            label = lbl_dir / (img.name + ".txt")
            # also try the .rf. variant seen in val
            if not label.exists():
                # search by prefix
                candidates = list(lbl_dir.glob(img.stem + "*"))
                label = candidates[0] if candidates else None

        if label is None or not label.exists() or label.stat().st_size == 0:
            cls = "normal"
        else:
            # read first char = class id
            try:
                first = open(label).read().strip().split()[0]
                cls_id = int(float(first))
                cls = CLASSES.get(cls_id, "malignant")
            except:
                cls = "normal"

        shutil.copy2(img, DST / split / cls / img.name)
        counts[split][cls] = counts[split].get(cls, 0) + 1

print("Done! Data layout:")
for split in ["train","val","test"]:
    print(f"  {split}: {counts[split]}  total={sum(counts[split].values())}")

# also create binary version (normal vs tumor) via symlinks/copies for simple training
BIN = Path("data_binary")
for split in ["train","val","test"]:
    for c in ["normal","tumor"]:
        (BIN / split / c).mkdir(parents=True, exist_ok=True)

import os
for split in ["train","val","test"]:
    for img in (DST / split / "benign").glob("*"):
        shutil.copy2(img, BIN / split / "tumor" / img.name)
    for img in (DST / split / "malignant").glob("*"):
        shutil.copy2(img, BIN / split / "tumor" / img.name)
    for img in (DST / split / "normal").glob("*"):
        shutil.copy2(img, BIN / split / "normal" / img.name)

print("\nBinary layout (normal vs tumor):")
for split in ["train","val","test"]:
    n = len(list((BIN/split/"normal").glob("*")))
    t = len(list((BIN/split/"tumor").glob("*")))
    print(f"  {split}: normal={n} tumor={t}")
