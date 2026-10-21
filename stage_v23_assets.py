"""Stage the V23 fine-tuning assets as a Kaggle dataset.

Contents:
  stageB21.pt        the V21 specialist (the model we fine-tune)
  wild/images/*.png all 199 labelled wild films (the kernel splits them itself
                     using wild_split.json, and asserts train/val never overlap)
  wild_split.json    the deterministic 121/78 split
  wild_labels/       per-image .txt label (malignant/benign) for the kernel
"""
import json
import shutil
from pathlib import Path

ROOT = Path("kaggle_data/v23_assets")
if ROOT.exists():
    shutil.rmtree(ROOT)
(ROOT / "wild" / "images").mkdir(parents=True, exist_ok=True)
(ROOT / "wild" / "labels").mkdir(parents=True, exist_ok=True)

shutil.copy2("checkpoints/stageB21.pt", ROOT / "stageB21.pt")
shutil.copy2("checkpoints/wild_split.json", ROOT / "wild_split.json")

n = 0
for p in sorted(Path("wild_test").glob("ra_*.png")):
    shutil.copy2(p, ROOT / "wild" / "images" / p.name)
    lab = "malignant" if p.name.startswith("ra_malignant") else "benign"
    (ROOT / "wild" / "labels" / (p.stem + ".txt")).write_text(lab)
    n += 1

print(f"staged {n} wild images + labels")
print("files:")
for p in sorted(ROOT.rglob("*")):
    if p.is_file():
        print(f"   {p.relative_to(ROOT)}  {p.stat().st_size}")
