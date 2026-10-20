"""CPU-only probe: what do the main dataset's .txt labels actually contain?

If they are YOLO rows (class cx cy w h) we can crop every tumor to its lesion,
which is the biggest untested accuracy lever for the specialist (Model 2):
instead of making it find the lesion in a whole radiograph, hand it a close-up.

Prints: token counts, sample rows, bbox size distribution, and how many
train/val/test tumors are croppable. No GPU, no training.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/kaggle/input")


def find_main():
    for p in ROOT.rglob("train/images"):
        if p.is_dir() and (p.parent / "labels").is_dir():
            return p.parent.parent
    return None


MAIN = find_main()
print("MAIN:", MAIN, flush=True)
if MAIN is None:
    sys.exit(2)

report = {"main": str(MAIN), "splits": {}}
for split in ["train", "val", "test"]:
    lbl_dir = MAIN / split / "labels"
    img_dir = MAIN / split / "images"
    if not lbl_dir.is_dir():
        continue
    tok_counts = Counter()
    samples = []
    widths, heights, areas = [], [], []
    n_files = 0
    for lp in sorted(lbl_dir.glob("*.txt")):
        n_files += 1
        txt = lp.read_text(errors="replace").strip()
        if not txt:
            tok_counts["empty"] += 1
            continue
        for line in txt.splitlines():
            parts = line.split()
            tok_counts[len(parts)] += 1
            if len(samples) < 8:
                samples.append({"file": lp.name, "row": line.strip()})
            if len(parts) >= 5:
                try:
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                    widths.append(w)
                    heights.append(h)
                    areas.append(w * h)
                except ValueError:
                    pass
    n_imgs = len(list(img_dir.glob("*"))) if img_dir.is_dir() else 0
    rep = {
        "label_files": n_files,
        "images": n_imgs,
        "token_count_histogram": dict(tok_counts),
        "samples": samples,
        "bbox_width": {
            "min": min(widths) if widths else None,
            "median": sorted(widths)[len(widths) // 2] if widths else None,
            "max": max(widths) if widths else None,
        },
        "bbox_height": {
            "min": min(heights) if heights else None,
            "median": sorted(heights)[len(heights) // 2] if heights else None,
            "max": max(heights) if heights else None,
        },
        "bbox_area_fraction_median": sorted(areas)[len(areas) // 2] if areas else None,
        "croppable_rows": len(areas),
    }
    report["splits"][split] = rep
    print(f"\n=== {split} ===", flush=True)
    print(json.dumps(rep, indent=1)[:1500], flush=True)

out = Path("/kaggle/working/label_report.json")
out.write_text(json.dumps(report, indent=1))
print("\nDONE", flush=True)