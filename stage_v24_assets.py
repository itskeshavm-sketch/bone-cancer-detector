"""Stage V24 assets with the wild images in ONE zip (one upload instead of 199).

Kaggle's dataset uploader sends each file as a separate HTTP request; 401 files
took forever on this connection. Zipping the 199 wild PNGs + their labels into a
single archive turns that into one transfer. The kernel unzips it.
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path("kaggle_data/v24_assets")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)

shutil.copy2("checkpoints/stageB21.pt", ROOT / "stageB21.pt")

split = json.loads(Path("checkpoints/wild_split.json").read_text())
n = 0
with zipfile.ZipFile(ROOT / "wild_images.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path("wild_test").glob("ra_*.png")):
        stem = p.stem
        z.write(p, f"wildimg_{stem}.png")
        lab = "malignant" if p.name.startswith("ra_malignant") else "benign"
        z.writestr(f"wildlbl_{stem}.txt", lab)
        n += 1

for key in ("train", "val"):
    for e in split[key]:
        e["file"] = f"wildimg_{Path(e['file']).stem}.png"
Path(ROOT / "wild_split.json").write_text(json.dumps(split, indent=1))

(ROOT / "dataset-metadata.json").write_text(json.dumps({
    "id": "juyhtrfewqasdfrgt/v24-assets",
    "title": "V24 assets: V21 weights + wild pool zip",
    "licenses": [{"name": "CC0-1.0"}]}, indent=1))

files = [(p.name, p.stat().st_size) for p in sorted(ROOT.iterdir()) if p.is_file()]
print(f"zipped {n} wild images")
for name, size in files:
    print(f"   {name:<20} {size:>12,} bytes")
print(f"total {sum(s for _, s in files):,} bytes in {len(files)} files (was 401)")