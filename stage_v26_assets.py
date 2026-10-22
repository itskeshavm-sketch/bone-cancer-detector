"""Stage V26 assets: V21 weights + the wild pool (121 train / 78 held-out) + the
122 harvested blind-spot radiographs.

The repeat factor is chosen so V26 differs from V25 in ONE variable only:
  V25: 159 real images x 6 = 954 items   vs 2,736 main  -> 25.9% real
  V26: 243 real images x 4 = 972 items   vs 2,736 main  -> 26.2% real
So the mix weight is held constant and the amount of real data is 3.2x larger.
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path("kaggle_data/v26_assets")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)

shutil.copy2("checkpoints/stageB21.pt", ROOT / "stageB21.pt")

split = json.loads(Path("checkpoints/wild_split.json").read_text())
man = {m["file"]: m for m in json.loads(Path("wild_train2/manifest.json").read_text())}

n_wild = n_harv = 0
with zipfile.ZipFile(ROOT / "wild_images.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path("wild_test").glob("ra_*.png")):
        stem = p.stem
        z.write(p, f"wildimg_{stem}.png")
        z.writestr(f"wildlbl_{stem}.txt",
                   "malignant" if p.name.startswith("ra_malignant") else "benign")
        n_wild += 1
    harvested = []
    for fname, m in man.items():
        p = Path("wild_train2") / fname
        if not p.exists():
            continue
        z.write(p, f"wildimg2_{Path(fname).stem}.png")
        z.writestr(f"wildlbl2_{Path(fname).stem}.txt", m["label"])
        harvested.append({"file": f"wildimg2_{Path(fname).stem}.png",
                          "label": m["label"], "entity": m["entity"]})
        n_harv += 1

for key in ("train", "val"):
    for e in split[key]:
        e["file"] = f"wildimg_{Path(e['file']).stem}.png"
split["train"] = split["train"] + harvested
split["harvested_added"] = len(harvested)
Path(ROOT / "wild_split.json").write_text(json.dumps(split, indent=1))

(ROOT / "dataset-metadata.json").write_text(json.dumps({
    "id": "juyhtrfewqasdfrgt/v26-assets",
    "title": "V26 assets: weights, wild pool, 122 harvested radiographs",
    "licenses": [{"name": "CC0-1.0"}]}, indent=1))

print(f"wild images: {n_wild} | harvested: {n_harv}")
print(f"split train {len(split['train'])} (incl {len(harvested)} harvested) | "
      f"val {len(split['val'])}")
print(f"effective real weight: {len(split['train']) * 4} vs 2736 main = "
      f"{len(split['train']) * 4 / (len(split['train']) * 4 + 2736):.1%} real")
for name, size in [(p.name, p.stat().st_size) for p in sorted(ROOT.iterdir()) if p.is_file()]:
    print(f"   {name:<24} {size:>12,}")