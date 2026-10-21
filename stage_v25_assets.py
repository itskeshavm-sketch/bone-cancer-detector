"""Stage V25 assets: V21 weights + wild pool (121 train / 78 held-out) + the 38
VERIFIED harvested radiographs of the blind-spot entities.

The harvested images go in with a `wildimg2_` prefix and are added to the TRAIN
side of the split only - the 78-film held-out slice stays byte-identical to V24's,
so V25 vs V24 is a like-for-like comparison.

Caveat recorded in the manifest: the harvested images come from the same
bonetumor.org diagnosis pages as some held-out wild films, so the held-out
number means "unseen images, same entities", not "unseen source".
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path("kaggle_data/v25_assets")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)

shutil.copy2("checkpoints/stageB21.pt", ROOT / "stageB21.pt")

split = json.loads(Path("checkpoints/wild_split.json").read_text())
n_wild = n_harv = 0
with zipfile.ZipFile(ROOT / "wild_images.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path("wild_test").glob("ra_*.png")):
        stem = p.stem
        z.write(p, f"wildimg_{stem}.png")
        lab = "malignant" if p.name.startswith("ra_malignant") else "benign"
        z.writestr(f"wildlbl_{stem}.txt", lab)
        n_wild += 1
    # harvested blind-spot radiographs (already filtered to real radiographs)
    man = {m["file"]: m for m in json.loads(Path("wild_train2/manifest.json").read_text())}
    for p in sorted(Path("wild_train2").glob("bt2_*.png")):
        m = man.get(p.name)
        if not m:
            continue
        stem = p.stem
        z.write(p, f"wildimg2_{stem}.png")
        z.writestr(f"wildlbl2_{stem}.txt", m["label"])
        n_harv += 1

for key in ("train", "val"):
    for e in split[key]:
        e["file"] = f"wildimg_{Path(e['file']).stem}.png"
# harvested blind-spot radiographs (already filtered to real radiographs)
# NOTE: the manifest contains duplicate filenames (harvest v1 and v2 both wrote some),
# so dedupe by filename or the split would carry phantom duplicate rows.
man = {m["file"]: m for m in json.loads(Path("wild_train2/manifest.json").read_text())}
harvested = []
seen = set()
for fname, m in man.items():
    if fname in seen:
        continue
    if not (Path("wild_train2") / fname).exists():
        continue
    seen.add(fname)
    harvested.append({"file": f"wildimg2_{Path(fname).stem}.png",
                      "label": m["label"], "entity": m["entity"]})
harvested.sort(key=lambda e: e["file"])
split["train"] = split["train"] + harvested
split["harvested_added"] = len(harvested)
Path(ROOT / "wild_split.json").write_text(json.dumps(split, indent=1))

(ROOT / "dataset-metadata.json").write_text(json.dumps({
    "id": "juyhtrfewqasdfrgt/v25-assets",
    "title": "V25 assets: weights + wild pool + harvested blindspots",
    "licenses": [{"name": "CC0-1.0"}]}, indent=1))

print(f"wild images: {n_wild} | harvested added to TRAIN: {n_harv}")
print(f"split train {len(split['train'])} (incl {len(harvested)} harvested) | "
      f"val {len(split['val'])}")
files = [(p.name, p.stat().st_size) for p in sorted(ROOT.iterdir()) if p.is_file()]
for name, size in files:
    print(f"   {name:<24} {size:>12,}")
