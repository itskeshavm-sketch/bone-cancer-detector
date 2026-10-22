"""Stage V28 assets: V21 weights + the wild pool (121 train / 78 held-out) + ONLY
the 79 harvested radiographs that survived a by-eye modality audit.

Why this run exists
-------------------
V26 and V27 both added the full 122-image harvest and both LOST to V25. A by-eye
audit of all 122 on contact sheets found 43 of them are not radiographs at all:
CT slices, MRI images, pathology micrographs and whole-body bone scans, each
carrying a page-level bone-tumour label. Training on them teaches the network
that a CT slice is a radiograph.

V28 is V27 minus those 43 images. Everything else is identical:
  V27: 121 wild + 122 harvested (43 poisoned) x repeat 6
  V28: 121 wild +  79 harvested (0 poisoned)  x repeat 6
Same init, same lr, same epochs, same 78-film held-out slice, same repeat factor.
The only variable is whether the harvest contains non-radiographs.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "kaggle_out_probe"))
from harvest_audit import REJECT_MODALITY  # noqa: E402

ROOT = Path("kaggle_data/v28_assets")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)

shutil.copy2("checkpoints/stageB21.pt", ROOT / "stageB21.pt")

split = json.loads(Path("checkpoints/wild_split.json").read_text())
man = json.loads(Path("wild_train2/manifest.json").read_text())
reject = set(REJECT_MODALITY)

n_wild = n_harv = n_drop = 0
harvested = []
with zipfile.ZipFile(ROOT / "wild_images.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path("wild_test").glob("ra_*.png")):
        stem = p.stem
        z.write(p, f"wildimg_{stem}.png")
        z.writestr(f"wildlbl_{stem}.txt",
                   "malignant" if p.name.startswith("ra_malignant") else "benign")
        n_wild += 1
    for i, m in enumerate(man):
        if i in reject:
            n_drop += 1
            continue
        fname = m["file"]
        p = Path("wild_train2") / fname
        if not p.exists():
            n_drop += 1
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
split["harvested_rejected"] = sorted(reject)
split["audit"] = "by-eye modality audit; CT / MRI / pathology / bone-scan rejected"
Path(ROOT / "wild_split.json").write_text(json.dumps(split, indent=1))

(ROOT / "dataset-metadata.json").write_text(json.dumps({
    "id": "juyhtrfewqasdfrgt/v28-assets",
    "title": "V28 assets: weights, wild pool, 79 clean rads",
    "licenses": [{"name": "CC0-1.0"}]}, indent=1))

print(f"wild images: {n_wild} | harvested kept: {n_harv} | rejected: {n_drop}")
print(f"split train {len(split['train'])} (incl {len(harvested)} harvested) | "
      f"val {len(split['val'])}")
print(f"effective real weight at repeat 6: {len(split['train']) * 6} vs 2736 main = "
      f"{len(split['train']) * 6 / (len(split['train']) * 6 + 2736):.1%} real")
print(f"V27 comparison: 243 images x 6 = {243*6} items, 43 of them non-radiographs")
for name, size in [(p.name, p.stat().st_size) for p in sorted(ROOT.iterdir()) if p.is_file()]:
    print(f"   {name:<24} {size:>12,}")
