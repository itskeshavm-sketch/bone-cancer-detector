"""Prove the V25 label fix against the REAL split JSON and the REAL zip.

V25 burned ~1.5h of GPU training on zero wild images. This must never happen
again, so the matching is verified here first, and the check is run against the
same assets the kernel will see.
"""
import json
import zipfile
from pathlib import Path

SPLIT = Path("kaggle_data/v25_assets/wild_split.json")
ZIP = Path("kaggle_data/v25_assets/wild_images.zip")


def wild_key(name):
    """One rule for image and label: drop the prefix, drop the extension, drop
    leading underscores. 'wildimg' is 7 chars and 'wildlbl' is 8, so asymmetric
    stripping leaves residuals differing by an underscore - that was the V25 bug."""
    stem = name
    for pref in ("wildimg", "wildlbl"):
        if stem.startswith(pref):
            stem = stem[len(pref):]
            break
    for ext in (".png", ".txt"):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    return stem.lstrip("_")


split = json.loads(SPLIT.read_text())
names = zipfile.ZipFile(ZIP).namelist()
wimg, wlab = {}, {}
for n in names:
    base = Path(n).name
    if base.endswith(".png") and base.startswith("wildimg"):
        wimg.setdefault(wild_key(base), n)
    if base.endswith(".txt") and base.startswith("wildlbl"):
        wlab.setdefault(wild_key(base), n)

print(f"zip files {len(names)} | images {len(wimg)} | labels {len(wlab)}")
assert len(wimg) == 237, len(wimg)
assert len(wlab) == 237, len(wlab)

for key in ("train", "val"):
    unresolved = []
    resolved = 0
    malignant = 0
    for e in split[key]:
        k = wild_key(e["file"])
        if k in wimg and k in wlab:
            resolved += 1
            malignant += e["label"] == "malignant"
        else:
            unresolved.append(e["file"])
    print(f"  {key}: {resolved}/{len(split[key])} resolved "
          f"({malignant} malignant) | unresolved {len(unresolved)}")
    if unresolved:
        print("     e.g.", unresolved[:3])
        raise SystemExit("STILL BROKEN")

# overlap check
tr = {wild_key(e["file"]) for e in split["train"]}
vl = {wild_key(e["file"]) for e in split["val"]}
assert not (tr & vl), "train/val overlap"
print(f"\ntrain {len(tr)} | val {len(vl)} | overlap 0")
harv = [e for e in split["train"] if e["file"].startswith("wildimg2_")]
print(f"harvested in train: {len(harv)}")
print("\nLABEL MATCHING VERIFIED - safe to push")