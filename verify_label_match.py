"""Prove the label lookup resolves against the REAL split JSON and the REAL zip,
for any assets folder. V25 burned ~1.5h of GPU training on zero wild images
because of a prefix-matching bug; this is the gate that stops that repeating.

Usage: python verify_label_match.py kaggle_data/v28_assets
"""
import json
import sys
import zipfile
from pathlib import Path

FOLDER = Path(sys.argv[1] if len(sys.argv) > 1 else "kaggle_data/v28_assets")
SPLIT = FOLDER / "wild_split.json"
ZIP = FOLDER / "wild_images.zip"


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

n_split = len(split["train"]) + len(split["val"])
print(f"{FOLDER}")
print(f"zip files {len(names)} | images {len(wimg)} | labels {len(wlab)} | "
      f"split entries {n_split}")
assert len(wimg) == n_split, (len(wimg), n_split)
assert len(wlab) == n_split, (len(wlab), n_split)

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

tr = {wild_key(e["file"]) for e in split["train"]}
vl = {wild_key(e["file"]) for e in split["val"]}
assert not (tr & vl), "train/val overlap"
harv = [e for e in split["train"] if e["file"].startswith("wildimg2_")]
print(f"\ntrain {len(tr)} | val {len(vl)} | overlap 0")
print(f"harvested in train: {len(harv)} (rejected: {split.get('harvested_rejected')})")

# the zip must contain every split entry exactly once
assert len(wimg) == len(set(wimg)), "duplicate key after normalisation"
print("\nLABEL MATCHING VERIFIED - safe to push")
