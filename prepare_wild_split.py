"""Prepare a deterministic wild-pool split for V23 fine-tuning.

The error anatomy says 93% of V21's wild errors are CLASSIFICATION failures on
specific entity families (ABC, bone island, chondroblastoma, CMF, BFH,
chondrosarcoma, adamantinoma, chordoma...). The wild pool already CONTAINS those
entities - 199 labelled real films. So V23 fine-tunes on a slice of it.

Split: stratified by entity group, 60% train / 40% held-out validation, so the
held-out set keeps the same entity mix (the question is "does training on these
images of these entities fix the blind spots", not "can it guess a new entity").

Images already in the held-out set are NEVER trained on, and the benchmark stays
a separate guard metric so we can see whether fine-tuning on a different source
distribution damages the in-distribution score.

Outputs: wild_split.json + a folder of train images ready for a Kaggle dataset.
"""
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from random import Random

WILD = Path("wild_test")
OUT = Path("kaggle_data/wild_train")
TRAIN_FRAC = 0.60
SEED = 23

files = sorted(WILD.glob("ra_*.png"))
labeled = [p for p in files if p.name.startswith("ra_")]

# Canonical diagnosis keys: the hunt list groups variants of one diagnosis
# together (chondrosarcoma2/3/7a are all chondrosarcoma). Without this the
# split produced 174 groups for 199 films and a useless 16-film val set.
CANON = [
    (r"^abc", "abc"), (r"^aneurysmal", "abc"),
    (r"^boneisland|^bone_island", "boneisland"),
    (r"^chondrobl", "chondroblastoma"),
    (r"^cmf|^chondromyxoid", "cmf"),
    (r"^bfh|^fibroushistiocytoma", "bfh"),
    (r"^brown", "browntumor"),
    (r"^perichondroma", "perichondroma"),
    (r"^nora|^bizarre", "nora"),
    (r"^ench", "enchondroma"),
    (r"^gct|^giantcell", "gct"),
    (r"^fd|^fibrousdysplasia|^multiple_fd", "fd"),
    (r"^nof|^nonossifying", "nof"),
    (r"^eg|^eosinophilic", "eg"),
    (r"^sbc|^simplebonecyst", "sbc"),
    (r"^osteoid|^osteoidosteoma|^matrix", "osteoidosteoma"),
    (r"^osteoch|^osteochondrom", "osteochondroma"),
    (r"^cs|^chondrosarcoma|^lowgrade_cs|^chondrotumor", "chondrosarcoma"),
    (r"^adam", "adamantinoma"),
    (r"^angio", "angiosarcoma"),
    (r"^chordoma", "chordoma"),
    (r"^ewing", "ewing"),
    (r"^met|^meta", "metastasis"),
    (r"^myelo", "myeloma"),
    (r"^lymph", "lymphoma"),
    (r"^parosteal", "parosteal_os"),
    (r"^os_|^osteosarcoma|^osteosark", "osteosarcoma"),
    (r"^cortical", "cortical_os"),
]


def entity(name):
    stem = re.sub(r"^ra_(malignant|benign)_", "", Path(name).stem).lower()
    stem = re.sub(r"^web_?", "", stem)
    for pat, key in CANON:
        if re.match(pat, stem):
            return key
    stem = re.sub(r"[0-9]+$", "", stem)
    stem = re.sub(r"[_0-9a-z]*$", "", stem) if len(stem) > 6 else stem
    return stem or "other"


groups = defaultdict(list)
for p in labeled:
    groups[entity(p.name)].append(p)

print("entity groups:")
for g in sorted(groups, key=lambda k: -len(groups[k])):
    if len(groups[g]) > 1:
        print(f"   {g:<22} {len(groups[g])}")

rng = Random(SEED)
train, val = [], []
for g in sorted(groups):
    members = sorted(groups[g])
    rng.shuffle(members)
    n_train = max(1, round(len(members) * TRAIN_FRAC)) if len(members) > 1 else len(members)
    train.extend(members[:n_train])
    val.extend(members[n_train:])

train.sort()
val.sort()
print(f"labeled wild films: {len(labeled)}")
print(f"  train: {len(train)}  val: {len(val)}")
print(f"  entities: {len(groups)}")

meta = {
    "seed": SEED, "train_frac": TRAIN_FRAC,
    "train": [{"file": p.name, "label": "malignant" if p.name.startswith("ra_malignant") else "benign"}
              for p in train],
    "val": [{"file": p.name, "label": "malignant" if p.name.startswith("ra_malignant") else "benign"}
            for p in val],
}
Path("checkpoints/wild_split.json").write_text(json.dumps(meta, indent=1))

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True, exist_ok=True)
for p in train:
    shutil.copy2(p, OUT / "images" / p.name)
print(f"copied {len(list((OUT / 'images').glob('*.png')))} images -> {OUT}")

# sanity: no overlap between train and val
tr = {p.name for p in train}
vl = {p.name for p in val}
assert not (tr & vl), "train/val overlap!"
print("no train/val overlap: OK")
print(f"train malignant: {sum(1 for p in train if p.name.startswith('ra_malignant'))}"
      f" | val malignant: {sum(1 for p in val if p.name.startswith('ra_malignant'))}")
