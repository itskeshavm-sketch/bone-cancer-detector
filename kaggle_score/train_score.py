"""Kaggle GPU: score a candidate specialist on all THREE boards, so the champion
rule can be applied to it.

The local box has one core and is saturated by another workload, so scoring 813
films through an EfficientNet-B6 at 456px on the CPU is not viable. On a T4 the
same job is a couple of minutes.

Boards:
  bench  the 564 BTXRD test films, from the mounted bone-tumor dataset
  wild   the 199 real radiographs, from the assets bundle's wild_images.zip
         (the split's train+val entries ARE the full 199-film board, with labels)
  wkmd   the 50 third-source Wikimedia films, from the tta-assets bundle

Writes score_<model>.json for each board, ready to merge into the local
probability cache.

Usage: python kaggle_out_probe/push_any.py kaggle_score
"""
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, efficientnet_b6

WORK = Path("/kaggle/working")
TMP = Path("/tmp/kworkscore")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH = 24
device = torch.device("cuda")
MODEL_TAG = os.environ.get("MODEL_TAG", "V30")


def find_spec():
    """Prefer an explicitly named weights file over any spec*.pt, so a bundle
    holding several candidates cannot silently score the wrong one."""
    tag = MODEL_TAG.lower().replace("v", "")      # V30 -> 30
    for name in (f"specB{tag}.pt", f"spec{tag}.pt", f"spec{MODEL_TAG}.pt"):
        hit = next((p for p in Path("/kaggle/input").rglob(name) if p.is_file()), None)
        if hit:
            return hit
    for p in Path("/kaggle/input").rglob("spec*.pt"):
        if p.is_file():
            return p
    return None


def mark(s):
    print("STAGE:", s, flush=True)
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "STAGE.txt").write_text(s)
    except Exception:
        pass


def find_main():
    for p in Path("/kaggle/input").rglob("train/images"):
        if p.is_dir() and (p.parent / "labels").is_dir():
            return p.parent.parent
    return None


def load(path, maker, ncls=2, drop=0.4):
    ckpt = torch.load(path, map_location="cpu")
    m = maker(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    m.load_state_dict(ckpt["model"])
    return m.eval().to(device)


def tf(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def main():
    print("cuda:", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        sys.exit(2)
    TMP.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    MAIN = find_main()
    SPLIT = next((p for p in Path("/kaggle/input").rglob("wild_split.json")), None)
    GA = next((p for p in Path("/kaggle/input").rglob("stageA18.pt") if p.is_file()), None)
    SPEC = find_spec()
    MAN = next((p for p in Path("/kaggle/input").rglob("manifest.json")
                if p.parent.name == "tta-assets"), None)
    print("MAIN:", MAIN, "| SPLIT:", SPLIT, "| GATE:", GA, "| SPEC:", SPEC, "| MAN:", MAN, flush=True)
    if not (MAIN and SPLIT and GA and SPEC and MAN):
        print("missing inputs - aborting", flush=True)
        sys.exit(2)
    mark("inputs")

    gate = load(GA, efficientnet_b3)
    spec = load(SPEC, efficientnet_b6)
    evA, evB = tf(300), tf(456)

    # ---- benchmark ----
    MCLS = {0: "benign", 1: "malignant"}
    # The BTXRD test labels follow the training kernel's main_label rule exactly:
    # an EMPTY label file means "normal", 0 means benign, 1 means malignant.
    bench = []
    src = MAIN / "test" / "images"
    lbl_dir = MAIN / "test" / "labels"
    for img in sorted(src.glob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lb = lbl_dir / (img.stem + ".txt")
        if not lb.exists():
            lb = lbl_dir / (img.name + ".txt")
        truth = 2
        if lb.exists() and lb.stat().st_size:
            try:
                truth = MCLS.get(int(float(open(lb).read().strip().split()[0])), 2)
            except Exception:
                truth = 2
        bench.append((str(img), truth))
    print(f"benchmark films: {len(bench)} | "
          f"normal {sum(1 for _, t in bench if t == 2)} "
          f"benign {sum(1 for _, t in bench if t == 0)} "
          f"malignant {sum(1 for _, t in bench if t == 1)}", flush=True)

    # ---- wild board: the assets zip holds all 199 films + labels ----
    split = json.loads(SPLIT.read_text())
    wild_root = SPLIT.parent
    zpath = next((p for p in wild_root.glob("wild_images.zip")), None)
    wdir = TMP / "wild"
    wdir.mkdir(parents=True, exist_ok=True)
    if zpath is not None:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(str(wdir))
    label_of = {}
    for p in wdir.rglob("wildlbl*.txt"):
        key = p.name.replace("wildlbl", "").replace(".txt", "").lstrip("_")
        label_of[key] = p.read_text().strip()
    wild = []
    for p in sorted(wdir.rglob("wildimg*.png")):
        key = p.name.replace("wildimg", "").replace(".png", "").lstrip("_")
        lab = label_of.get(key)
        if lab is None:
            continue
        wild.append((str(p), 1 if lab == "malignant" else 0))
    print(f"wild films: {len(wild)}", flush=True)

    # ---- third source ----
    man = json.loads(MAN.read_text())
    wkmd = [(str(MAN.parent / m["file"]), 1 if m["label"] == "malignant" else 0) for m in man]
    print(f"third-source films: {len(wkmd)}", flush=True)
    mark("boards")

    for tag, items in (("bench", bench), ("wild", wild), ("wkmd", wkmd)):
        recs = []
        t0 = time.time()
        for start in range(0, len(items), BATCH):
            chunk = items[start:start + BATCH]
            pils = [Image.open(p).convert("RGB") for p, _ in chunk]
            with torch.inference_mode():
                xa = torch.stack([evA(im) for im in pils]).to(device)
                gp = F.softmax(gate(xa), 1)[:, 1].cpu()
                xb = torch.stack([evB(im) for im in pils]).to(device)
                ps = F.softmax(spec(xb), 1)[:, 1].cpu()
            for i, (p, t) in enumerate(chunk):
                recs.append({"file": Path(p).name, "path": p, "truth": int(t),
                             "gate": float(gp[i]), MODEL_TAG: float(ps[i])})
            print(f"  {tag} {min(start+BATCH, len(items))}/{len(items)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        (WORK / f"score_{tag}.json").write_text(json.dumps(recs))
    mark("done")
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print("FATAL:\n" + tb, flush=True)
        try:
            (WORK / "ERROR.txt").write_text(tb)
        except Exception:
            pass
        sys.exit(1)
