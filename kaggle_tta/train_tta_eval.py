"""Kaggle GPU: score the TTA variants on the benchmark and the third-source board.

The local CPU has ONE core and is saturated by another workload - scoring 614
films x 6 views through an EfficientNet-B6 at 456px was measured at 78 s/film,
i.e. 7 hours for what the GPU does in about two minutes.

Writes per-film probabilities for four inference variants:
  single-view      the path every number in the project was measured on
  5-view TTA       identity, h-flip, rotate +/-10, h-flip+rotate+10
  6-view TTA+zoom  the 5 views plus a 1.25x centre zoom
and the gate probability for both single-view and TTA, so the champion rule can
be applied to each variant across all three boards.

Output: tta_probs.json in the kernel output, pulled and scored locally.

Usage: push with kaggle_out_probe/push_any.py kaggle_tta
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, efficientnet_b6
from torchvision.transforms import functional as TF

WORK = Path("/kaggle/working")
TMP = Path("/tmp/kworktta")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH = 24
device = torch.device("cuda")


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


def views5(pil):
    return [pil, TF.hflip(pil), TF.rotate(pil, 10), TF.rotate(pil, -10),
            TF.rotate(TF.hflip(pil), 10)]


def zoom(pil, frac=0.125):
    w, h = pil.size
    return pil.crop((int(w * frac), int(h * frac), int(w * (1 - frac)), int(h * (1 - frac))))


def main():
    print("cuda:", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        sys.exit(2)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    TMP.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    MAIN = find_main()
    ASSETS = next((p for p in Path("/kaggle/input").rglob("manifest.json")
                   if p.parent.name == "tta-assets"), None)
    SB25 = next((p for p in Path("/kaggle/input").rglob("stageB25.pt") if p.is_file()), None)
    GA = next((p for p in Path("/kaggle/input").rglob("stageA18.pt") if p.is_file()), None)
    print("MAIN:", MAIN, "| ASSETS:", ASSETS, "| B25:", SB25, "| GATE:", GA, flush=True)
    if not (MAIN and ASSETS and SB25 and GA):
        print("missing inputs - aborting", flush=True)
        sys.exit(2)
    mark("inputs")

    gate = load(GA, efficientnet_b3)
    spec = load(SB25, efficientnet_b6)
    evA, evB = tf(300), tf(456)

    # ---- board 1: the benchmark test split ----
    MCLS = {0: "benign", 1: "malignant"}
    bench = []
    for split in ["test"]:
        for cls in ["normal", "benign", "malignant"]:
            d = MAIN / split / "images"
            if not d.is_dir():
                continue
            lbl_dir = MAIN / split / "labels"
            for img in sorted(d.iterdir()):
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lb = lbl_dir / (img.stem + ".txt")
                if not lb.exists():
                    lb = lbl_dir / (img.name + ".txt")
                lab = None
                if lb.exists():
                    try:
                        lab = MCLS.get(int(float(open(lb).read().strip().split()[0])))
                    except Exception:
                        lab = None
                if cls == "normal":
                    lab = 2
                elif lab is None:
                    lab = 0
                bench.append((str(img), lab))
    print(f"benchmark films: {len(bench)}", flush=True)

    # ---- board 2: the third-source pool ----
    man = json.loads(ASSETS.read_text())
    wkmd = [(str(ASSETS.parent / m["file"]), 1 if m["label"] == "malignant" else 0)
            for m in man]
    print(f"third-source films: {len(wkmd)}", flush=True)
    mark("boards")

    out = {}
    for tag, items in (("bench", bench), ("wkmd", wkmd)):
        recs = []
        t0 = time.time()
        for start in range(0, len(items), BATCH):
            chunk = items[start:start + BATCH]
            pils = [Image.open(p).convert("RGB") for p, _ in chunk]
            views = [views5(p) for p in pils]
            zooms = [zoom(p) for p in pils]
            with torch.inference_mode():
                ga = []
                sa = []
                for v in views:
                    xa = torch.stack([evA(x) for x in v]).to(device)
                    gb = F.softmax(gate(xa), 1)[:, 1]
                    xb = torch.stack([evB(x) for x in v]).to(device)
                    sb = F.softmax(spec(xb), 1)[:, 1]
                    ga.append(gb.cpu())
                    sa.append(sb.cpu())
                xz = torch.stack([evB(z) for z in zooms]).to(device)
                sz = F.softmax(spec(xz), 1)[:, 1].cpu()
            ga = torch.stack(ga).numpy()
            sa = torch.stack(sa).numpy()
            for i, (p, t) in enumerate(chunk):
                recs.append({
                    "file": p, "truth": t,
                    "gate_1": float(ga[0][i]), "gate_tta": float(ga[:, i].mean()),
                    "mal_1": float(sa[0][i]), "mal_tta": float(sa[:, i].mean()),
                    "mal_tta_zoom": float((sa[:, i].sum() + sz[i]) / (sa.shape[0] + 1)),
                })
            print(f"  {tag} {min(start+BATCH, len(items))}/{len(items)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        out[tag] = recs
        (WORK / f"tta_probs_{tag}.json").write_text(json.dumps(recs))
    (WORK / "tta_probs.json").write_text(json.dumps(out))
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
