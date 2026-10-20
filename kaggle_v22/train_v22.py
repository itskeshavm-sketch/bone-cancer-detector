"""Kaggle GPU V22: lesion-aware + higher-resolution specialist (Model 2).

Hypothesis: the specialist's remaining errors come from two things we can attack
without new data - (1) it must FIND the lesion itself inside a whole radiograph,
and (2) at 456px a small lesion is only a fraction of the tensor.

Design:
  * 50% of training samples are presented as a LESION CROP (bbox expanded 1.6x,
    squared), 50% as the full image. Crop is an augmentation, never a replacement:
    evaluation and the demo always use the full image, because nobody boxes a
    lesion in the real world.
  * Resolution 456px -> 600px. Batch 8; if the GPU OOMs the run automatically
    halves the batch and switches on gradient accumulation x2, so the effective
    batch size never changes.
  * 40 epochs (V21's own curve peaks at epoch 39), CE + label_smoothing 0.05,
    AdamW 3e-4, cosine, AMP. Best-val checkpoint, SWA compared on val.
  * Crop source is auto-detected at runtime:
      - YOLO rows "class cx cy w h" in the label .txt  -> true bbox
      - otherwise -> no crop (pure high-resolution run)
    The run states which mode it used in metrics.json, so a late label-format
    surprise cannot invalidate it.

Gate is reused (stageA18, 99.37% val). Chained 5-view TTA eval: val threshold
sweep, then test once.

Module-level so the crop logic is unit-testable on CPU without a GPU.
"""
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

random.seed(22)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# crop geometry - pure functions, unit-testable
# ----------------------------------------------------------------------------


def parse_yolo_label(text):
    """Parse a label .txt into {'cls','cx','cy','w','h'} or None.

    The dataset stores YOLO-SEGMENTATION rows, not boxes:
        class x1 y1 x2 y2 ... xn yn
    (verified against real samples - e.g. a 831-byte benign label holds one class
    token followed by ~60 coordinate pairs). A plain YOLO box row
    ("class cx cy w h") is also accepted for robustness.

    A polygon is reduced to its axis-aligned bounding box, and multiple rows in
    one file are UNIONED so every annotated lesion is inside the crop.

    Returns None for empty files, class-id-only labels, or unparseable content.
    """
    if not text:
        return None
    cls = None
    boxes = []
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            # plain YOLO box: cx cy w h
            cx, cy, w, h = vals
            if w <= 0 or h <= 0:
                continue
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            # polygon: x1 y1 x2 y2 ...
            xs = vals[0::2]
            ys = vals[1::2]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        else:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            pass
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    if x1 <= x0 or y1 <= y0:
        return None
    return {"cls": cls if cls is not None else 0,
            "cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0,
            "w": x1 - x0, "h": y1 - y0}


def crop_box(size, cx, cy, bw, bh, margin=1.6, min_frac=0.08):
    """Square crop box in pixels around a normalised YOLO box.

    Expands by `margin`, squares it, enforces a minimum side of `min_frac` of the
    image so a tiny annotation cannot produce a degenerate extreme zoom, and
    shifts the box back inside the image instead of shrinking it (keeps the zoom
    level). Returns (left, top, right, bottom) or None.
    """
    W, H = size
    x0 = (cx - bw / 2.0) * W
    y0 = (cy - bh / 2.0) * H
    x1 = (cx + bw / 2.0) * W
    y1 = (cy + bh / 2.0) * H
    side = max(x1 - x0, y1 - y0) * margin
    side = max(side, min_frac * max(W, H))
    cx_p, cy_p = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(round(cx_p - side / 2.0))
    top = int(round(cy_p - side / 2.0))
    right = int(round(cx_p + side / 2.0))
    bottom = int(round(cy_p + side / 2.0))
    if right - left > W:
        left, right = 0, W
    if bottom - top > H:
        top, bottom = 0, H
    if left < 0:
        left, right = 0, min(W, right - left)
    if top < 0:
        top, bottom = 0, min(H, bottom - top)
    if right > W:
        right, left = W, max(0, left - (right - W))
    if bottom > H:
        bottom, top = H, max(0, top - (bottom - H))
    if right - left < 16 or bottom - top < 16:
        return None
    return (left, top, right, bottom)


def apply_crop(pil, box):
    """Crop a PIL image to `box`; returns the original on any failure."""
    if box is None:
        return pil
    try:
        return pil.crop(box)
    except Exception:
        return pil


# ----------------------------------------------------------------------------
# io helpers
# ----------------------------------------------------------------------------

ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TMP = Path("/tmp/kwork22")
MCLS = {0: "benign", 1: "malignant"}


def find_main():
    for p in ROOT.rglob("train/images"):
        if p.is_dir() and (p.parent / "labels").is_dir():
            return p.parent.parent
    return None


def mark(s):
    print("STAGE:", s, flush=True)
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "STAGE.txt").write_text(s)
    except Exception:
        pass


def link(src, dst):
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main_label(img, lbl_dir):
    label = lbl_dir / (img.stem + ".txt")
    if not label.exists():
        label = lbl_dir / (img.name + ".txt")
        if not label.exists():
            c = list(lbl_dir.glob(img.stem + "*"))
            label = c[0] if c else None
    if label is None or not label.exists() or label.stat().st_size == 0:
        return "normal"
    try:
        return MCLS.get(int(float(open(label).read().strip().split()[0])), "malignant")
    except Exception:
        return "normal"


def aug_train(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])


def aug_eval(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ----------------------------------------------------------------------------
# crop-aware dataset
# ----------------------------------------------------------------------------

class SubtypeDataset(Dataset):
    """Yields (tensor, label). With probability `crop_prob` returns a lesion crop
    instead of the full image, but ONLY when a crop source is available."""

    def __init__(self, root, transform, boxes=None, crop_prob=0.5):
        self.folder = datasets.ImageFolder(str(root))
        self.samples = list(self.folder.samples)
        self.targets = list(self.folder.targets)
        self.transform = transform
        self.boxes = boxes or {}
        self.crop_prob = crop_prob
        self.classes = list(self.folder.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (64, 64), 0)
        if self.boxes and random.random() < self.crop_prob:
            pil = apply_crop(pil, self.boxes.get(path))
        return self.transform(pil), label


# ----------------------------------------------------------------------------
# main training program
# ----------------------------------------------------------------------------

def main():
    print("=== DIAGNOSTICS ===", flush=True)
    print("cuda:", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        print("no GPU - aborting", flush=True)
        sys.exit(2)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    TMP.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    MAIN_SRC = find_main()
    GA = next((p for p in ROOT.rglob("stageA18.pt") if p.is_file()), None)
    print("MAIN:", MAIN_SRC, "GATE:", GA, flush=True)
    if MAIN_SRC is None or GA is None:
        print("missing main dataset or gate weights - aborting", flush=True)
        sys.exit(2)

    from torch.cuda.amp import GradScaler, autocast
    from torchvision.models import (efficientnet_b3, efficientnet_b6,
                                    EfficientNet_B3_Weights, EfficientNet_B6_Weights)
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from tqdm import tqdm

    # ---------------- stage the data (proven V21 spine) ----------------
    D3 = TMP / "data3"
    for split in ["train", "val", "test"]:
        for c in ["normal", "benign", "malignant"]:
            (D3 / split / c).mkdir(parents=True, exist_ok=True)
        for img in (MAIN_SRC / split / "images").iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                link(img, D3 / split / main_label(img, MAIN_SRC / split / "labels") / img.name)
    mark("main-staged")

    test_hashes = {md5(i) for c in ["normal", "benign", "malignant"]
                   for i in (D3 / "test" / c).glob("*")}
    dropped = 0
    for base in [D3 / "train", D3 / "val"]:
        for img in list(base.rglob("*")):
            if img.is_file() and md5(img) in test_hashes:
                img.unlink()
                dropped += 1
    print("dedup dropped:", dropped, flush=True)

    SUB = TMP / "data_sub"
    for split in ["train", "val"]:
        for c in ["benign", "malignant"]:
            (SUB / split / c).mkdir(parents=True, exist_ok=True)
            for img in (D3 / split / c).glob("*"):
                link(img, SUB / split / c / img.name)
    ntr = sum(1 for _ in (SUB / "train").rglob("*") if _.is_file())
    nva = sum(1 for _ in (SUB / "val").rglob("*") if _.is_file())
    print(f"subtype train={ntr} val={nva}", flush=True)
    if ntr < 2000 or nva < 400:
        print("unexpected subtype counts - aborting", flush=True)
        sys.exit(2)
    mark("subtype-staged")

    # ---------------- detect the crop source ----------------
    # ~64% of label files are tumour annotations (the other 36% are the 0-byte
    # normals), so a parse ratio of 30%+ means the polygon format is present.
    label_dir = MAIN_SRC / "train" / "labels"
    probe_files = list(label_dir.glob("*.txt"))
    parsed = 0
    for lp in probe_files:
        if parse_yolo_label(lp.read_text(errors="replace")) is not None:
            parsed += 1
    bbox_ratio = parsed / max(len(probe_files), 1)
    print(f"label format probe: {parsed}/{len(probe_files)} parse as annotation "
          f"(ratio {bbox_ratio:.2f})", flush=True)

    crop_mode = "bbox" if bbox_ratio >= 0.30 else "none"
    boxes = {}
    if crop_mode == "bbox":
        for split in ["train", "val"]:
            ldir = MAIN_SRC / split / "labels"
            img_dir = MAIN_SRC / split / "images"
            for lp in ldir.glob("*.txt"):
                info = parse_yolo_label(lp.read_text(errors="replace"))
                if info is None:
                    continue
                cands = list(img_dir.glob(lp.stem + ".*"))
                if not cands:
                    continue
                try:
                    with Image.open(cands[0]) as im:
                        box = crop_box(im.size, info["cx"], info["cy"], info["w"], info["h"])
                except Exception:
                    box = None
                if box:
                    boxes[str(cands[0])] = box
    print(f"crop mode={crop_mode} boxes={len(boxes)}", flush=True)
    mark("crop-source")

    # ---------------- build loaders ----------------
    # 600px is 1.73x the pixels of the proven 456px V21 run; batch 6 with
    # gradient accumulation x2 keeps the EFFECTIVE batch at 12 (identical to V21)
    # while cutting peak memory below V21's. If it still OOMs, batch drops to 3
    # and accumulation rises to 4 - effective batch never changes.
    IMG, BATCH, ACCUM, EPOCHS, CROP_PROB = 600, 6, 2, 40, 0.5
    tr = SubtypeDataset(SUB / "train", aug_train(IMG), boxes, CROP_PROB)
    va = SubtypeDataset(SUB / "val", aug_eval(IMG), {}, 0.0)  # eval is ALWAYS full image
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True,
                       drop_last=True, persistent_workers=True)
    va_ld = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    print(f"train={len(tr)} val={len(va)} steps/ep={len(tr_ld)} img={IMG} "
          f"batch={BATCH} accum={ACCUM} effective={BATCH*ACCUM}", flush=True)

    model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 2))
    model = model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler()
    accum = ACCUM

    def evaluate():
        P, T = [], []
        model.eval()
        with torch.no_grad():
            for x, y in va_ld:
                P.extend(model(x.to(device)).argmax(1).cpu().tolist())
                T.extend(y.tolist())
        return accuracy_score(T, P)

    def train_epoch(ep):
        model.train()
        t0, co, to = time.time(), 0, 0
        opt.zero_grad()
        step = 0
        for x, y in tqdm(tr_ld, desc=f"B22 {ep}/{EPOCHS}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                loss = crit(logits, y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            step += 1
            co += (logits.argmax(1) == y).sum().item()
            to += x.size(0)
        if step % accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        return co / to, time.time() - t0

    best, out = 0.0, WORK / "specB22.pt"
    out_swa = WORK / "specB22_swa.pt"
    swa_state, swa_n = None, 0
    swa_start = EPOCHS - 12 + 1
    hist = []

    for ep in range(1, EPOCHS + 1):
        try:
            tr_acc, ep_s = train_epoch(ep)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at epoch {ep} -> batch 3, gradient accumulation 4 "
                  f"(effective batch unchanged)", flush=True)
            torch.cuda.empty_cache()
            opt.zero_grad()
            tr_ld = DataLoader(tr, batch_size=3, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, persistent_workers=True)
            va_ld = DataLoader(va, batch_size=3, shuffle=False, num_workers=4,
                               pin_memory=True)
            accum = 4
            tr_acc, ep_s = train_epoch(ep)
        sched.step()
        vacc = evaluate()
        hist.append({"ep": ep, "train": tr_acc, "val": vacc, "secs": ep_s})
        print(f"B22 Ep{ep}: train={tr_acc:.4f} val={vacc:.4f} ({ep_s:.0f}s) "
              f"peakmem={torch.cuda.max_memory_allocated()/2**30:.1f}GB accum={accum}",
              flush=True)
        if vacc > best:
            best = vacc
            torch.save({"model": model.state_dict(), "acc": vacc, "crop_mode": crop_mode},
                       str(out))
            print(f"  *** saved ({vacc:.4f})", flush=True)
        if ep >= swa_start:
            with torch.no_grad():
                msd = model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.clone() for k, v in msd.items() if v.is_floating_point()}
                    swa_n = 1
                else:
                    swa_n += 1
                    for k, v in msd.items():
                        if v.is_floating_point():
                            swa_state[k].mul_(swa_n - 1).add_(v).div_(swa_n)
    print(f"B22 best val: {best:.4f}", flush=True)

    swa_acc = 0.0
    if swa_state is not None:
        with torch.no_grad():
            merged = dict(model.state_dict())
            for k, v in swa_state.items():
                merged[k] = v.to(model.state_dict()[k].dtype)
            backup = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(merged)
            swa_acc = evaluate()
            model.load_state_dict(backup)
        torch.save({"model": merged, "acc": swa_acc, "crop_mode": crop_mode}, str(out_swa))
        print(f"B22 SWA(last12): val={swa_acc:.4f}", flush=True)
    final = out_swa if swa_acc > best else out
    print(f"B22 choice: {final} (raw={best:.4f} swa={swa_acc:.4f})", flush=True)
    torch.save(torch.load(final), str(WORK / "specB22_final.pt"))
    mark("spec-done")

    # ---------------- chained TTA eval with the reused gate ----------------
    mA = efficientnet_b3(weights=None)
    mA.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(mA.classifier[1].in_features, 2))
    mA.load_state_dict(torch.load(str(GA), map_location=device)["model"])
    mA = mA.to(device).eval()
    mB = efficientnet_b6(weights=None)
    mB.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(mB.classifier[1].in_features, 2))
    mB.load_state_dict(torch.load(str(final), map_location=device)["model"])
    mB = mB.to(device).eval()

    import torchvision.transforms.functional as TF
    names3 = ["benign", "malignant", "normal"]
    evA, evB = aug_eval(300), aug_eval(IMG)

    def tta_views(pil):
        return [pil, TF.hflip(pil), TF.rotate(pil, 10), TF.rotate(pil, -10),
                TF.rotate(TF.hflip(pil), 10)]

    def chained_eval(split, thresh):
        P, T = [], []
        with torch.no_grad():
            for ci, cn in enumerate(names3):
                for img_p in sorted((D3 / split / cn).glob("*")):
                    pil = Image.open(img_p).convert("RGB")
                    tp = sum(torch.softmax(mA(evA(v).unsqueeze(0).to(device)), 1)[0, 1].item()
                             for v in tta_views(pil)) / 5
                    if tp < thresh:
                        P.append(2)
                    else:
                        bp = torch.zeros(2)
                        for v in tta_views(pil):
                            bp += torch.softmax(mB(evB(v).unsqueeze(0).to(device)), 1)[0].cpu()
                        P.append(int((bp / 5).argmax()))
                    T.append(ci)
        return accuracy_score(T, P), P, T

    best_thr, best_v = 0.5, 0.0
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        vacc, _, _ = chained_eval("val", thr)
        print(f"val thresh {thr}: {vacc:.4f}", flush=True)
        if vacc > best_v:
            best_v, best_thr = vacc, thr
    acc, P, T = chained_eval("test", best_thr)
    print(f"CHAINED TEST: {acc:.4f} (thr {best_thr})", flush=True)
    print(classification_report(T, P, target_names=names3, zero_division=0), flush=True)

    (WORK / "metrics.json").write_text(json.dumps({
        "v": 22, "crop_mode": crop_mode, "boxes": len(boxes), "crop_prob": CROP_PROB,
        "img_size": IMG, "batch": BATCH, "epochs": EPOCHS, "accum": accum,
        "spec_val": best, "swa_val": swa_acc, "gate_thresh": best_thr,
        "val_tta": best_v, "test_tta": acc,
        "report": classification_report(T, P, target_names=names3,
                                        zero_division=0, output_dict=True),
        "confusion": confusion_matrix(T, P).tolist(),
        "hist": hist}, indent=1))
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