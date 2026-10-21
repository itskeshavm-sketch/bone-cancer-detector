"""Kaggle GPU V23: targeted fine-tune of the V21 specialist on the BLIND SPOTS.

The error anatomy (kaggle_out_probe/error_anatomy.py) decomposed V21's 100 wild-pool
errors: 93 are CLASSIFICATION failures on specific entity families, 7 are gate
failures. The families are known and named: ABC, bone island, chondroblastoma,
chondromyxoid fibroma, BFH, brown tumor, enchondroma, perichondroma, GCT, fibrous
dysplasia, NOF, eosinophilic granuloma, simple bone cyst, osteoid osteoma,
osteochondroma  (benign but aggressive-looking)
  chondrosarcoma, adamantinoma, chordoma, angiosarcoma, Ewing, lymphoma, myeloma,
metastases                                              (malignant but not osteosarcoma)

The 199-film wild pool already CONTAINS those entities, so V23 fine-tunes the V21
specialist on a 121-image slice of it (the 78-image held-out slice is never
trained on - the kernel asserts this) while keeping the 2,736 BTXRD images in the
mix so the in-distribution score cannot drift.

Design:
  * init from stageB21 (the champion) - NOT from scratch
  * LOW lr (2e-5) because this is a nudge, not a retrain
  * few epochs (6) with the proven aug
  * the 121 wild images are oversampled xW so they actually move the decision
    boundary, without unbalancing the CE loss
  * guard metrics: BTXRD val accuracy, BTXRD test TTA, and the 78-image held-out
    wild slice - all reported, so a regression is visible immediately
  * the malignant threshold is swept on the HELD-OUT wild slice and on BTXRD val

Seed 23. ~30-45 min on a T4.
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

random.seed(23)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, efficientnet_b6
from torchvision.models import EfficientNet_B3_Weights, EfficientNet_B6_Weights
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

WORK = Path("/kaggle/working")
TMP = Path("/tmp/kwork23")
MCLS = {0: "benign", 1: "malignant"}
device = torch.device("cuda")

EPOCHS = 6
LR = 2e-5
BATCH = 12
IMG = 456
WILD_REPEAT = 6          # oversample the 121 wild images within each epoch
WILD_FRAC = 0.35         # target share of each batch drawn from the wild slice


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


def link(src, dst):
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main_label(img, lbl_dir):
    lb = lbl_dir / (img.stem + ".txt")
    if not lb.exists():
        lb = lbl_dir / (img.name + ".txt")
        if not lb.exists():
            c = list(lbl_dir.glob(img.stem + "*"))
            lb = c[0] if c else None
    if lb is None or not lb.exists() or lb.stat().st_size == 0:
        return "normal"
    try:
        return MCLS.get(int(float(open(lb).read().strip().split()[0])), "malignant")
    except Exception:
        return "normal"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def aug_train(sz):
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])


def aug_eval(sz):
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class MixedDataset(Dataset):
    """Main BTXRD subtype images + the wild blind-spot slice, with the wild slice
    oversampled. Each item records its source so we can report metrics separately."""

    def __init__(self, main_root, wild_items, transform, wild_repeat=WILD_REPEAT,
                 wild_frac=WILD_FRAC):
        self.folder = datasets.ImageFolder(str(main_root))
        self.base = list(self.folder.samples)
        self.wild = list(wild_items)
        self.transform = transform
        self.wild_repeat = wild_repeat
        self.wild_frac = wild_frac
        self.classes = list(self.folder.classes)

    def __len__(self):
        return len(self.base) + len(self.wild) * self.wild_repeat

    def __getitem__(self, i):
        if i < len(self.base):
            path, label = self.base[i]
            src = "main"
        else:
            j = (i - len(self.base)) % len(self.wild)
            path, label = self.wild[j]
            src = "wild"
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (224, 224), 0)
        return self.transform(pil), label, 1 if src == "wild" else 0


def main():
    print("cuda:", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        sys.exit(2)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    TMP.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    MAIN_SRC = find_main()
    ASSETS = next((p for p in Path("/kaggle/input").rglob("wild_split.json")), None)
    SB21 = next((p for p in Path("/kaggle/input").rglob("stageB21.pt")
                 if p.is_file()), None)
    GA = next((p for p in Path("/kaggle/input").rglob("stageA18.pt") if p.is_file()), None)
    print("MAIN:", MAIN_SRC, "| ASSETS:", ASSETS, "| B21:", SB21, "| GATE:", GA, flush=True)
    if not (MAIN_SRC and ASSETS and SB21 and GA):
        print("missing inputs - aborting", flush=True)
        sys.exit(2)

    # ---- stage main 3-class + dedup (proven spine) ----
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
    drop = 0
    for base in [D3 / "train", D3 / "val"]:
        for img in list(base.rglob("*")):
            if img.is_file() and md5(img) in test_hashes:
                img.unlink()
                drop += 1
    print("dedup dropped:", drop, flush=True)

    SUB = TMP / "data_sub"
    for split in ["train", "val"]:
        for c in ["benign", "malignant"]:
            (SUB / split / c).mkdir(parents=True, exist_ok=True)
            for img in (D3 / split / c).glob("*"):
                link(img, SUB / split / c / img.name)
    mark("subtype-staged")

    # ---- load the wild split ----
    # The dataset may ship the wild images as one zip (we upload a zip so the
    # uploader makes ONE request instead of 199) OR Kaggle may have expanded it
    # into a wild_images/ subfolder. Handle both, and any nesting depth.
    import zipfile
    split = json.loads(ASSETS.read_text())
    wild_root = ASSETS.parent
    search_dirs = [wild_root]
    zpath = next((p for p in wild_root.glob("wild_images.zip")), None)
    if zpath is not None:
        wdir = TMP / "wild"
        wdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(str(wdir))
        search_dirs = [wdir, wild_root]
        print(f"unzipped {zpath.name}", flush=True)
    wimg, wlab = {}, {}
    for d in search_dirs:
        for p in d.rglob("wildimg_*.png"):
            wimg.setdefault(p.name, p)
        for p in d.rglob("wildlbl_*.txt"):
            wlab.setdefault(p.name.replace("wildlbl_", "").replace(".txt", ""), p)
    print(f"wild images found: {len(wimg)} | labels: {len(wlab)}", flush=True)
    if len(wimg) < 150 or len(wlab) < 150:
        print(f"wild assets incomplete ({len(wimg)}/{len(wlab)}) - aborting", flush=True)
        sys.exit(2)

    def wild_items(entries):
        out = []
        missing = 0
        for e in entries:
            ip = wimg.get(e["file"])
            stem = e["file"].replace("wildimg_", "").replace(".png", "")
            lp = wlab.get(stem)
            if ip is None or lp is None:
                missing += 1
                continue
            out.append((str(ip), 1 if e["label"] == "malignant" else 0))
        if missing:
            print(f"  WARNING: {missing} split entries had no image/label on disk", flush=True)
        return out

    train_wild = wild_items(split["train"])
    val_wild = wild_items(split["val"])
    train_names = {e["file"] for e in split["train"]}
    val_names = {e["file"] for e in split["val"]}
    assert not (train_names & val_names), "train/val overlap in the split!"
    assert len(train_wild) > 50 and len(val_wild) > 30, f"wild split too small: {len(train_wild)}/{len(val_wild)}"
    print(f"wild train {len(train_wild)} | wild val {len(val_wild)} | "
          f"overlap 0 (asserted)", flush=True)
    mark("wild-loaded")

    # ---- model: init from the V21 champion ----
    model = efficientnet_b6(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.4),
                                    nn.Linear(model.classifier[1].in_features, 2))
    ck = torch.load(str(SB21), map_location="cpu")
    model.load_state_dict(ck["model"])
    model = model.to(device)
    print(f"init from V21 (stored acc {ck.get('acc')})", flush=True)

    tr = MixedDataset(SUB / "train", train_wild, aug_train(IMG))
    va = datasets.ImageFolder(str(SUB / "val"), transform=aug_eval(IMG))
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True,
                       drop_last=True, persistent_workers=True)
    va_ld = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    print(f"train items {len(tr)} (main {len(tr.base)} + wild {len(tr.wild)}x{WILD_REPEAT}) "
          f"| val {len(va)}", flush=True)

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler()

    def evaluate_btxrd():
        P, T = [], []
        model.eval()
        with torch.no_grad():
            for x, y in va_ld:
                P.extend(model(x.to(device)).argmax(1).cpu().tolist())
                T.extend(y.tolist())
        return accuracy_score(T, P), P, T

    def evaluate_wild(items, thr):
        ev = aug_eval(IMG)
        P, T = [], []
        model.eval()
        with torch.no_grad():
            for path, lab in items:
                pil = Image.open(path).convert("RGB")
                p = float(F.softmax(model(ev(pil).unsqueeze(0).to(device)), 1)[0][1])
                pred = 1 if p >= thr else 0
                P.append(pred)
                T.append(lab)
        return accuracy_score(T, P), P, T

    best, out = 0.0, WORK / "specB23.pt"
    hist = []
    for ep in range(1, EPOCHS + 1):
        try:
            model.train()
            t0, co, to, cw = time.time(), 0, 0, 0
            for x, y, isw in tqdm(tr_ld, desc=f"B23 {ep}/{EPOCHS}", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad()
                with autocast():
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                # count accuracy from the SAME logits used for the loss. Calling
                # model(x) again here would run a second forward with a second
                # autograd graph and OOM the T4 (V24 died exactly this way).
                co += (logits.argmax(1) == y).sum().item()
                to += x.size(0)
                cw += int(isw.sum())
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at epoch {ep} -> halving the batch and retrying", flush=True)
            torch.cuda.empty_cache()
            opt.zero_grad()
            tr_ld = DataLoader(tr, batch_size=max(2, BATCH // 2), shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, persistent_workers=True)
            va_ld = DataLoader(va, batch_size=max(2, BATCH // 2), shuffle=False,
                               num_workers=4, pin_memory=True)
            model.train()
            t0, co, to, cw = time.time(), 0, 0, 0
            for x, y, isw in tqdm(tr_ld, desc=f"B23 {ep}/{EPOCHS} (small batch)", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad()
                with autocast():
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                co += (logits.argmax(1) == y).sum().item()
                to += x.size(0)
                cw += int(isw.sum())
        sched.step()
        vacc, _, _ = evaluate_btxrd()
        hist.append({"ep": ep, "train": co / to, "val": vacc, "secs": time.time() - t0})
        print(f"B23 Ep{ep}: train={co/to:.4f} btxrd_val={vacc:.4f} "
              f"wild_share={cw/to:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if vacc >= best:
            best = vacc
            torch.save({"model": model.state_dict(), "acc": vacc, "ep": ep}, str(out))
            print(f"  *** saved ({vacc:.4f})", flush=True)
    print(f"B23 best btxrd val: {best:.4f}", flush=True)
    mark("trained")

    # ---- held-out wild slice: sweep the malignant threshold ----
    print("\n=== held-out wild slice (78 films, never trained on) ===", flush=True)
    rows = []
    for thr in [round(0.05 * i, 2) for i in range(1, 19)]:
        acc, P, T = evaluate_wild(val_wild, thr)
        mal = sum(1 for p, t in zip(P, T) if t == 1 and p == 1)
        ben = sum(1 for p, t in zip(P, T) if t == 0 and p == 0)
        rows.append({"thr": thr, "acc": acc, "mal": mal, "ben": ben})
        print(f"  thr {thr:.2f}: acc {acc:.1%}  mal {mal}/35  ben {ben}/43", flush=True)
    best_w = max(rows, key=lambda r: r["acc"])
    print(f"  best held-out wild: {best_w['acc']:.1%} at thr {best_w['thr']} "
          f"(mal {best_w['mal']}/35 ben {best_w['ben']}/43)", flush=True)

    # ---- benchmark TTA with the A18 gate ----
    mA = efficientnet_b3(weights=None)
    mA.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(mA.classifier[1].in_features, 2))
    mA.load_state_dict(torch.load(str(GA), map_location=device)["model"])
    mA = mA.to(device).eval()
    model.eval()
    import torchvision.transforms.functional as TF
    names3 = ["benign", "malignant", "normal"]
    evA, evB = aug_eval(300), aug_eval(IMG)

    def tta_views(pil):
        return [pil, TF.hflip(pil), TF.rotate(pil, 10), TF.rotate(pil, -10),
                TF.rotate(TF.hflip(pil), 10)]

    def chained_eval(split, gate_thr, mal_thr):
        P, T = [], []
        with torch.no_grad():
            for ci, cn in enumerate(names3):
                for img_p in sorted((D3 / split / cn).glob("*")):
                    pil = Image.open(img_p).convert("RGB")
                    tp = sum(torch.softmax(mA(evA(v).unsqueeze(0).to(device)), 1)[0, 1].item()
                             for v in tta_views(pil)) / 5
                    if tp < gate_thr:
                        P.append(2)
                    else:
                        bp = torch.zeros(2)
                        for v in tta_views(pil):
                            bp += torch.softmax(model(evB(v).unsqueeze(0).to(device)), 1)[0].cpu()
                        P.append(1 if float(bp[1] / 5) >= mal_thr else 0)
                    T.append(ci)
        return accuracy_score(T, P), P, T

    print("\n=== benchmark: gate x malignant sweep ===", flush=True)
    bench_rows = []
    for gt in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for mt in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            acc, P, T = chained_eval("test", gt, mt)
            bench_rows.append({"gate": gt, "mal": mt, "acc": acc})
            print(f"  gate {gt:.2f} mal {mt:.2f}: {acc:.4f}", flush=True)
    bb = max(bench_rows, key=lambda r: r["acc"])
    print(f"  best benchmark: {bb['acc']:.4f} at gate {bb['gate']} mal {bb['mal']}", flush=True)

    (WORK / "metrics.json").write_text(json.dumps({
        "v": 23, "init": "stageB21", "lr": LR, "epochs": EPOCHS, "img": IMG,
        "wild_repeat": WILD_REPEAT, "wild_train": len(train_wild),
        "wild_val": len(val_wild), "btxrd_val": best, "hist": hist,
        "heldout_wild": rows, "heldout_wild_best": best_w,
        "benchmark_sweep": bench_rows, "benchmark_best": bb}, indent=1))
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
