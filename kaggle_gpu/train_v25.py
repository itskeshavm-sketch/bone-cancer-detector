"""Kaggle GPU V25: V24's recipe + the 38 harvested blind-spot radiographs.

V24 proved the mechanism: fine-tuning the V21 champion on 121 real-world images
of the entity families the error anatomy flagged (ABC, bone island,
chondroblastoma, CMF, BFH, chondrosarcoma, adamantinoma, chordoma,
angiosarcoma...) moved the wild-pool score from 49.2% to 67.8% with the
benchmark unchanged.

V25 changes exactly ONE variable: it adds 38 more radiographs of those same
entities, harvested from bonetumor.org diagnosis pages and individually
verified to be plain radiographs (CT / MRI / histology / surgical photos and
infographics were rejected, some automatically and 11 by eye).

Same recipe as V24 otherwise: init from stageB21, lr 2e-5, 6 epochs, wild slice
oversampled x6, A18 gate reused, held-out 78-film wild slice byte-identical to
V24's so the comparison is like-for-like, plus the BTXRD benchmark as guard.
"""
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

random.seed(25)
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
TMP = Path("/tmp/kwork25")
MCLS = {0: "benign", 1: "malignant"}
device = torch.device("cuda")

EPOCHS = 6
LR = 2e-5
BATCH = 12
IMG = 456
WILD_REPEAT = 6
WILD_FRAC = 0.35


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
    def __init__(self, main_root, wild_items, transform, wild_repeat=WILD_REPEAT):
        self.folder = datasets.ImageFolder(str(main_root))
        self.base = list(self.folder.samples)
        self.wild = list(wild_items)
        self.transform = transform
        self.wild_repeat = wild_repeat
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
    SB21 = next((p for p in Path("/kaggle/input").rglob("stageB21.pt") if p.is_file()), None)
    GA = next((p for p in Path("/kaggle/input").rglob("stageA18.pt") if p.is_file()), None)
    print("MAIN:", MAIN_SRC, "| ASSETS:", ASSETS, "| B21:", SB21, "| GATE:", GA, flush=True)
    if not (MAIN_SRC and ASSETS and SB21 and GA):
        print("missing inputs - aborting", flush=True)
        sys.exit(2)

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

    # ---- wild assets: zip or already-expanded, any nesting depth ----
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
        for p in d.rglob("wildimg*.png"):
            wimg.setdefault(p.name, p)
        for p in d.rglob("wildlbl*.txt"):
            wlab.setdefault(p.name.replace("wildlbl", "").replace(".txt", ""), p)
    print(f"wild images found: {len(wimg)} | labels: {len(wlab)}", flush=True)
    if len(wimg) < 190 or len(wlab) < 190:
        print("wild assets incomplete - aborting", flush=True)
        sys.exit(2)

    def wild_items(entries):
        out, missing = [], 0
        for e in entries:
            ip = wimg.get(e["file"])
            # image wildimg[_2]_<stem>.png  <->  label wildlbl[_2]_<stem>.txt
            # The label map is keyed by the text between "wildlbl" and ".txt",
            # so the image stem is matched the same way. Two candidate keys are
            # tried because the optional "2" marks the harvested batch.
            raw = e["file"]
            for pref in ("wildimg2_", "wildimg_"):
                if raw.startswith(pref):
                    stem = raw[len(pref):].replace(".png", "")
                    break
            else:
                stem = raw.replace(".png", "")
            key = None
            for cand in (stem, stem[1:] if stem[:1].isdigit() else stem):
                if cand in wlab:
                    key = cand
                    break
            if ip is None or key is None:
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
    print(f"wild train {len(train_wild)} | wild val {len(val_wild)} | overlap 0 (asserted)",
          flush=True)
    mark("wild-loaded")

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
        return accuracy_score(T, P)

    def evaluate_wild(items, thr):
        ev = aug_eval(IMG)
        P, T = [], []
        model.eval()
        with torch.no_grad():
            for path, lab in items:
                pil = Image.open(path).convert("RGB")
                p = float(F.softmax(model(ev(pil).unsqueeze(0).to(device)), 1)[0][1])
                P.append(1 if p >= thr else 0)
                T.append(lab)
        return accuracy_score(T, P), P, T

    best, out = 0.0, WORK / "specB25.pt"
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, co, to, cw = time.time(), 0, 0, 0
        for x, y, isw in tqdm(tr_ld, desc=f"B25 {ep}/{EPOCHS}", leave=False):
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
        vacc = evaluate_btxrd()
        hist.append({"ep": ep, "train": co / to, "val": vacc, "secs": time.time() - t0})
        print(f"B25 Ep{ep}: train={co/to:.4f} btxrd_val={vacc:.4f} wild_share={cw/to:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vacc >= best:
            best = vacc
            torch.save({"model": model.state_dict(), "acc": vacc, "ep": ep}, str(out))
            print(f"  *** saved ({vacc:.4f})", flush=True)
    print(f"B25 best btxrd val: {best:.4f}", flush=True)
    mark("trained")

    print("\n=== held-out wild slice (78 films, never trained on) ===", flush=True)
    rows = []
    for thr in [round(0.05 * i, 2) for i in range(1, 19)]:
        acc, P, T = evaluate_wild(val_wild, thr)
        mal = sum(1 for p, t in zip(P, T) if t == 1 and p == 1)
        ben = sum(1 for p, t in zip(P, T) if t == 0 and p == 0)
        rows.append({"thr": thr, "acc": acc, "mal": mal, "ben": ben})
        print(f"  thr {thr:.2f}: acc {acc:.1%}  mal {mal}/35  ben {ben}/43", flush=True)
    best_w = max(rows, key=lambda r: r["acc"])
    print(f"  best held-out wild: {best_w['acc']:.1%} at thr {best_w['thr']}", flush=True)

    (WORK / "metrics.json").write_text(json.dumps({
        "v": 25, "init": "stageB21", "lr": LR, "epochs": EPOCHS, "img": IMG,
        "wild_repeat": WILD_REPEAT, "wild_train": len(train_wild),
        "wild_val": len(val_wild), "harvested_added": split.get("harvested_added"),
        "btxrd_val": best, "hist": hist, "heldout_wild": rows,
        "heldout_wild_best": best_w}, indent=1))
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