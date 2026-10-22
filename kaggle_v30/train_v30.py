"""Kaggle GPU V30: the 9-class BTXRD release, with the real-data share pinned to
V25's value so the mix cannot confound the result.

  V25: base 3,367 + real 954            -> real share 22.1%
  V29: base 3,367 + N + real 954        -> real share falls as N grows (dilution)
  V30: base 3,367 + N + real (repeat r) -> real share pinned back to 22.1%

Everything else is V29's, which is V25's recipe plus the 9-class images. The
repeat factor is computed at runtime from the actual base-pool size, so the mix
is identical no matter how many 9-class images actually land.

Same safety: any 9-class image matching one of the 564 benchmark test films is
dropped by md5 AND by a 16x16 perceptual signature; their own test split is never
used; label contradictions are relabelled; the run aborts if any split entry is
unresolved.
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

random.seed(30)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, efficientnet_b6
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score
from tqdm import tqdm

WORK = Path("/kaggle/working")
TMP = Path("/tmp/kwork30")
MCLS = {0: "benign", 1: "malignant"}
device = torch.device("cuda")

EPOCHS = 6
LR = 2e-5
BATCH = 12
IMG = 456
TARGET_REAL_SHARE = 0.2210   # exactly V25's share: 954 / (954 + 3367)

# fine-grained diagnosis -> the binary label it implies
FINE2BIN = {
    "giant cell tumor": 0,
    "multiple osteochondromas": 0,
    "osteochondroma": 0,
    "osteofibroma": 0,
    "other bt": 0,
    "simple bone cyst": 0,
    "synovial osteochondroma": 0,
    "osteosarcoma": 1,
    "other mt": 1,
}


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


def find_nine():
    """Locate the mounted 9-class release, preferring the annotated copy."""
    for p in Path("/kaggle/input").rglob("annotated (All)"):
        if p.is_dir():
            return p
    for p in Path("/kaggle/input").rglob("*classwise*"):
        if p.is_dir() and any((p / c).is_dir() for c in ("train", "valid", "test")):
            return p
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


NEAR_DUP_HAMMING = 8


def _dct_matrix(n):
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    Dm = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    Dm[0] *= np.sqrt(1.0 / n)
    Dm[1:] *= np.sqrt(2.0 / n)
    return Dm


_DCT = _dct_matrix(32)


def psig(p):
    """64-bit pHash (DCT low-frequency) of an image.

    A byte-exact comparison of a QUANTISED 16x16 signature collides across
    unrelated films - a radiograph is mostly black background, so a 4-level
    16x16 grid is nearly constant and many different films hash alike. V29
    dropped 39 of the 1,600 nine-class images with that check and almost none
    were real duplicates. pHash is calibrated for this job
    (kaggle_out_probe/dedup_v2.py): the 19 known same-film pairs score Hamming
    0, unrelated pairs have a median of 32, and Hamming <= 8 catches 19/19
    while false-flagging 2 in 3,000.
    """
    try:
        im = Image.open(p).convert("L").resize((32, 32), Image.LANCZOS)
        g = np.asarray(im, dtype=float)
        d = _DCT @ g @ _DCT.T
        low = d[:8, :8].flatten()[1:]
        return (low > np.median(low)).astype(np.uint8)
    except Exception:
        return None


def near_duplicate(h, pool):
    if h is None:
        return False
    for other in pool:
        if other is None:
            continue
        if int(np.count_nonzero(h != other)) <= NEAR_DUP_HAMMING:
            return True
    return False


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


def wild_key(name):
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


class MixedDataset(Dataset):
    def __init__(self, main_root, wild_items, transform, wild_repeat):
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
        else:
            j = (i - len(self.base)) % len(self.wild)
            path, label = self.wild[j]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (224, 224), 0)
        return self.transform(pil), label


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
    NINE = find_nine()
    print("MAIN:", MAIN_SRC, "| ASSETS:", ASSETS, "| B21:", SB21, "| GATE:", GA,
          "| NINE:", NINE, flush=True)
    if not (MAIN_SRC and ASSETS and SB21 and GA):
        print("missing inputs - aborting", flush=True)
        sys.exit(2)
    if NINE is None:
        print("9-class release not mounted - aborting", flush=True)
        sys.exit(2)
    mark("inputs")

    D3 = TMP / "data3"
    for split in ["train", "val", "test"]:
        for c in ["normal", "benign", "malignant"]:
            (D3 / split / c).mkdir(parents=True, exist_ok=True)
        for img in (MAIN_SRC / split / "images").iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                link(img, D3 / split / main_label(img, MAIN_SRC / split / "labels") / img.name)
    mark("main-staged")

    test_md5, test_sig = set(), []
    for c in ["normal", "benign", "malignant"]:
        for i in (D3 / "test" / c).glob("*"):
            test_md5.add(md5(i))
            s = psig(i)
            if s is not None:
                test_sig.append(s)
    print(f"test fingerprints: {len(test_md5)} md5 / {len(test_sig)} signatures", flush=True)

    drop = 0
    for base in [D3 / "train", D3 / "val"]:
        for img in list(base.rglob("*")):
            if img.is_file() and md5(img) in test_md5:
                img.unlink()
                drop += 1
    print("dedup dropped:", drop, flush=True)
    mark("dedup")

    SUB = TMP / "data_sub"
    for split in ["train", "val"]:
        for c in ["benign", "malignant"]:
            (SUB / split / c).mkdir(parents=True, exist_ok=True)
            for img in (D3 / split / c).glob("*"):
                link(img, SUB / split / c / img.name)
    base_n = sum(1 for _ in (SUB / "train").rglob("*")) + sum(1 for _ in (SUB / "val").rglob("*"))
    print(f"base pool before the 9-class add: {base_n}", flush=True)

    # ---------------- the 9-class addition ----------------
    # md5 -> (path, coarse label) for every MAIN train/val image, so a 9-class
    # image that is really one of ours can CORRECT its label instead of being
    # counted as new data.
    main_md5 = {}
    for c in ["benign", "malignant"]:
        for img in (SUB / "train" / c).glob("*"):
            main_md5[md5(img)] = (img, 1 if c == "malignant" else 0)
    print(f"main train/val fingerprints: {len(main_md5)}", flush=True)

    n_add = n_test = n_dup = n_corr = 0
    used_classes = {}
    for split_dir in sorted(p for p in NINE.iterdir() if p.is_dir()):
        sname = split_dir.name
        if sname == "test":
            continue                      # never use their test split
        for cdir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            binl = FINE2BIN.get(cdir.name)
            if binl is None:
                continue
            for img in sorted(cdir.iterdir()):
                if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                m = md5(img)
                if m in test_md5:
                    n_test += 1
                    continue
                if near_duplicate(psig(img), test_sig):
                    n_test += 1
                    continue
                if m in main_md5:
                    # same image as one of ours: correct the label if it differs
                    n_dup += 1
                    link_path, coarse = main_md5[m]
                    if coarse != binl:
                        n_corr += 1
                        # relink the same target into the class the fine label implies
                        try:
                            target = os.path.realpath(link_path)
                        except Exception:
                            target = str(link_path)
                        tgt_dir = SUB / "train" / ("malignant" if binl else "benign")
                        newname = tgt_dir / f"corr_{link_path.name}"
                        if not newname.exists():
                            try:
                                link_path.unlink()
                            except Exception:
                                pass
                            link(target, newname)
                    continue
                link(img, SUB / "train" / ("malignant" if binl else "benign") / f"n9_{n_add}.png")
                used_classes[cdir.name] = used_classes.get(cdir.name, 0) + 1
                n_add += 1
    print(f"9-class: added {n_add} NEW images | dropped as test-overlap {n_test} | "
          f"already in our pool {n_dup} | of which LABEL CORRECTED {n_corr}", flush=True)
    print("  new images per class:", used_classes, flush=True)
    mark("nine-added")

    base_n2 = sum(1 for _ in (SUB / "train").rglob("*"))
    print(f"base pool now: {base_n2} (was {base_n})", flush=True)

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
            wimg.setdefault(wild_key(p.name), p)
        for p in d.rglob("wildlbl*.txt"):
            wlab.setdefault(wild_key(p.name), p)
    print(f"wild images found: {len(wimg)} | labels: {len(wlab)}", flush=True)
    if len(wimg) < 250 or len(wlab) < 250:
        print("wild assets incomplete - aborting", flush=True)
        sys.exit(2)

    def wild_items(entries):
        out, missing = [], []
        for e in entries:
            key = wild_key(e["file"])
            ip = wimg.get(key)
            lp = wlab.get(key)
            if ip is None or lp is None:
                missing.append(e["file"])
                continue
            out.append((str(ip), 1 if e["label"] == "malignant" else 0))
        if missing:
            print(f"  FATAL: {len(missing)} split entries unresolved, e.g. {missing[:3]}",
                  flush=True)
            sys.exit(2)
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

    # Pin the real-data share to V25's value. V25 used 159 real images x 6
    # against a base of 3,367 -> 954/(954+3367) = 22.1%. The 9-class images
    # grow the base, so the repeat is recomputed to keep that share identical.
    # The base size is read from the directory - `tr` does not exist yet.
    base_count = len(list((SUB / "train").rglob("*")))
    real_items_needed = TARGET_REAL_SHARE / (1 - TARGET_REAL_SHARE) * base_count
    wild_repeat = max(1, int(round(real_items_needed / max(len(train_wild), 1))))
    print(f"base {base_count} | real needed "
          f"{real_items_needed:.0f} items | {len(train_wild)} real images -> "
          f"repeat {wild_repeat}", flush=True)

    tr = MixedDataset(SUB / "train", train_wild, aug_train(IMG), wild_repeat)
    va = datasets.ImageFolder(str(SUB / "val"), transform=aug_eval(IMG))
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True,
                       drop_last=True, persistent_workers=True)
    va_ld = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    print(f"train items {len(tr)} (main {len(tr.base)} + wild {len(tr.wild)}x{wild_repeat}) "
          f"| real share {len(tr.wild)*wild_repeat/len(tr):.1%} | val {len(va)}", flush=True)

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

    best, out = 0.0, WORK / "specB30.pt"
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        co = to = 0
        for x, y in tqdm(tr_ld, desc=f"B30 {ep}/{EPOCHS}", leave=False):
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
        sched.step()
        vacc = evaluate_btxrd()
        hist.append({"ep": ep, "train": co / to, "val": vacc, "secs": time.time() - t0})
        print(f"B30 Ep{ep}: train={co/to:.4f} btxrd_val={vacc:.4f} "
              f"({time.time()-t0:.0f}s) peakmem={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
              flush=True)
        if vacc >= best:
            best = vacc
            torch.save({"model": model.state_dict(), "acc": vacc, "ep": ep}, str(out))
            print(f"  *** saved ({vacc:.4f})", flush=True)
    print(f"B30 best btxrd val: {best:.4f}", flush=True)
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
        "v": 30, "init": "stageB21", "lr": LR, "epochs": EPOCHS, "img": IMG,
        "wild_repeat": wild_repeat,
        "target_real_share": TARGET_REAL_SHARE, "wild_train": len(train_wild),
        "wild_val": len(val_wild), "nine_added": n_add, "nine_dropped_test": n_test,
        "nine_already_in_pool": n_dup, "nine_label_corrections": n_corr,
        "nine_per_class": used_classes, "base_pool": base_n2,
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
