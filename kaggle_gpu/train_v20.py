"""Kaggle GPU V20: specialist-only B5 run. Reuses A18 gate (99.37% val, no retrain).
B5-456px benign/malignant specialist, seed 20, proven V12 aug (RandAugment mag9 +
Erasing 0.25, plain CE + label-smooth 0.05), 50 epochs, batch12 AMP. Chained TTA
eval with A18 gate on val (thresh sweep) + test once. Fits ~3.7h in 5h39 quota."""
import hashlib, json, os, random, shutil, sys, time, traceback
from pathlib import Path

random.seed(20)
import torch
torch.manual_seed(20)

print("=== DIAGNOSTICS ===", flush=True)
print("cuda:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    sys.exit(2)

ROOT = Path("/kaggle/input")


def find_main():
    for p in ROOT.rglob("train/images"):
        if p.is_dir() and (p.parent / "labels").is_dir():
            return p.parent.parent
    return None


MAIN_SRC = find_main()
GA = next((p for p in ROOT.rglob("stageA18.pt") if p.is_file()), None)
print("MAIN:", MAIN_SRC, "GATE:", GA, flush=True)
if MAIN_SRC is None or GA is None:
    sys.exit(2)


def mark(s):
    print("STAGE:", s, flush=True)
    try:
        Path("/kaggle/working/STAGE.txt").write_text(s)
    except Exception:
        pass


def link(src, dst):
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, efficientnet_b5
from torchvision.models import EfficientNet_B3_Weights, EfficientNet_B5_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from PIL import Image

WORK = Path("/kaggle/working")
TMP = Path("/tmp/kwork")
TMP.mkdir(parents=True, exist_ok=True)
MCLS = {0: "benign", 1: "malignant"}
device = torch.device("cuda")


def main_label(img: Path, lbl_dir: Path) -> str:
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


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


try:
    D3 = TMP / "data3"
    for split in ["train", "val", "test"]:
        for c in ["normal", "benign", "malignant"]:
            (D3 / split / c).mkdir(parents=True, exist_ok=True)
        for img in (MAIN_SRC / split / "images").iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                link(img, D3 / split / main_label(img, MAIN_SRC / split / "labels") / img.name)
    mark("main-staged")

    test_hashes = {md5(img) for c in ["normal", "benign", "malignant"] for img in (D3 / "test" / c).glob("*")}
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
    mark("subtype-staged")

    EPOCHS, BATCH, IMG = 50, 12, 456
    tr = datasets.ImageFolder(str(SUB / "train"), transform=aug_train(IMG))
    va = datasets.ImageFolder(str(SUB / "val"), transform=aug_eval(IMG))
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    va_ld = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 2))
    model = model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler()

    def evaluate():
        P, T = [], []
        model.eval()
        with torch.no_grad():
            for x, y in va_ld:
                P.extend(model(x.to(device)).argmax(1).cpu().tolist())
                T.extend(y.tolist())
        return accuracy_score(T, P)

    best, out = 0.0, WORK / "specB20.pt"
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, co, to = time.time(), 0, 0
        for x, y in tqdm(tr_ld, desc=f"B20 {ep}/{EPOCHS}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
        vacc = evaluate()
        print(f"B20 Ep{ep}: train={co/to:.4f} val={vacc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vacc > best:
            best = vacc
            torch.save({"model": model.state_dict(), "acc": vacc}, str(out))
            print(f"  *** saved ({vacc:.4f})", flush=True)
    print(f"B20 best val: {best:.4f}", flush=True)
    mark("spec-done")

    mA = efficientnet_b3(weights=None)
    mA.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(mA.classifier[1].in_features, 2))
    mA.load_state_dict(torch.load(str(GA), map_location=device)["model"])
    mA = mA.to(device).eval()
    mB = efficientnet_b5(weights=None)
    mB.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(mB.classifier[1].in_features, 2))
    mB.load_state_dict(torch.load(str(out), map_location=device)["model"])
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
                for img_p in (D3 / split / cn).glob("*"):
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
        "v": 20, "spec_val": best, "gate_thresh": best_thr,
        "val_tta": best_v, "test_tta": acc,
        "report": classification_report(T, P, target_names=names3,
                                        zero_division=0, output_dict=True),
        "confusion": confusion_matrix(T, P).tolist()}, indent=1))
    mark("done")
    print("DONE", flush=True)
except Exception:
    tb = traceback.format_exc()
    print("FATAL:\n" + tb, flush=True)
    try:
        Path("/kaggle/working/ERROR.txt").write_text(tb)
    except Exception:
        pass
    sys.exit(1)
