"""Kaggle GPU V19b: SSL-vs-ImageNet face-off. B3 specialist (384px, proven V12 aug) trained
twice identically — once from DINO backbone, once from ImageNet. Linear probe (8ep frozen)
then full fine-tune (40ep). Gate reuses stageA18 (99.37% val, no retrain). Chained TTA eval
on val (thresh sweep) + test once. Winner decided on test + reported. Seed 19."""
import csv, hashlib, json, os, random, shutil, sys, time, traceback
from pathlib import Path

random.seed(19)
import torch
torch.manual_seed(19)

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
BB = next((p for p in ROOT.rglob("dino_backbone.pt") if p.is_file()), None)
GA = next((p for p in ROOT.rglob("stageA18.pt") if p.is_file()), None)
print("MAIN:", MAIN_SRC, "BACKBONE:", BB, "GATE:", GA, flush=True)
if MAIN_SRC is None or BB is None or GA is None:
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
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


def make_b3_dino(ncls, drop=0.4):
    m = efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    sd = torch.load(str(BB), map_location="cpu")["backbone"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print("dino load: missing:", len(missing), "unexpected:", len(unexpected), flush=True)
    return m.to(device)


def make_b3_imagenet(ncls, drop=0.4):
    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    return m.to(device)


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
    mark("subtype-staged")

    def train_specialist(tag, maker, epochs_probe=8, epochs_full=40):
        tr = datasets.ImageFolder(str(SUB / "train"), transform=aug_train(384))
        va = datasets.ImageFolder(str(SUB / "val"), transform=aug_eval(384))
        tr_ld = DataLoader(tr, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        va_ld = DataLoader(va, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        model = maker(2)

        def evaluate():
            P, T = [], []
            model.eval()
            with torch.no_grad():
                for x, y in va_ld:
                    P.extend(model(x.to(device)).argmax(1).cpu().tolist())
                    T.extend(y.tolist())
            return accuracy_score(T, P)

        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        # probe: head only
        for p in model.features.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = GradScaler()
        for ep in range(1, epochs_probe + 1):
            model.train()
            for x, y in tqdm(tr_ld, desc=f"{tag} probe {ep}/{epochs_probe}", leave=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad()
                with autocast():
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        probe_acc = evaluate()
        print(f"{tag} probe val: {probe_acc:.4f}", flush=True)
        # full fine-tune, discriminative lrs
        for p in model.features.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 3e-4}], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_full)
        best, out = 0.0, WORK / f"{tag}.pt"
        for ep in range(1, epochs_full + 1):
            model.train()
            t0, co, to = time.time(), 0, 0
            for x, y in tqdm(tr_ld, desc=f"{tag} ft {ep}/{epochs_full}", leave=False):
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
            va_acc = evaluate()
            print(f"{tag} Ep{ep}: train={co/to:.4f} val={va_acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if va_acc > best:
                best = va_acc
                torch.save({"model": model.state_dict(), "acc": va_acc}, str(out))
        print(f"{tag} best val: {best:.4f} (probe was {probe_acc:.4f})", flush=True)
        return str(out), probe_acc, best

    ssl_pt, ssl_probe, ssl_best = train_specialist("specSSL", make_b3_dino)
    mark("ssl-done")
    img_pt, img_probe, img_best = train_specialist("specIMG", make_b3_imagenet)
    mark("img-done")

    # ---- chained eval with A18 gate, TTA ----
    mA = efficientnet_b3(weights=None)
    mA.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(mA.classifier[1].in_features, 2))
    mA.load_state_dict(torch.load(str(GA), map_location=device)["model"])
    mA = mA.to(device).eval()

    def load_spec(path):
        m = efficientnet_b3(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, 2))
        m.load_state_dict(torch.load(path, map_location=device)["model"])
        return m.to(device).eval()

    mSSL, mIMG = load_spec(ssl_pt), load_spec(img_pt)
    import torchvision.transforms.functional as TF
    names3 = ["benign", "malignant", "normal"]
    evA, evB = aug_eval(300), aug_eval(384)

    def tta_views(pil):
        return [pil, TF.hflip(pil), TF.rotate(pil, 10), TF.rotate(pil, -10),
                TF.rotate(TF.hflip(pil), 10)]

    def chained_eval(mB, split, thresh):
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

    results = {"ssl_probe": ssl_probe, "ssl_val": ssl_best,
               "img_probe": img_probe, "img_val": img_best}
    for tag, mB in [("ssl", mSSL), ("img", mIMG)]:
        best_thr, best_v = 0.5, 0.0
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            vacc, _, _ = chained_eval(mB, "val", thr)
            print(f"{tag} val thresh {thr}: {vacc:.4f}", flush=True)
            if vacc > best_v:
                best_v, best_thr = vacc, thr
        acc, P, T = chained_eval(mB, "test", best_thr)
        results[tag] = {"thresh": best_thr, "val": best_v, "test": acc,
                        "report": classification_report(T, P, target_names=names3,
                                                        zero_division=0, output_dict=True),
                        "confusion": confusion_matrix(T, P).tolist()}
        print(f"{tag} CHAINED TEST: {acc:.4f} (thr {best_thr})", flush=True)
    (WORK / "metrics.json").write_text(json.dumps(results, indent=1))
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
