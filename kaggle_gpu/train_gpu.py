"""Kaggle GPU V18 (V17 xmerg-path fix): identical recipe; XMROOT (merged_dataset root) correctly
addresses train/ and val/ CSVs — V17 wrongly joined the train dir with the val filename. Seed 18."""
import csv, hashlib, json, os, random, shutil, sys, time, traceback
from pathlib import Path

random.seed(18)
import torch
torch.manual_seed(18)

print("=== DIAGNOSTICS ===", flush=True)
try:
    import torch
    print("cuda:", torch.cuda.is_available(), flush=True)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE", flush=True)
    assert torch.cuda.is_available()
except Exception as e:
    print("GPU fail:", e, flush=True)
    sys.exit(2)

ROOT = Path("/kaggle/input")


def find_main():
    for p in ROOT.rglob("train/images"):
        if p.is_dir() and (p.parent / "labels").is_dir():
            return p.parent.parent
    return None


MAIN_SRC = find_main()
_z = [p.parent.parent for p in ROOT.rglob("train/_classes.csv")]
ZIYA = _z[0] if _z else None
_x = [p.parent.parent for p in ROOT.rglob("train/_train_labels.csv") if p.is_file()]
XMROOT = _x[0] if _x else None
XMERG = next((p.parent for p in ROOT.rglob("_train_labels.csv") if p.is_file()), None)
print("MAIN:", MAIN_SRC, "ZIYA:", ZIYA, "XMROOT:", XMROOT, "XMERG:", XMERG, flush=True)
if MAIN_SRC is None or ZIYA is None or XMROOT is None:
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
from torchvision.models import efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_l, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights, EfficientNet_V2_L_Weights
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


def aug_train(img_size, strong=False):
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ]
    if strong:
        base.append(transforms.RandAugment(num_ops=2, magnitude=12))
        base.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)))
    base += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if strong:
        base.append(transforms.RandomErasing(p=0.35))
    return transforms.Compose(base)


def aug_eval(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_b3(ncls, drop=0.4):
    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    return m.to(device)


def make_b4(ncls, drop=0.4):
    m = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    return m.to(device)


def make_b5(ncls, drop=0.4):
    m = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    return m.to(device)


def make_b6(ncls, drop=0.4):
    m = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    return m.to(device)


def make_b7(ncls, drop=0.4):
    m = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    return m.to(device)


def make_ev2l(ncls, drop=0.4):
    m = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    return m.to(device)


try:
    # ---- data: main 3-class ----
    D3 = TMP / "data3"
    for split in ["train", "val", "test"]:
        for c in ["normal", "benign", "malignant"]:
            (D3 / split / c).mkdir(parents=True, exist_ok=True)
        for img in (MAIN_SRC / split / "images").iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                link(img, D3 / split / main_label(img, MAIN_SRC / split / "labels") / img.name)
    mark("main-staged")

    # ---- ziya: train+valid+test CSVs ----
    zrows = {}
    for split in ["train", "valid", "test"]:
        with open(ZIYA / split / "_classes.csv") as f:
            r = csv.DictReader(f)
            cols = r.fieldnames
            zrows[split] = [(row[cols[0]], row[cols[1]].strip() == "1") for row in r]
    print("ziya rows:", {k: len(v) for k, v in zrows.items()}, flush=True)

    ZB = TMP / "ziyabin"
    for split, dest in [("train", "train"), ("valid", "val"), ("test", "train")]:
        for c in ["tumor", "normal"]:
            (ZB / dest / c).mkdir(parents=True, exist_ok=True)
        n = 0
        for fn, is_tumor in zrows[split]:
            src = ZIYA / split / fn
            if src.exists():
                link(src, ZB / dest / ("tumor" if is_tumor else "normal") / Path(fn).name)
                n += 1
        print(f"ziya {split}->{dest}: {n}", flush=True)
    mark("ziya-staged")

    # ---- dedup all train/val vs OUR test ----
    test_hashes = {md5(img) for c in ["normal", "benign", "malignant"] for img in (D3 / "test" / c).glob("*")}
    dropped = 0
    for base in [D3 / "train", D3 / "val", ZB / "train", ZB / "val"]:
        for img in list(base.rglob("*")):
            if img.is_file() and md5(img) in test_hashes:
                img.unlink()
                dropped += 1
    print("dedup dropped:", dropped, flush=True)
    mark("dedup-done")

    # ---- xmerg rows: image_id,binary_label (1=tumor, 0=normal — verified by hash match) ----
    # NOTE: xmerg TEST split is never touched (likely overlaps our test set).
    xrows = {}
    for split, dest in [("train", "train"), ("val", "val")]:
        with open(XMROOT / split / f"_{split}_labels.csv") as f:
            r = csv.DictReader(f)
            xrows[dest] = [(row["image_id"], row["binary_label"].strip() == "1") for row in r]
    print("xmerg rows:", {k: len(v) for k, v in xrows.items()}, flush=True)

    # ---- merged binary (main + ziya + xmerg) ----
    BIN = TMP / "data_bin"
    for split in ["train", "val"]:
        for c in ["tumor", "normal"]:
            (BIN / split / c).mkdir(parents=True, exist_ok=True)
        m = "train" if split == "train" else "val"
        for img in (D3 / m / "benign").glob("*"):
            link(img, BIN / split / "tumor" / img.name)
        for img in (D3 / m / "malignant").glob("*"):
            link(img, BIN / split / "tumor" / ("m_" + img.name))
        for img in (D3 / m / "normal").glob("*"):
            link(img, BIN / split / "normal" / img.name)
        z = "train" if split == "train" else "val"
        for img in (ZB / z / "tumor").glob("*"):
            link(img, BIN / split / "tumor" / ("z_" + img.name))
        for img in (ZB / z / "normal").glob("*"):
            link(img, BIN / split / "normal" / ("z_" + img.name))
        xn = 0
        for fn, is_tumor in xrows[split]:
            src = XMROOT / split / fn
            if not src.exists():
                cands = list(XMROOT.rglob(Path(fn).name))
                src = cands[0] if cands else None
            if src is not None and src.exists():
                link(src, BIN / split / ("tumor" if is_tumor else "normal") / ("x2_" + Path(fn).name))
                xn += 1
        print(f"xmerg {split}: {xn}/{len(xrows[split])} linked", flush=True)
    print("binary:", {s: {d.name: len(list(d.glob("*"))) for d in (BIN / s).iterdir()} for s in ["train", "val"]}, flush=True)
    mark("binary-merged")

    # ---- subtype folders ----
    SUB = TMP / "data_sub"
    for split in ["train", "val"]:
        m = "train" if split == "train" else "val"
        for c in ["benign", "malignant"]:
            (SUB / split / c).mkdir(parents=True, exist_ok=True)
            for img in (D3 / m / c).glob("*"):
                link(img, SUB / split / c / img.name)
    mark("subtype-staged")

    def run_stage(tag, train_dir, val_dir, names, epochs, batch, img, maker, strong_aug, lr=3e-4, label_smoothing=0.05, swa_last=10, focal_alpha=0.25, focal_gamma=2.0):
        tr = datasets.ImageFolder(str(train_dir), transform=aug_train(img, strong_aug))
        va = datasets.ImageFolder(str(val_dir), transform=aug_eval(img))
        tr_ld = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
        va_ld = DataLoader(va, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
        counts = torch.zeros(len(names))
        for _, y in DataLoader(tr, batch_size=512, num_workers=2):
            for c in y.tolist():
                counts[c] += 1
        w = (counts.sum() / (len(names) * counts)).to(device)
        print(tag, "counts:", counts.tolist(), "img:", img, "batch:", batch, flush=True)
        model = maker(len(names))
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)

        # Focal loss wrapper for hard-example mining
        if focal_gamma > 0:
            class FocalLoss(nn.Module):
                def __init__(self, alpha, gamma, num_classes):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.ce = nn.CrossEntropyLoss()
                    self.num_classes = num_classes

                def forward(self, logits, targets):
                    ce = self.ce(logits, targets)
                    probs = F.softmax(logits, dim=1)
                    target_probs = torch.zeros_like(probs)
                    target_probs.scatter_(1, targets.unsqueeze(1), 1)
                    loss = -(self.alpha * target_probs * probs.pow(self.gamma)).sum(1).log().mean()
                    return loss * self.num_classes

            crit = FocalLoss(focal_alpha, focal_gamma, len(names))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        scaler = GradScaler()
        best, hist = 0.0, []
        out = WORK / f"{tag}.pt"
        out_swa = WORK / f"{tag}_swa.pt"
        swa_state, swa_n = None, 0
        swa_start = epochs - swa_last + 1

        def evaluate():
            P, T = [], []
            with torch.no_grad():
                for x, y in va_ld:
                    P.extend(model(x.to(device)).argmax(1).cpu().tolist())
                    T.extend(y.tolist())
            return accuracy_score(T, P), f1_score(T, P, average="macro", zero_division=0)

        for ep in range(1, epochs + 1):
            model.train()
            t0, co, to = time.time(), 0, 0
            for x, y in tqdm(tr_ld, desc=f"{tag} {ep}/{epochs}", leave=False):
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
            model.eval()
            va_acc, vf1 = evaluate()
            msg = f"{tag} Ep{ep}: train={co/to:.4f} val={va_acc:.4f} F1={vf1:.4f} ({time.time()-t0:.0f}s)"
            hist.append({"ep": ep, "train": co / to, "val": va_acc})
            if va_acc > best:
                best = va_acc
                torch.save({"model": model.state_dict(), "classes": names, "acc": va_acc}, str(out))
                print(f"  *** saved {out} ({va_acc:.4f})", flush=True)
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
            print(msg, flush=True)
        swa_acc = 0.0
        if swa_state is not None:
            with torch.no_grad():
                msd = model.state_dict()
                backup = {k: v.clone() for k, v in msd.items()}
                merged = dict(msd)
                for k, v in swa_state.items():
                    merged[k] = v.to(msd[k].dtype)
                model.load_state_dict(merged)
                swa_acc, swa_f1 = evaluate()
                model.load_state_dict(backup)
            torch.save({"model": merged, "classes": names, "acc": swa_acc}, str(out_swa))
            print(f"{tag} SWA(last{swa_last}): val={swa_acc:.4f} F1={swa_f1:.4f}", flush=True)
        return out, hist, best, out_swa, swa_acc

    a_pt, a_hist, a_best, _, _ = run_stage("stageA", BIN / "train", BIN / "val", ["normal", "tumor"],
                                           12, 64, 300, make_b3, False, swa_last=0)
    b_pt, b_hist, b_best, b_swa_pt, b_swa = run_stage("stageB", SUB / "train", SUB / "val", ["benign", "malignant"],
                                                     60, 12, 456, make_b6, True, label_smoothing=0.05, swa_last=10, focal_gamma=0)
    b_final = str(b_swa_pt) if b_swa > b_best else str(b_pt)
    print(f"stageB raw={b_best:.4f} swa={b_swa:.4f} chained uses: {b_final}", flush=True)

    # ---- chained eval WITH test-time augmentation ----
    mA = make_b3(2)
    mA.load_state_dict(torch.load(str(a_pt), map_location=device)["model"])
    mA.eval()
    mB = make_b6(2)
    mB.load_state_dict(torch.load(b_final, map_location=device)["model"])
    mB.eval()
    import torchvision.transforms.functional as TF
    names3 = ["benign", "malignant", "normal"]
    evA, evB = aug_eval(300), aug_eval(456)

    def tta_views(pil):
        return [pil,
                TF.hflip(pil),
                TF.rotate(pil, 10),
                TF.rotate(pil, -10),
                TF.rotate(TF.hflip(pil), 10)]

    def chained_eval(split, thresh):
        P, T = [], []
        with torch.no_grad():
            for ci, cn in enumerate(names3):
                for img_p in (D3 / split / cn).glob("*"):
                    pil = Image.open(img_p).convert("RGB")
                    # stage A: average tumor-prob over views
                    tp = 0.0
                    for v in tta_views(pil):
                        tp += torch.softmax(mA(evA(v).unsqueeze(0).to(device)), 1)[0, 1].item()
                    tp /= 5
                    if tp < thresh:
                        P.append(2)
                    else:
                        bp = torch.zeros(2)
                        for v in tta_views(pil):
                            bp += torch.softmax(mB(evB(v).unsqueeze(0).to(device)), 1)[0].cpu()
                        P.append(int((bp / 5).argmax()))
                    T.append(ci)
        return accuracy_score(T, P), P, T

    best_thr, best_vacc = 0.5, 0.0
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        vacc, _, _ = chained_eval("val", thr)
        print(f"val thresh {thr}: {vacc:.4f}", flush=True)
        if vacc > best_vacc:
            best_vacc, best_thr = vacc, thr
    print(f"chosen thresh: {best_thr} (val {best_vacc:.4f})", flush=True)
    acc, P, T = chained_eval("test", best_thr)
    metrics = {"stageA_val": a_best, "stageB_val": b_best, "stageB_swa_val": b_swa,
               "stageB_choice": b_final, "gate_thresh": best_thr, "chained_val_acc_tta": best_vacc,
               "chained_test_acc_tta": acc,
               "report": classification_report(T, P, target_names=names3, zero_division=0, output_dict=True),
               "confusion": confusion_matrix(T, P).tolist(),
               "histA": a_hist, "histB": b_hist}
    print("CHAINED TEST TTA:", round(acc, 4), flush=True)
    print(classification_report(T, P, target_names=names3, zero_division=0), flush=True)
    (WORK / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (WORK / "classes.json").write_text(json.dumps(names3))
    print("DONE", flush=True)
except Exception:
    tb = traceback.format_exc()
    print("FATAL:\n" + tb, flush=True)
    try:
        Path("/kaggle/working/ERROR.txt").write_text(tb)
    except Exception:
        pass
    sys.exit(1)
