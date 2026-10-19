"""Kaggle pilot V21-p: X-ray-pretrained (TorchXRayVision DenseNet121, ~1M chest X-rays)
vs ImageNet-pretrained DenseNet121 as init for the benign/malignant specialist.
Same data/aug/seed/epochs per arm; val decides. CPU-verified wrapper (xrv's own
18-output head breaks on a replaced classifier, so we wrap .features directly).
Decision rule: gap >1pp funds full 456px V21; else domain-init dies. ~40 min."""
import os, random, subprocess, sys, time, traceback, json
from pathlib import Path

random.seed(21)
import torch
torch.manual_seed(21)

print("=== DIAGNOSTICS ===", flush=True)
print("cuda:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    sys.exit(2)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchxrayvision"], check=False)
import torchxrayvision as xrv
print("xrv:", getattr(xrv, "__version__", "?"), flush=True)

ROOT = Path("/kaggle/input")


def find_main():
    for p in ROOT.rglob("train/images"):
        if p.is_dir() and (p.parent / "labels").is_dir():
            return p.parent.parent
    return None


MAIN_SRC = find_main()
print("MAIN:", MAIN_SRC, flush=True)
if MAIN_SRC is None:
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
        import shutil
        shutil.copy2(src, dst)


import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
from tqdm import tqdm

WORK = Path("/kaggle/working")
TMP = Path("/tmp/kpilot")
TMP.mkdir(parents=True, exist_ok=True)
MCLS = {0: "benign", 1: "malignant"}
device = torch.device("cuda")
IMG, BATCH, EPOCHS = 224, 32, 15


def label_of(img: Path, lbl_dir: Path) -> str:
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


def md5(p: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def xrv_range(x):
    """xrv.utils.normalize is (2*(img/255) - 1) * 1024 -> expects [-1024, 1024].
    ToTensor already yields img/255 in [0,1], so this reproduces xrv's scale exactly."""
    return (2.0 * x - 1.0) * 1024.0


def aug_tr(gray):
    ts = [transforms.Resize((IMG, IMG))]
    if gray:
        ts.append(transforms.Grayscale(num_output_channels=1))
    ts += [transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
           transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
           transforms.ColorJitter(brightness=0.15, contrast=0.15),
           transforms.RandAugment(num_ops=2, magnitude=9), transforms.ToTensor()]
    ts.append(transforms.Lambda(xrv_range) if gray
              else transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    ts.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(ts)


def aug_ev(gray):
    ts = [transforms.Resize((IMG, IMG))]
    if gray:
        ts.append(transforms.Grayscale(num_output_channels=1))
    ts += [transforms.ToTensor(),
           transforms.Lambda(xrv_range) if gray
           else transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ts)


class XRVHead(nn.Module):
    """Wrap xrv features + fresh 2-class head (its own forward() assumes 18 outputs)."""
    def __init__(self, base, ncls=2, drop=0.4):
        super().__init__()
        self.feats = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(1024, ncls))

    def forward(self, x):
        return self.fc(self.pool(self.feats(x)).flatten(1))


def make_xrv():
    return XRVHead(xrv.models.DenseNet(weights="densenet121-res224-all")).to(device)


def make_img():
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier.in_features, 2))
    return m.to(device)


def run_arm(tag, maker, gray, amp):
    torch.manual_seed(21)
    random.seed(21)
    tr = datasets.ImageFolder(str(SUB / "train"), transform=aug_tr(gray))
    va = datasets.ImageFolder(str(SUB / "val"), transform=aug_ev(gray))
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    va_ld = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    model = maker()
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler(enabled=amp)
    print(f"{tag}: amp={amp} input_scale={'xrv[-1024,1024]' if gray else 'imagenet[-2.2,2.2]'}", flush=True)
    best = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for x, y in tqdm(tr_ld, desc=f"{tag} {ep}/{EPOCHS}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            # xrv-normalized inputs (|x|~1024) overflow fp16 in conv1 (peak 3.2e6 >> 65504),
            # so that arm MUST run fp32; imagenet-scale inputs are safe under AMP.
            with autocast(enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        P, T = [], []
        model.eval()
        with torch.no_grad():
            for x, y in va_ld:
                P.extend(model(x.to(device)).argmax(1).cpu().tolist())
                T.extend(y.tolist())
        acc = accuracy_score(T, P)
        print(f"{tag} ep{ep}: val={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "acc": acc}, str(WORK / f"{tag}.pt"))
    print(f"{tag} BEST VAL: {best:.4f}", flush=True)
    return best


try:
    D3 = TMP / "d3"
    for split in ["train", "val", "test"]:
        for c in ["normal", "benign", "malignant"]:
            (D3 / split / c).mkdir(parents=True, exist_ok=True)
        for img in (MAIN_SRC / split / "images").iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                link(img, D3 / split / label_of(img, MAIN_SRC / split / "labels") / img.name)
    mark("main-staged")

    th = {md5(i) for c in ["normal", "benign", "malignant"] for i in (D3 / "test" / c).glob("*")}
    drop = 0
    for base in [D3 / "train", D3 / "val"]:
        for img in list(base.rglob("*")):
            if img.is_file() and md5(img) in th:
                img.unlink()
                drop += 1
    print("dedup dropped:", drop, flush=True)

    SUB = TMP / "sub"
    for split in ["train", "val"]:
        for c in ["benign", "malignant"]:
            (SUB / split / c).mkdir(parents=True, exist_ok=True)
            for img in (D3 / split / c).glob("*"):
                link(img, SUB / split / c / img.name)
    ntr = sum(1 for _ in (SUB / "train").rglob("*") if _.is_file())
    nva = sum(1 for _ in (SUB / "val").rglob("*") if _.is_file())
    print(f"train={ntr} val={nva}", flush=True)
    mark("subtype-staged")

    xrv_best = run_arm("pilotXRV", make_xrv, gray=True, amp=False)
    mark("xrv-done")
    img_best = run_arm("pilotIMG", make_img, gray=False, amp=True)
    mark("img-done")

    gap = xrv_best - img_best
    verdict = "domain-init WINS -> fund full 456px V21 on Kaggle" if gap > 0.01 \
        else "domain-init does NOT transfer -> kill idea, spend quota on demo"
    print(f"X-ray-init val: {xrv_best:.4f} | ImageNet-init val: {img_best:.4f} | gap: {gap:+.4f}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    (WORK / "metrics.json").write_text(json.dumps({
        "pilot": "v21p", "xrv_best": xrv_best, "img_best": img_best, "gap": gap,
        "verdict": verdict, "epochs": EPOCHS, "img_size": IMG,
        "confound": "xrv arm 1-ch grayscale (native), imagenet arm 3-ch RGB (native)"}, indent=1))
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