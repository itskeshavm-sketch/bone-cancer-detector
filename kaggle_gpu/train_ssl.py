"""Kaggle GPU V19a2 (V19a OOM/shm fix): DINO on ~14k unlabeled train X-rays, B3 backbone.
2 global (192) + 2 local (96) crops, batch 48, workers=2, 20 epochs (~5h). Saves teacher
backbone every 2 epochs for V19b fine-tune. Seed 19."""
import csv, hashlib, json, os, random, shutil, sys, time, traceback
from pathlib import Path

random.seed(19)
import torch
torch.manual_seed(19)

print("=== DIAGNOSTICS ===", flush=True)
print("cuda:", torch.cuda.is_available(), flush=True)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE", flush=True)
if not torch.cuda.is_available():
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
print("MAIN:", MAIN_SRC, "ZIYA:", ZIYA, "XMROOT:", XMROOT, flush=True)
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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.models import efficientnet_b3
from PIL import Image

WORK = Path("/kaggle/working")
TMP = Path("/tmp/kwork")
TMP.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class UnlabeledXRay(Dataset):
    def __init__(self, files, global_tf, local_tf):
        self.files = files
        self.global_tf = global_tf
        self.local_tf = local_tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return [self.global_tf(img), self.global_tf(img)] + \
               [self.local_tf(img) for _ in range(2)]


def crop_tf(size, scale):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale, ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.GELU(),
            nn.Linear(2048, out_dim))

    def forward(self, x):
        return self.mlp(x)


def backbone():
    m = efficientnet_b3(weights=None)
    m.classifier = nn.Identity()
    return m


try:
    # ---- pool every TRAIN image, labels ignored ----
    POOL = TMP / "pool"
    POOL.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in [MAIN_SRC / "train" / "images"]:
        for img in d.iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                link(img, POOL / f"m_{img.name}")
                n += 1
    for split in ["train"]:
        with open(ZIYA / split / "_classes.csv") as f:
            r = csv.DictReader(f)
            cols = r.fieldnames
            for row in r:
                src = ZIYA / split / row[cols[0]]
                if src.exists():
                    link(src, POOL / f"z_{Path(row[cols[0]]).name}")
                    n += 1
    with open(XMROOT / "train" / "_train_labels.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            src = XMROOT / "train" / row["image_id"]
            if not src.exists():
                c = list(XMROOT.rglob(row["image_id"]))
                src = c[0] if c else None
            if src is not None and src.exists():
                link(src, POOL / f"x_{Path(row['image_id']).name}")
                n += 1
    files = [p for p in POOL.iterdir() if p.is_file()]
    print(f"pool: {len(files)} images", flush=True)
    mark("pool-ready")
    assert len(files) > 5000

    EPOCHS, BATCH = 20, 48
    ds = UnlabeledXRay(files, crop_tf(192, (0.4, 1.0)), crop_tf(96, (0.05, 0.4)))
    ld = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)

    student_bb, teacher_bb = backbone().to(device), backbone().to(device)
    student_hd = DINOHead(1536).to(device)
    teacher_hd = DINOHead(1536).to(device)
    teacher_bb.load_state_dict(student_bb.state_dict())
    teacher_hd.load_state_dict(student_hd.state_dict())
    for p in list(teacher_bb.parameters()) + list(teacher_hd.parameters()):
        p.requires_grad = False

    opt = torch.optim.AdamW(list(student_bb.parameters()) + list(student_hd.parameters()),
                            lr=3e-4, weight_decay=0.04)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = GradScaler()
    center = torch.zeros(1, 256, device=device)
    mom_sched = [0.996 + 0.5 * (1.0 - 0.996) * (1 + __import__("math").cos(3.14159 * e / EPOCHS))
                 for e in range(EPOCHS)]
    hist = []
    for ep in range(EPOCHS):
        student_bb.train()
        student_hd.train()
        tot, cnt, t0 = 0.0, 0, time.time()
        for views in ld:
            views = [v.to(device, non_blocking=True) for v in views]
            opt.zero_grad()
            with autocast():
                s_out = [student_hd(student_bb(v)) / 0.1 for v in views]
                with torch.no_grad():
                    t_out = [teacher_hd(teacher_bb(v)) for v in views[:2]]
                    t_out = [(t - center).softmax(1) for t in t_out]
                loss = 0.0
                for ti, tq in enumerate(t_out):
                    for si, sq in enumerate(s_out):
                        if si == ti:
                            continue
                        loss += -(tq * F.log_softmax(sq, dim=1)).sum(1).mean()
                loss = loss / (2 * 3)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                m = mom_sched[ep]
                pairs = list(zip(student_bb.parameters(), teacher_bb.parameters())) + \
                    list(zip(student_hd.parameters(), teacher_hd.parameters()))
                for ps, pt in pairs:
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                tb = torch.cat([teacher_hd(teacher_bb(v)) for v in views[:2]], 0).mean(0, keepdim=True)
                center.mul_(0.9).add_(tb, alpha=0.1)
            tot += loss.item()
            cnt += 1
        sched.step()
        msg = f"dino ep{ep + 1}/{EPOCHS}: loss={tot / cnt:.4f} ({time.time() - t0:.0f}s)"
        print(msg, flush=True)
        hist.append({"ep": ep + 1, "loss": tot / cnt})
        if (ep + 1) % 2 == 0 or ep == EPOCHS - 1:
            torch.save({"backbone": teacher_bb.state_dict(), "ep": ep + 1},
                       str(WORK / f"dino_bb_ep{ep + 1}.pt"))
    torch.save({"backbone": teacher_bb.state_dict(), "ep": EPOCHS}, str(WORK / "dino_backbone.pt"))
    (WORK / "dino_hist.json").write_text(json.dumps(hist))
    mark("pretrain-done")
    print("DONE", flush=True)
except Exception:
    tb = traceback.format_exc()
    print("FATAL:\n" + tb, flush=True)
    try:
        Path("/kaggle/working/ERROR.txt").write_text(tb)
    except Exception:
        pass
    sys.exit(1)
