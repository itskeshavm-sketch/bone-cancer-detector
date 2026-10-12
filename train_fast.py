"""Fast resume training — loads best_model.pt, trains faster on CPU.
Speed tricks: 160px images, batch 128, workers 0, light aug, all 8 CPU threads.
"""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def args_fn():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--out", default="checkpoints/best_model.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

def main():
    a = args_fn()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    device = torch.device("cpu")
    print(f"Fast train: img={a.img_size} batch={a.batch_size} epochs={a.epochs} threads=8")

    S = a.img_size
    train_tf = transforms.Compose([
        transforms.Resize((S, S)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((S, S)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    dd = Path(a.data_dir)
    train_ds = datasets.ImageFolder(str(dd/"train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(dd/"val"), transform=val_tf)
    names = train_ds.classes
    print(f"Classes: {names} train={len(train_ds)} val={len(val_ds)}")

    train_ld = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_ld = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(names)))
    start_acc = 0.0
    ckpt_path = Path(a.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_acc = float(ckpt.get("acc", 0))
        print(f"Resumed from {ckpt_path} (val acc={start_acc:.4f})")
    model = model.to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best = start_acc
    for ep in range(1, a.epochs+1):
        model.train()
        t0 = time.time()
        loss_sum, corr, tot = 0, 0, 0
        for x, y in tqdm(train_ld, desc=f"Ep {ep}/{a.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*x.size(0)
            corr += (logits.argmax(1)==y).sum().item()
            tot += x.size(0)
        sched.step()
        dt = time.time()-t0
        print(f"  train acc={corr/tot:.4f} loss={loss_sum/tot:.4f} ({tot/dt:.0f} img/s, {dt:.0f}s)")

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_ld:
                preds.extend(model(x.to(device)).argmax(1).tolist())
                trues.extend(y.tolist())
        vacc = accuracy_score(trues, preds)
        print(f"  -> val acc={vacc:.4f} (best={best:.4f})")
        if vacc > best:
            best = vacc
            torch.save({"model": model.state_dict(), "classes": names, "acc": vacc}, str(a.out))
            Path("checkpoints/classes.json").write_text(json.dumps(names))
            print(f"  *** saved {a.out}")

    print(f"Done. best={best:.4f} started={start_acc:.4f}")

if __name__ == "__main__":
    main()
