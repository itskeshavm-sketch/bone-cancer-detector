"""Accuracy-focused resume: full 224px, low LR, label smoothing, class weights.
Resumes checkpoints/best_model.pt (71.9%), overwrites it only on improvement.
"""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


def main():
    a = get_args()
    torch.set_num_threads(8)
    device = torch.device("cpu")
    print(f"V2: 224px batch={a.batch_size} epochs={a.epochs} lr={a.lr}", flush=True)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dd = Path(a.data_dir)
    train_ds = datasets.ImageFolder(str(dd / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(dd / "val"), transform=val_tf)
    names = train_ds.classes
    print(f"Classes: {names} train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_ld = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

    # Class weights from train distribution (helps benign/malignant balance)
    counts = torch.zeros(len(names))
    for _, y in DataLoader(train_ds, batch_size=512, num_workers=0):
        for c in y.tolist():
            counts[c] += 1
    weights = counts.sum() / (len(names) * counts)
    print(f"Train counts: {counts.tolist()} weights: {[round(float(w), 3) for w in weights]}", flush=True)

    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(names)))
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    best = float(ckpt.get("acc", 0))
    print(f"Resumed (val={best:.4f})", flush=True)
    model = model.to(device)

    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    history = []

    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, corr, tot = 0.0, 0, 0
        for x, y in tqdm(train_ld, desc=f"Ep {ep}/{a.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            corr += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
        sched.step()
        dt = time.time() - t0

        model.eval()
        P, T = [], []
        with torch.no_grad():
            for x, y in val_ld:
                P.extend(model(x.to(device)).argmax(1).tolist())
                T.extend(y.tolist())
        vacc = accuracy_score(T, P)
        vf1 = f1_score(T, P, average="macro", zero_division=0)
        print(f"Ep{ep}: train acc={corr/tot:.4f} loss={loss_sum/tot:.4f} "
              f"val acc={vacc:.4f} F1={vf1:.4f} ({dt:.0f}s)", flush=True)
        history.append({"ep": ep, "train_acc": corr / tot, "val_acc": vacc, "val_f1": vf1})
        if vacc > best:
            best = vacc
            torch.save({"model": model.state_dict(), "classes": names, "acc": vacc},
                       a.checkpoint)
            print(f"  *** saved {a.checkpoint} ({vacc:.4f})", flush=True)

    Path("checkpoints/v2_history.json").write_text(json.dumps(history, indent=1))
    print(f"Done. best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
