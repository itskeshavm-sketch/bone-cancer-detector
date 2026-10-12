"""
Hackathon Bone Cancer Classifier — EfficientNet-B0 transfer learning
- 3-class: normal / benign / malignant  OR  Binary: normal / tumor
- Uses ./data (3-class) by default. Use --binary to train on ./data_binary
- Grad-CAM saved for demo
"""
import argparse, os, time, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Path to data/ (imagefolder)")
    p.add_argument("--binary", action="store_true", help="Binary mode info (just changes label names)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default="checkpoints")
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms — strong augmentation for X-ray
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.9,1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    data_dir = Path(args.data_dir)
    train_ds = datasets.ImageFolder(str(data_dir/"train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(data_dir/"val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(str(data_dir/"test"),  transform=val_tf)

    class_names = train_ds.classes
    print(f"Classes: {class_names} -> {train_ds.class_to_idx}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model — EfficientNet-B0 pretrained
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    # Add dropout
    model.classifier = nn.Sequential(nn.Dropout(0.3), model.classifier[1])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Save class names
    (out / "classes.json").write_text(json.dumps(class_names))

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)

        scheduler.step()

        # Validation
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss += criterion(logits, y).item() * x.size(0)
                pred = logits.argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)
                all_pred.extend(pred.cpu().tolist())
                all_true.extend(y.cpu().tolist())

        v_acc = v_correct / v_total
        v_loss /= v_total
        print(f"  -> Val loss {v_loss:.4f}  acc {v_acc:.4f}")
        print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save({"model": model.state_dict(), "classes": class_names, "acc": v_acc}, str(out / "best_model.pt"))
            print(f"  *** Saved best_model.pt (acc={v_acc:.4f})")

    # Final test eval
    print("\n=== TEST ===")
    ckpt = torch.load(str(out / "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(1).cpu().tolist()
            all_pred.extend(pred); all_true.extend(y.tolist())
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(all_true, all_pred))
    print(f"\nBest val acc: {best_acc:.4f}")
    print(f"Checkpoint: {out/'best_model.pt'}")

if __name__ == "__main__":
    main()
