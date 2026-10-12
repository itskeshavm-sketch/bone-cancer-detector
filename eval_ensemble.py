"""Local single-view chained eval: A10 gate + Stage-B ensemble (B6/B7/B10 averaged probs)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, efficientnet_b4, efficientnet_b5
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image
from tqdm import tqdm


def tf_(s):
    return transforms.Compose([
        transforms.Resize((s, s)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load(path, maker, ncls):
    ckpt = torch.load(path, map_location="cpu")
    m = maker(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, ncls))
    m.load_state_dict(ckpt["model"])
    return m.eval(), ckpt["classes"]


mA, ca = load("checkpoints/stageA10.pt", efficientnet_b3, 2)
mB6, c6 = load("checkpoints/stageB6.pt", efficientnet_b4, 2)
mB7, c7 = load("checkpoints/stageB7.pt", efficientnet_b5, 2)
mB10, c10 = load("checkpoints/stageB10.pt", efficientnet_b5, 2)
print("A:", ca, "| B6:", c6, "| B7:", c7, "| B10:", c10, flush=True)

evA, ev380, ev456 = tf_(300), tf_(380), tf_(456)
names3 = ["benign", "malignant", "normal"]
ds = datasets.ImageFolder("data/test", transform=tf_(300))

P, T = [], []
with torch.no_grad():
    for path, t in tqdm(ds.samples, desc="ensemble eval"):
        pil = Image.open(path).convert("RGB")
        a = int(mA(evA(pil).unsqueeze(0)).argmax(1).item())
        T.append(t)
        if a == 0:
            P.append(2)
        else:
            p = (F.softmax(mB6(ev380(pil).unsqueeze(0)), 1)[0]
                 + F.softmax(mB7(ev456(pil).unsqueeze(0)), 1)[0]
                 + F.softmax(mB10(ev456(pil).unsqueeze(0)), 1)[0]) / 3
            P.append(int(p.argmax().item()))

print("ensemble single-view test_acc:", round(accuracy_score(T, P), 4), flush=True)
print(classification_report(T, P, target_names=names3, zero_division=0), flush=True)
print("confusion:", confusion_matrix(T, P).tolist(), flush=True)
