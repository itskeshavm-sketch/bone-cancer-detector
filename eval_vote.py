"""Benchmark the B11+B12 two-expert vote (single-view) vs B12 alone."""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, efficientnet_b6
from sklearn.metrics import accuracy_score
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
    return m.eval()


mA = load("checkpoints/stageA12.pt", efficientnet_b3, 2)
mB12 = load("checkpoints/stageB12.pt", efficientnet_b6, 2)
mB11 = load("checkpoints/stageB11.pt", efficientnet_b6, 2)
evA, evB = tf_(300), tf_(456)
ds = datasets.ImageFolder("data/test", transform=tf_(300))
Pv, Ps, T = [], [], []
with torch.no_grad():
    for path, t in tqdm(ds.samples, desc="vote benchmark"):
        pil = Image.open(path).convert("RGB")
        a = int(mA(evA(pil).unsqueeze(0)).argmax(1).item())
        T.append(t)
        if a == 0:
            Pv.append(2)
            Ps.append(2)
        else:
            x = evB(pil).unsqueeze(0)
            p12 = F.softmax(mB12(x), 1)[0]
            pv = (p12 + F.softmax(mB11(x), 1)[0]) / 2
            Ps.append(int(p12.argmax().item()))
            Pv.append(int(pv.argmax().item()))
res = {"vote_single": round(accuracy_score(T, Pv), 4),
       "b12_single": round(accuracy_score(T, Ps), 4), "n": len(T)}
json.dump(res, open("checkpoints/metrics_vote_b11b12.json", "w"), indent=1)
print(res)
