"""List V12's test mistakes (single-view) for error analysis."""
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, efficientnet_b6
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
mB = load("checkpoints/stageB12.pt", efficientnet_b6, 2)
evA, evB = tf_(300), tf_(456)
names3 = ["benign", "malignant", "normal"]
ds = datasets.ImageFolder("data/test", transform=tf_(300))
errs = []
with torch.no_grad():
    for path, t in tqdm(ds.samples, desc="v12 errors"):
        pil = Image.open(path).convert("RGB")
        a = int(mA(evA(pil).unsqueeze(0)).argmax(1).item())
        p = 2 if a == 0 else int(mB(evB(pil).unsqueeze(0)).argmax(1).item())
        if p != t:
            errs.append({"file": path, "true": names3[t], "pred": names3[p]})
json.dump(errs, open("checkpoints/errors_v12.json", "w"), indent=1)
print("errors:", len(errs), "of", len(ds.samples))
