import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, efficientnet_b4
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
    ckpt = torch.load(path, map_location='cpu')
    m = maker(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, ncls))
    m.load_state_dict(ckpt['model'])
    return m.eval(), ckpt['classes']


mA, ca = load('kaggle_out/stageA6.pt', efficientnet_b3, 2)  # [normal, tumor]
mB, cb = load('kaggle_out/stageB6.pt', efficientnet_b4, 2)  # [benign, malignant]
print('A:', ca, 'B:', cb, flush=True)

ds = datasets.ImageFolder('data/test', transform=tf_(300))  # only for file list + labels
evA, evB = tf_(300), tf_(380)
names3 = ['benign', 'malignant', 'normal']
P, T = [], []
with torch.no_grad():
    for path, t in tqdm(ds.samples, desc="chained eval"):
        pil = Image.open(path).convert("RGB")
        a = int(mA(evA(pil).unsqueeze(0)).argmax(1).item())
        T.append(t)
        P.append(2 if a == 0 else int(mB(evB(pil).unsqueeze(0)).argmax(1).item()))

print('chained (single-view) test_acc:', round(accuracy_score(T, P), 4), flush=True)
print(classification_report(T, P, target_names=names3, zero_division=0), flush=True)
print('confusion:', confusion_matrix(T, P).tolist(), flush=True)
