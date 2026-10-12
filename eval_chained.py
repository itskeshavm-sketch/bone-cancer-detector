import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

S = 300
tf = transforms.Compose([
    transforms.Resize((S, S)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load(path, ncls):
    ckpt = torch.load(path, map_location='cpu')
    m = efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, ncls))
    m.load_state_dict(ckpt['model'])
    return m.eval(), ckpt['classes']

mA, ca = load('kaggle_out/stageA.pt', 2)   # [normal, tumor]
mB, cb = load('kaggle_out/stageB.pt', 2)   # [benign, malignant]
print('A:', ca, 'B:', cb)

ds = datasets.ImageFolder('data/test', transform=tf)
ld = DataLoader(ds, batch_size=16, num_workers=0)
 # map dataset class index -> chained label: benign->0, malignant->1, normal->2
idx_of = ds.class_to_idx  # {'benign':0,'malignant':1,'normal':2}
names3 = ['benign', 'malignant', 'normal']
P, T = [], []
with torch.no_grad():
    for x, y in ld:
        pa = mA(x).argmax(1).tolist()
        pb = mB(x).argmax(1).tolist()
        for a, b, t in zip(pa, pb, y.tolist()):
            T.append(t)
            P.append(2 if a == 0 else b)

print('chained test_acc:', round(accuracy_score(T, P), 4))
print(classification_report(T, P, target_names=names3, zero_division=0))
print('confusion:', confusion_matrix(T, P).tolist())
