import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ckpt = torch.load('kaggle_out/best_model.pt', map_location='cpu')
names = ckpt['classes']
S = int(ckpt.get('img', 300))
print('val_acc:', round(float(ckpt.get('acc', 0)), 4), 'img:', S)

m = efficientnet_b3(weights=None)
m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, len(names)))
m.load_state_dict(ckpt['model'])
m.eval()

tf = transforms.Compose([
    transforms.Resize((S, S)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder('data/test', transform=tf)
ld = DataLoader(ds, batch_size=16, num_workers=0)
P, T = [], []
with torch.no_grad():
    for x, y in ld:
        P.extend(m(x).argmax(1).tolist())
        T.extend(y.tolist())

print('test_acc:', round(accuracy_score(T, P), 4))
print(classification_report(T, P, target_names=names, zero_division=0))
print('confusion (rows=true, cols=pred):', confusion_matrix(T, P).tolist())
