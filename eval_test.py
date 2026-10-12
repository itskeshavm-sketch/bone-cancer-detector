import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu')
names = ckpt['classes']
print('classes:', names, 'val_acc:', round(float(ckpt.get('acc', 0)), 4))

m = efficientnet_b0(weights=None)
m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(names)))
m.load_state_dict(ckpt['model'])
m.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder('data/test', transform=tf)
ld = DataLoader(ds, batch_size=64, num_workers=0)
P, T = [], []
with torch.no_grad():
    for x, y in ld:
        P.extend(m(x).argmax(1).tolist())
        T.extend(y.tolist())

print('test_acc:', round(accuracy_score(T, P), 4))
print(classification_report(T, P, target_names=names, zero_division=0))
print('confusion (rows=true, cols=pred):', confusion_matrix(T, P).tolist())
print('class order:', ds.classes)
