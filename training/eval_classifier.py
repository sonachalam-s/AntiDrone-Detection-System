import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

val_dir = Path("datasets/classifier/val")
weights = Path("models/efficientnet/best.pth")

if not val_dir.exists():
    raise SystemExit(f"Missing {val_dir}")
if not weights.exists():
    raise SystemExit(f"Missing {weights}")

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder(str(val_dir), transform=tf)
loader = DataLoader(ds, batch_size=8, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(str(weights), map_location=device, weights_only=False)
if isinstance(ckpt, dict):
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    classes = ckpt.get("classes", ds.classes)
else:
    state = ckpt
    classes = ds.classes

if any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

n = len(classes)
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
model.load_state_dict(state, strict=False)
model.to(device).eval()

idx_to_true = {i: c for c, i in ds.class_to_idx.items()}
correct = total = 0
cm = defaultdict(int)

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu()
        for ti, pi in zip(y.tolist(), pred.tolist()):
            tname = idx_to_true[ti]
            pname = classes[pi] if pi < len(classes) else str(pi)
            cm[(tname, pname)] += 1
            if tname == pname:
                correct += 1
            total += 1

print("=" * 50)
print("CLASSIFICATION EVALUATION")
print("=" * 50)
print("Val images :", total)
print("Classes    :", list(classes))
print("Accuracy   : {:.2f}%  ({}/{})".format(100.0 * correct / max(total, 1), correct, total))
print()
print("Confusion matrix (rows=true, cols=pred):")
print()

names = list(ds.classes)
header = "{:12s}".format("true/pred") + "".join("{:>12s}".format(n[:10]) for n in names)
print(header)
for t in names:
    row = "{:12s}".format(t[:12])
    for p in names:
        row += "{:12d}".format(cm[(t, p)])
    print(row)

print()
print("Per-class recall (correct / true count):")
for t in names:
    tot_t = sum(cm[(t, p)] for p in names)
    hit = cm[(t, t)]
    rec = 100.0 * hit / tot_t if tot_t else 0.0
    print("  {:12s}  {:6.2f}%  ({}/{})".format(t, rec, hit, tot_t))
