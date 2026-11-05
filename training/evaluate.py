"""
Evaluation helper – prints basic metrics after training.

YOLO:
  python training/evaluate.py --task yolo --weights models/yolo/best.pt --data datasets/data.yaml

Classifier:
  python training/evaluate.py --task clf --weights models/efficientnet/best.pth --data datasets/classifier/val
"""

from __future__ import annotations

import argparse
from pathlib import Path


def eval_yolo(weights: str, data: str):
    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = model.val(data=data)
    print("YOLO validation metrics:")
    print(metrics)


def eval_clf(weights: str, data: str):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names", [])
    arch = ckpt.get("architecture", "efficientnet_b0")
    state = ckpt.get("model_state_dict", ckpt)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names) or 5)
    model.load_state_dict(state)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(data, transform=tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / max(total, 1)
    print(f"Classifier accuracy: {acc:.3f}  ({correct}/{total})")
    print(f"Classes: {ds.classes}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["yolo", "clf"], required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    args = p.parse_args()

    if not Path(args.weights).exists():
        print(f"Weights not found: {args.weights}")
        return
    if args.task == "yolo":
        eval_yolo(args.weights, args.data)
    else:
        eval_clf(args.weights, args.data)


if __name__ == "__main__":
    main()
