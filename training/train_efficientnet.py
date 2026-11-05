"""
EfficientNet training script – Stage 13 helper

Expects ImageFolder layout:
  datasets/classifier/
    train/{quadcopter,hexacopter,fixed_wing,racing_drone,unknown_drone}/
    val/...

Example:
  python training/train_efficientnet.py --data datasets/classifier --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet drone classifier")
    parser.add_argument("--data", type=str, default="datasets/classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--arch", type=str, default="efficientnet_b0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="models/efficientnet/best.pth")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists():
        print(f"Training folder not found: {train_dir}")
        print("Create ImageFolder structure under datasets/classifier/train/<class_name>/")
        return

    tf_train = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=tf_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = None
    if val_dir.exists():
        val_ds = datasets.ImageFolder(str(val_dir), transform=tf_val)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")

    if args.arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_acc = correct / max(total, 1)
        print(f"Epoch {epoch}/{args.epochs}  loss={loss_sum/total:.4f}  acc={train_acc:.3f}")

        if val_loader:
            model.eval()
            vtotal, vcorrect = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x).argmax(1)
                    vcorrect += (pred == y).sum().item()
                    vtotal += x.size(0)
            vacc = vcorrect / max(vtotal, 1)
            print(f"  val_acc={vacc:.3f}")
            if vacc >= best_acc:
                best_acc = vacc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": train_ds.classes,
                    "architecture": args.arch,
                }, out_path)
                print(f"  saved {out_path}")
        else:
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_ds.classes,
                "architecture": args.arch,
            }, out_path)

    print(f"Done. Weights: {out_path}")


if __name__ == "__main__":
    main()
