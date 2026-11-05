"""
YOLO custom training script – Stage 13 helper

Example:
  python training/train_yolo.py --data datasets/data.yaml --epochs 50 --imgsz 640

Requires a prepared dataset:
  datasets/
    images/{train,val,test}/
    labels/{train,val,test}/
    data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train custom YOLO drone detector")
    parser.add_argument("--data", type=str, default="datasets/data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base pretrained model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--project", type=str, default="models/yolo")
    parser.add_argument("--name", type=str, default="train")
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset config not found: {data_path}")
        print("Create datasets/data.yaml with paths and class names, e.g.:")
        print("  path: .")
        print("  train: images/train")
        print("  val: images/val")
        print("  names: [drone, bird, helicopter, aircraft, unknown_object]")
        return

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # Copy best weights to expected location
    best = Path(args.project) / args.name / "weights" / "best.pt"
    dest = Path("models/yolo/best.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        import shutil
        shutil.copy(best, dest)
        print(f"Best weights saved to {dest}")
    else:
        print("Training finished – check runs folder for weights.")


if __name__ == "__main__":
    main()
