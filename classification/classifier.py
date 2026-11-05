"""
EfficientNet-based drone type / model classifier.

Responsibilities:
- Crop a tracked detection
- Run EfficientNet inference
- Return class name + confidence

IMPORTANT:
- Never silently use random weights.
- If trained weights are missing → status = "CLASSIFIER OFFLINE"
- Does NOT label anything as "enemy" / "hostile".
  Authorization is handled by a later module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch / torchvision not available – classifier disabled")


# Default class list (can be overridden via config)
DEFAULT_CLASSES = [
    "quadcopter",
    "hexacopter",
    "fixed_wing",
    "racing_drone",
    "unknown_drone",
]


class EfficientNetClassifier:
    """
    Visual classifier for drone type / model.

    Output example:
    {
        "class_name": "quadcopter",
        "confidence": 0.93,
        "status": "ok",          # or "CLASSIFIER OFFLINE"
        "model": "EfficientNet-B0"
    }
    """

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        class_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.50,
        device: str = "auto",
        image_size: int = 224,
        architecture: str = "efficientnet_b0",
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.class_names = class_names or DEFAULT_CLASSES
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.architecture = architecture

        self.model: Optional[nn.Module] = None
        self.device = "cpu"
        self.enabled = False
        self.status = "CLASSIFIER OFFLINE"
        self.model_name = architecture

        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available – classifier remains offline")
            return

        self.device = self._select_device(device)
        self.transform = self._build_transform()

        # Attempt to load weights
        self._load_model()

    def _select_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for classifier but unavailable → CPU")
            return "cpu"
        return device

    def _build_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _create_backbone(self, num_classes: int) -> nn.Module:
        """Create EfficientNet backbone with correct head."""
        if self.architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif self.architecture == "efficientnet_b1":
            model = models.efficientnet_b1(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif self.architecture == "efficientnet_b2":
            model = models.efficientnet_b2(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            # Fallback
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            self.architecture = "efficientnet_b0"
        return model

    def _load_model(self):
        """
        Load trained weights.
        If weights file is missing or invalid → stay offline and log clearly.
        Never fall back to random / ImageNet weights silently.
        """
        if self.weights_path is None or not self.weights_path.exists():
            msg = (
                f"No trained EfficientNet weights found at: {self.weights_path}. "
                "Classifier remains OFFLINE. "
                "Train a model with training/train_efficientnet.py and place the "
                "checkpoint at models/efficientnet/best.pth"
            )
            logger.warning(msg)
            self.enabled = False
            self.status = "CLASSIFIER OFFLINE"
            return

        try:
            checkpoint = torch.load(
                str(self.weights_path),
                map_location=self.device,
                weights_only=False,  # allow full checkpoint dict
            )

            # Support both pure state_dict and full training checkpoint
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                # Prefer class list stored inside checkpoint if present
                if "class_names" in checkpoint:
                    self.class_names = checkpoint["class_names"]
                if "architecture" in checkpoint:
                    self.architecture = checkpoint["architecture"]
            else:
                state_dict = checkpoint

            num_classes = len(self.class_names)
            self.model = self._create_backbone(num_classes)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.enabled = True
            self.status = "ok"
            self.model_name = f"EfficientNet-{self.architecture.split('_')[-1].upper()}"
            logger.info(
                f"EfficientNet classifier loaded | "
                f"weights={self.weights_path.name} | "
                f"classes={self.class_names} | "
                f"device={self.device}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load EfficientNet weights from {self.weights_path}: {e}. "
                "Classifier remains OFFLINE."
            )
            self.model = None
            self.enabled = False
            self.status = "CLASSIFIER OFFLINE"

    def preprocess_crop(self, frame: np.ndarray, bbox: List[int]) -> Optional[torch.Tensor]:
        """Crop bbox from frame and prepare tensor."""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # OpenCV is BGR → convert to RGB for torchvision
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb)
        return tensor.unsqueeze(0)  # add batch dim

    @torch.no_grad()
    def classify(
        self,
        frame: np.ndarray,
        bbox: List[int],
    ) -> Dict[str, Any]:
        """
        Classify a single crop.

        Returns always a dict so callers never crash:
        {
            "class_name": str,
            "confidence": float,
            "status": "ok" | "CLASSIFIER OFFLINE",
            "model": str
        }
        """
        result = {
            "class_name": "N/A",
            "confidence": 0.0,
            "status": self.status,
            "model": self.model_name if self.enabled else "N/A",
        }

        if not self.enabled or self.model is None:
            return result

        tensor = self.preprocess_crop(frame, bbox)
        if tensor is None:
            result["class_name"] = "invalid_crop"
            return result

        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        conf = float(conf.item())
        idx = int(idx.item())

        class_name = self.class_names[idx] if 0 <= idx < len(self.class_names) else "unknown_drone"

        # Optional threshold
        if conf < self.confidence_threshold:
            class_name = "unknown_drone"

        result.update({
            "class_name": class_name,
            "confidence": round(conf, 3),
            "status": "ok",
            "model": self.model_name,
        })
        return result

    def classify_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Classify every track in the list.
        Adds / updates keys: classification, class_name, class_confidence
        """
        for trk in tracks:
            bbox = trk.get("bbox")
            if bbox is None:
                trk["classification"] = {
                    "class_name": "N/A",
                    "confidence": 0.0,
                    "status": self.status,
                    "model": "N/A",
                }
                continue

            cls_result = self.classify(frame, bbox)
            trk["classification"] = cls_result
            # Convenience top-level keys
            trk["class_name"] = cls_result["class_name"]
            trk["class_confidence"] = cls_result["confidence"]

        return tracks

    def is_online(self) -> bool:
        return self.enabled and self.status == "ok"

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "model": self.model_name,
            "device": self.device,
            "weights": str(self.weights_path) if self.weights_path else None,
            "classes": self.class_names,
        }
