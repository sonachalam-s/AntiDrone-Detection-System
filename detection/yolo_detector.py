"""
YOLO-based drone detector using Ultralytics.
Automatically selects CUDA if available, otherwise CPU.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLODroneDetector:
    """
    YOLO object detector focused on drones (and other classes).
    
    Output format for each detection:
    {
        "bbox": [x1, y1, x2, y2],
        "class_id": int,
        "class_name": str,
        "confidence": float
    }
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.45,
        iou: float = 0.50,
        image_size: int = 640,
        device: str = "auto",
        classes_of_interest: Optional[List[int]] = None,
        models_dir: Optional[Path] = None,
        max_det: int = 20,
        use_fp16: bool = False,
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.classes_of_interest = classes_of_interest  # None = all classes
        self.max_det = max_det
        self.use_fp16 = use_fp16 and device != "cpu"

        # Device selection
        self.device = self._select_device(device)
        self.device_name = self._get_device_name()

        # Resolve model path
        self.model_path = self._resolve_model_path(model_name, models_dir)

        # Load model
        logger.info(f"Loading YOLO model: {self.model_path} on {self.device}")
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

        # FP16 only when explicitly requested and on CUDA
        if self.use_fp16 and self.device == "cuda":
            try:
                # Ultralytics handles half via predict(half=True); keep model FP32 weights
                logger.info("FP16 inference enabled (predict half=True)")
            except Exception as e:
                logger.warning(f"FP16 setup issue: {e} – staying FP32")
                self.use_fp16 = False
        else:
            self.use_fp16 = False

        # Warm-up
        self._warmup()

        # Inference FPS
        self._inference_times: List[float] = []
        self.inference_fps = 0.0

        logger.info(
            f"YOLO ready | device={self.device} | "
            f"GPU={self.device_name} | conf={self.confidence} | "
            f"imgsz={self.image_size} | max_det={self.max_det} | fp16={self.use_fp16}"
        )

    def _select_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device

    def _get_device_name(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"

    def _resolve_model_path(self, model_name: str, models_dir: Optional[Path]) -> Path:
        """
        Resolve model path.
        - If model_name is an absolute/relative existing path → use it
        - Else look inside models_dir
        - Else let Ultralytics download the official model (yolo11n.pt etc.)
        """
        p = Path(model_name)
        if p.exists():
            return p

        if models_dir is not None:
            candidate = models_dir / model_name
            if candidate.exists():
                return candidate

        # Ultralytics will download official models automatically
        return Path(model_name)

    def _warmup(self):
        """Run a dummy inference to initialize CUDA kernels / CUDA context."""
        import numpy as np
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                max_det=self.max_det,
                half=self.use_fp16,
                verbose=False,
            )
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")

    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR frame (OpenCV format).
        Returns list of detection dictionaries.
        """
        if frame is None:
            return []

        t0 = time.perf_counter()

        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            max_det=self.max_det,
            half=self.use_fp16,
            verbose=False,
        )

        elapsed = time.perf_counter() - t0
        self._inference_times.append(elapsed)
        if len(self._inference_times) > 30:
            self._inference_times.pop(0)
        if self._inference_times:
            avg = sum(self._inference_times) / len(self._inference_times)
            self.inference_fps = round(1.0 / avg, 1) if avg > 0 else 0.0

        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = result.names  # class id → name mapping

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Optional class filter (for later stages we will keep only "drone")
            if self.classes_of_interest is not None and cls_id not in self.classes_of_interest:
                continue

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(conf, 3),
            })

        return detections

    def get_inference_fps(self) -> float:
        return self.inference_fps

    def get_device_info(self) -> Dict[str, str]:
        return {
            "device": self.device,
            "device_name": self.device_name,
            "cuda_available": str(torch.cuda.is_available()),
        }
