"""
Runtime Optimizer – Stage 12 (RTX 2050 focused)

Provides:
- Recommended settings for RTX 2050 / similar 4GB GPUs
- Adaptive frame skipping when inference FPS drops
- Optional FP16 (disabled by default for stability)
- CUDA memory cleanup helpers
- Device info reporting
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# Recommended defaults for RTX 2050 (4 GB VRAM, mobile)
RTX_2050_RECOMMENDED = {
    "model_name": "yolo11n.pt",       # nano – best speed/VRAM tradeoff
    "image_size": 640,                # 416 if you need more FPS
    "process_every_n_frames": 1,      # 2 if camera is 60 fps
    "use_fp16": False,                # keep False unless tested stable
    "camera_width": 1280,
    "camera_height": 720,
    "max_det": 20,
}


def get_recommended_settings(gpu_name: str = "") -> Dict[str, Any]:
    """Return a settings dict tuned for the detected GPU."""
    name = (gpu_name or "").lower()
    if "2050" in name or "1650" in name or "1050" in name:
        return dict(RTX_2050_RECOMMENDED)
    if "3060" in name or "3070" in name or "4060" in name:
        return {
            "model_name": "yolo11s.pt",
            "image_size": 640,
            "process_every_n_frames": 1,
            "use_fp16": False,
            "camera_width": 1280,
            "camera_height": 720,
            "max_det": 30,
        }
    if "cpu" in name or not name:
        return {
            "model_name": "yolo11n.pt",
            "image_size": 416,
            "process_every_n_frames": 2,
            "use_fp16": False,
            "camera_width": 640,
            "camera_height": 480,
            "max_det": 10,
        }
    return dict(RTX_2050_RECOMMENDED)


class RuntimeOptimizer:
    """
    Adaptive performance helper used by the main loop.
    """

    def __init__(
        self,
        target_inference_fps: float = 15.0,
        min_process_every: int = 1,
        max_process_every: int = 4,
        adaptive: bool = True,
        use_fp16: bool = False,
        empty_cache_every_n: int = 300,
    ):
        self.target_inference_fps = target_inference_fps
        self.min_process_every = min_process_every
        self.max_process_every = max_process_every
        self.adaptive = adaptive
        self.use_fp16 = use_fp16
        self.empty_cache_every_n = empty_cache_every_n

        self.process_every = min_process_every
        self._frame_count = 0
        self._last_adjust = time.time()
        self._inf_fps_hist = []

        if TORCH_OK and torch.cuda.is_available():
            self._log_gpu()
            # Enable cudnn autotune for fixed input sizes
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark enabled")

        if use_fp16:
            logger.warning(
                "FP16 requested – ensure your YOLO/Ultralytics build supports it. "
                "Default remains FP32 for stability on RTX 2050."
            )

    def _log_gpu(self):
        if not TORCH_OK or not torch.cuda.is_available():
            return
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mb = props.total_memory / (1024 ** 2)
        logger.info(
            "GPU: %s | VRAM: %.0f MB | CUDA: %s",
            name, total_mb, torch.version.cuda,
        )
        if total_mb < 5000:
            logger.info(
                "Low-VRAM GPU detected – recommended: yolo11n.pt, imgsz=640, "
                "process_every_n_frames=1–2, USE_FP16=False"
            )

    def record_inference_fps(self, fps: float):
        if fps <= 0:
            return
        self._inf_fps_hist.append(fps)
        if len(self._inf_fps_hist) > 30:
            self._inf_fps_hist.pop(0)

    def maybe_adapt(self) -> int:
        """
        Adjust process_every based on recent inference FPS.
        Returns current process_every value.
        """
        self._frame_count += 1

        # Periodic CUDA cache cleanup
        if (
            TORCH_OK
            and torch.cuda.is_available()
            and self.empty_cache_every_n > 0
            and self._frame_count % self.empty_cache_every_n == 0
        ):
            self.empty_cuda_cache()

        if not self.adaptive or len(self._inf_fps_hist) < 10:
            return self.process_every

        now = time.time()
        if now - self._last_adjust < 3.0:
            return self.process_every

        avg = sum(self._inf_fps_hist) / len(self._inf_fps_hist)
        self._last_adjust = now

        if avg < self.target_inference_fps * 0.7 and self.process_every < self.max_process_every:
            self.process_every += 1
            logger.info(
                "Adaptive skip → process every %d frames (inf FPS≈%.1f)",
                self.process_every, avg,
            )
        elif avg > self.target_inference_fps * 1.3 and self.process_every > self.min_process_every:
            self.process_every -= 1
            logger.info(
                "Adaptive skip → process every %d frames (inf FPS≈%.1f)",
                self.process_every, avg,
            )

        return self.process_every

    def empty_cuda_cache(self):
        if TORCH_OK and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("CUDA cache emptied")

    def get_status(self) -> Dict[str, Any]:
        avg = (
            sum(self._inf_fps_hist) / len(self._inf_fps_hist)
            if self._inf_fps_hist else 0.0
        )
        info: Dict[str, Any] = {
            "process_every": self.process_every,
            "adaptive": self.adaptive,
            "use_fp16": self.use_fp16,
            "avg_inference_fps": round(avg, 1),
            "target_inference_fps": self.target_inference_fps,
        }
        if TORCH_OK and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_allocated_mb"] = round(
                torch.cuda.memory_allocated(0) / (1024 ** 2), 1
            )
            info["vram_reserved_mb"] = round(
                torch.cuda.memory_reserved(0) / (1024 ** 2), 1
            )
        return info
