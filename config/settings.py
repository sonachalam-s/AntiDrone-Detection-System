"""
Central configuration for AI-Based Drone Detection & Surveillance System
Stage 1: Camera + YOLO
Stage 2: DeepSORT multi-object tracking
Stage 3: EfficientNet classification
Stage 4: Authorization registry
Stage 5: Monocular range estimation
Stage 6: Optional receive-only RF monitoring
Stage 7: Sensor fusion
Stage 8: Threat engine + restricted zones
Stage 9: FastAPI backend + WebSocket
Stage 10: Professional web dashboard
Stage 11: Database event logging + snapshots
Stage 12: RTX 2050 optimization
"""

import os
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# DETECTION (YOLO)
# ============================================================
DETECTION = {
    "model_name": "models/yolo/best.pt",          # yolo11n recommended for RTX 2050 (4GB)
    "confidence": 0.45,
    "iou": 0.50,
    "image_size": 640,                   # 416 for more FPS, 640 balanced, 1280 if VRAM allows
    "device": "auto",                    # "auto", "cuda", "cpu"
    "process_every_n_frames": 1,         # 1 = every frame; 2–3 if GPU is saturated
    "classes_of_interest": None,         # None = all classes
    "max_det": 20,                       # limit detections per frame (VRAM/speed)
    "use_fp16": False,                   # FP16 off by default for stability
}

# ============================================================
# TRACKING (DeepSORT)
# ============================================================
TRACKING = {
    "max_age": 30,                       # frames to keep lost tracks
    "n_init": 3,                         # detections before track is confirmed
    "max_iou_distance": 0.7,
    "max_cosine_distance": 0.3,
    "nn_budget": 100,
    "embedder": "mobilenet",             # lightweight embedder suitable for RTX 2050
    "half": False,                       # FP16 for embedder (keep False for stability)
    "trajectory_maxlen": 60,             # how many past centroids to keep
}

# ============================================================
# CLASSIFICATION (EfficientNet)
# ============================================================
CLASSIFICATION = {
    "enabled": True,
    "architecture": "efficientnet_b0",   # efficientnet_b0 / b1 / b2
    "weights": "models/efficientnet/best.pth",  # relative to project root
    "confidence": 0.50,
    "image_size": 224,
    "device": "auto",
    "class_names": [
        "quadcopter",
        "hexacopter",
        "fixed_wing",
        "racing_drone",
        "unknown_drone",
    ],
}

# ============================================================
# AUTHORIZATION (Stage 4)
# ============================================================
AUTHORIZATION = {
    "enabled": True,
    "seed_demo": True,                   # insert a few demo authorized drones on startup
    "db_path": None,                     # None → data/antidrone.db
}

# ============================================================
# RANGE ESTIMATION (Stage 5)
# ============================================================
RANGE = {
    "enabled": True,
    "method": "monocular",
    "calibration_file": "config/calibration.yaml",
    "default_width_m": 0.40,
    "min_pixel_width": 8,
    "max_reasonable_m": 2000.0,
}

# ============================================================
# RF MONITORING (Stage 6) – receive-only, optional
# ============================================================
RF = {
    "enabled": True,                     # set False to disable completely
    "device": "auto",                    # auto | rtlsdr | soapy
    "center_freq_hz": 2.4e9,             # 2.4 GHz common drone band
    "sample_rate": 2.4e6,
    "gain": "auto",
    "poll_interval_s": 1.0,
}

# ============================================================
# SENSOR FUSION (Stage 7)
# ============================================================
FUSION = {
    "enabled": True,
    "rf_correlation_enabled": True,      # attempt weak RF↔track association
    "rf_min_correlation_conf": 0.35,
}

# ============================================================
# THREAT ENGINE (Stage 8)
# ============================================================
THREAT = {
    "enabled": True,
    "close_range_m": 100.0,
    "very_close_range_m": 40.0,
    "persistent_hits": 15,
    "high_rf_corr_conf": 0.55,
}

# ============================================================
# RESTRICTED ZONES (used by Threat Engine)
# Image-plane zones for development (no GPS required)
# ============================================================
RESTRICTED_ZONES = [
    # Example: central region of a 1280x720 frame
    # {
    #     "name": "central_keep_out",
    #     "type": "circle",
    #     "center": [640, 360],
    #     "radius": 180,
    #     "coordinate_system": "image",
    # },
]

# ============================================================
# API / FastAPI (Stage 9)
# ============================================================
API = {
    "host": "127.0.0.1",                 # do not expose publicly by default
    "port": 8001,
    "enable": True,
    "publish_jpeg": True,                # publish annotated frame as JPEG
    "jpeg_quality": 70,
}

# ============================================================
# EVENT LOGGING + SNAPSHOTS (Stage 11)
# ============================================================
EVENT_LOGGING = {
    "enabled": True,
    "save_snapshots": True,
    "snapshot_on_alert_only": False,   # False = also snapshot high-conf detections
    "snapshot_min_conf": 0.50,
    "log_every_n_frames": 1,           # 1 = every processed frame
}

# ============================================================
# OPTIMIZATION (Stage 12) – RTX 2050 tuned
# ============================================================
OPTIMIZATION = {
    "adaptive_frame_skip": True,         # auto-increase skip if inference FPS drops
    "target_inference_fps": 15.0,        # aim for at least this many AI FPS
    "min_process_every": 1,
    "max_process_every": 4,
    "empty_cuda_cache_every_n": 300,     # frames between torch.cuda.empty_cache()
    "cudnn_benchmark": True,
    "use_fp16": False,                   # keep False unless explicitly tested
}

# ============================================================
# CAMERA
# ============================================================
CAMERA = {
    "source": 0,                         # 0 = default webcam, 1 = USB, or path/rtsp
    "width": 1280,
    "height": 720,
    "fps": 30,
    "buffer_size": 2,
    "use_dshow": True,                   # Windows DirectShow for better webcam compatibility
    "reconnect_attempts": 5,
    "reconnect_delay": 2.0,
}

# ============================================================
# DISPLAY / DEBUG
# ============================================================
DISPLAY = {
    "show_window": True,
    "window_name": "AI Drone Detection - Stage 12 (Optimized)",
    "draw_fps": True,
    "draw_device": True,
    "draw_trajectory": True,             # draw recent path of each track
    "trajectory_length": 30,             # how many points to draw
    "draw_classification": True,
    "draw_authorization": True,
    "draw_range": True,
    "draw_rf": True,
    "draw_fusion_summary": True,
    "draw_threat": True,
}

# ============================================================
# PATHS
# ============================================================
PATHS = {
    "models_dir": BASE_DIR / "models" / "yolo",
    "efficientnet_dir": BASE_DIR / "models" / "efficientnet",
    "events_images": BASE_DIR / "events" / "images",
    "data_dir": BASE_DIR / "data",
}

# Create directories if missing
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
}
