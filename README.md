# AI-Based Live Drone Detection & Surveillance System

Production-structured real-time drone detection pipeline with YOLO, DeepSORT, EfficientNet, authorization, range estimation, optional RF, sensor fusion, threat engine, FastAPI, professional dashboard, SQLite logging, and RTX 2050 optimizations.

## Stages (all complete)

| Stage | Feature |
|-------|---------|
| 1 | Camera + YOLO detection |
| 2 | DeepSORT multi-object tracking |
| 3 | EfficientNet classification (offline until weights provided) |
| 4 | Authorization registry (AUTHORIZED / UNKNOWN / UNCONFIRMED) |
| 5 | Monocular range estimation |
| 6 | Optional receive-only RF (SDR) |
| 7 | Sensor fusion |
| 8 | Threat engine + restricted zones |
| 9 | FastAPI backend + WebSocket |
| 10 | Professional dark surveillance dashboard |
| 11 | SQLite event logging + JPEG snapshots |
| 12 | RTX 2050 optimization (adaptive skip, VRAM hygiene) |
| 13 | Unit/integration tests + training helpers |

---

## Quick Start (Windows + RTX 2050)

```powershell
cd antidrone
python -m venv venv
venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python main.py --source 0
```

**Dashboard:** http://127.0.0.1:8001/

| Input | Command |
|-------|---------|
| Webcam | `python main.py --source 0` |
| USB cam | `python main.py --source 1` |
| Video file | `python main.py --source path\to\video.mp4` |
| RTSP | `python main.py --source "rtsp://..."` |
| Headless API | `python main.py --source 0 --no-display` |

Useful flags: `--imgsz 416` `--process-every 2` `--no-rf` `--no-log` `--no-api` `--debug`

Press **q** in the OpenCV window to quit.

---

## Architecture

```
Camera → YOLO → DeepSORT → EfficientNet → Authorization
                              ↓
                         Range estimate
                              ↓
                    RF (optional, receive-only)
                              ↓
                       Sensor Fusion → Threat Engine
                              ↓
              FastAPI + WebSocket + Dashboard + SQLite/Snapshots
```

**Safety:** RF is receive-only. No jamming, spoofing, or takeover. Threat engine only raises visual/log/dashboard alerts.

---

## API (127.0.0.1:8001)

- `GET /` – Dashboard UI  
- `GET /api/status` `/api/tracks` `/api/detections` `/api/rf` `/api/alerts` `/api/events`  
- `GET /api/frame.jpg` – latest annotated frame  
- `POST /api/authorization/register`  
- `DELETE /api/authorization/{drone_id}`  
- `WS /ws/live` – real-time state stream  

---

## Missing weights behaviour (by design)

| Module | If missing |
|--------|------------|
| EfficientNet weights | `CLASSIFIER OFFLINE` – never random weights |
| SDR hardware | `RF SENSOR OFFLINE` – never fake RF values |
| GPS | Not fabricated |

Place trained classifier at: `models/efficientnet/best.pth`  
Place custom YOLO at: `models/yolo/best.pt` (or set `DETECTION.model_name`)

---

## Training

```powershell
# YOLO (prepare datasets/ per data.yaml structure first)
python training/train_yolo.py --data datasets/data.yaml --epochs 50

# EfficientNet (ImageFolder under datasets/classifier/train/<class>/)
python training/train_efficientnet.py --data datasets/classifier --epochs 20

# Evaluate
python training/evaluate.py --task yolo --weights models/yolo/best.pt --data datasets/data.yaml
python training/evaluate.py --task clf --weights models/efficientnet/best.pth --data datasets/classifier/val
```

---

## Tests (Stage 13)

```powershell
pip install pytest httpx
pytest tests/ -q
```

Coverage includes: range, authorization, threat, fusion, RF offline, geo zones, classifier offline, database logging, FastAPI endpoints, YOLO smoke (CPU), and an offline integration pipeline.

---

## RTX 2050 notes

- Default model: **yolo11n.pt**, imgsz **640**, FP16 **off**
- Adaptive frame skip if inference FPS drops
- CUDA cache cleared periodically
- Prefer `--imgsz 416` if you need higher FPS

---

## Project layout

```
antidrone/
├── main.py
├── config/          settings, calibration, yolo.yaml
├── camera/ detection/ tracking/ classification/
├── authorization/ database/ range/ rf/ fusion/
├── geo/ threat/ optimization/
├── dashboard/       FastAPI + static UI
├── training/        train_yolo, train_efficientnet, evaluate
├── tests/           unit + integration
├── models/ yolo/ efficientnet/
├── events/images/   snapshots
├── data/            SQLite DB
└── datasets/        training data (your own)
```

---

## Deployment checklist

1. Install CUDA-enabled PyTorch matching your driver  
2. `pip install -r requirements.txt`  
3. (Optional) Train custom YOLO / EfficientNet and place weights  
4. (Optional) Connect RTL-SDR + `pip install pyrtlsdr`  
5. Run `python main.py --source <cam|rtsp>`  
6. Open http://127.0.0.1:8001/  
7. Do **not** bind API to `0.0.0.0` unless behind auth/firewall  

Default API bind: **127.0.0.1** only.

---

Built incrementally Stages 1–13. Modular, no silent fake RF/range/identity values, graceful degradation when optional hardware or weights are absent.



commands :
cd C:\Users\sonac\Downloads\antidrone

# 2. Environment
.\venv_new\Scripts\Activate.ps1

# 3. Run
python main.py --source 0

# 4. Browser
# http://127.0.0.1:8001/
