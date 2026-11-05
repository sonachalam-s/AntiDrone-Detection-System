"""
FastAPI backend – Stage 9 + Stage 10 dashboard

Endpoints:
  GET  /                     → dashboard HTML
  GET  /api/status
  GET  /api/detections
  GET  /api/tracks
  GET  /api/classification
  GET  /api/rf
  GET  /api/alerts
  GET  /api/system
  GET  /api/config
  GET  /api/events
  POST /api/authorization/register
  DELETE /api/authorization/{drone_id}
  GET  /api/frame.jpg
  WS   /ws/live
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from dashboard.state import STATE

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Drone Detection API",
    description="Stage 9–10 – FastAPI backend + professional dashboard",
    version="1.0.0",
)

# Restrict CORS to localhost by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001",
                   "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).resolve().parent / "static"
_index_html = _static_dir / "index.html"

if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
def serve_dashboard():
    """Serve the professional surveillance dashboard."""
    if _index_html.exists():
        return FileResponse(str(_index_html), media_type="text/html")
    return JSONResponse({"ok": True, "message": "Dashboard HTML not found. API is running."})


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------
class RegisterDroneRequest(BaseModel):
    drone_id: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=64)
    type: str = ""
    description: str = ""
    status: str = "AUTHORIZED"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data}


# ------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------
@app.get("/api/status")
def api_status():
    return _ok(STATE.get_system())


@app.get("/api/system")
def api_system():
    return _ok(STATE.get_system())


@app.get("/api/detections")
def api_detections():
    return _ok(STATE.get_detections())


@app.get("/api/tracks")
def api_tracks():
    return _ok(STATE.get_tracks())


@app.get("/api/classification")
def api_classification():
    """Return classification info extracted from current tracks."""
    tracks = STATE.get_tracks()
    result = []
    for t in tracks:
        cls = t.get("classification") or {}
        result.append({
            "track_id": t.get("track_id"),
            "class_name": cls.get("class_name", "N/A"),
            "confidence": cls.get("confidence", 0.0),
            "status": cls.get("status", "CLASSIFIER OFFLINE"),
            "model": cls.get("model", "N/A"),
        })
    return _ok(result)


@app.get("/api/rf")
def api_rf():
    return _ok(STATE.get_rf())


@app.get("/api/alerts")
def api_alerts():
    return _ok(STATE.get_alerts())


@app.get("/api/config")
def api_config():
    return _ok(STATE.get_config())


@app.get("/api/events")
def api_events():
    return _ok(STATE.get_events())


@app.get("/api/frame.jpg")
def api_frame():
    """Latest annotated JPEG frame (for simple browser preview)."""
    jpeg = STATE.get_frame_jpeg()
    if not jpeg:
        raise HTTPException(status_code=404, detail="No frame available yet")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


# ------------------------------------------------------------------
# Authorization management
# ------------------------------------------------------------------
# These endpoints need a reference to the live AuthorizationRegistry.
# It is injected via app.state at startup from main.py.
def _get_registry():
    reg = getattr(app.state, "auth_registry", None)
    if reg is None:
        raise HTTPException(status_code=503, detail="Authorization registry not available")
    return reg


@app.post("/api/authorization/register")
def api_register_drone(body: RegisterDroneRequest):
    reg = _get_registry()
    ok = reg.register(
        drone_id=body.drone_id,
        model=body.model,
        type_=body.type,
        description=body.description,
        status=body.status,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to register drone")
    STATE.add_event("authorization", f"Registered {body.drone_id} ({body.model})")
    return _ok({"drone_id": body.drone_id, "model": body.model})


@app.delete("/api/authorization/{drone_id}")
def api_delete_drone(drone_id: str):
    reg = _get_registry()
    ok = reg.remove(drone_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    STATE.add_event("authorization", f"Removed {drone_id}")
    return _ok({"drone_id": drone_id, "removed": True})


@app.get("/api/authorization")
def api_list_authorized():
    reg = _get_registry()
    return _ok(reg.list_all())


# ------------------------------------------------------------------
# WebSocket – live updates
# ------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info("WebSocket client connected (%d total)", len(self.active))

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info("WebSocket client disconnected (%d total)", len(self.active))

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial snapshot
        await websocket.send_json(STATE.snapshot())
        while True:
            # Keep connection alive; client can send ping or we just sleep
            try:
                # Wait for client message with timeout, or just push periodically
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                if msg == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Push latest state every ~1s
                await websocket.send_json(STATE.snapshot())
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


# Background broadcaster (optional – called from pipeline if desired)
async def push_state():
    if manager.active:
        await manager.broadcast(STATE.snapshot())


@app.on_event("startup")
async def on_startup():
    STATE.update_system(status="running")
    STATE.add_event("system", "API server started")
    logger.info("FastAPI backend started")


@app.on_event("shutdown")
async def on_shutdown():
    STATE.update_system(status="stopped")
    STATE.add_event("system", "API server stopped")
    logger.info("FastAPI backend stopped")
