"""FastAPI backend: MJPEG video stream, WebSocket event push, small REST
control surface. The front end is static files (no Node build step).

Endpoints:
  GET  /                     static index.html
  GET  /api/state            full UI snapshot (engine.state())
  GET  /video/stream         MJPEG multipart stream of annotated frames
  WS   /ws                   event history catch-up + live events
  POST /api/session/start    start engine (recipe already chosen at boot)
  POST /api/session/stop
  POST /api/scan/begin       {ingredient} begin 10 s capture
  POST /api/scan/retake      {ingredient}
  POST /api/scan/skip
  POST /api/scan/finish      end scan phase -> cooking
  POST /api/verb             {verb} keyboard-mode verb injection
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline.engine import CookingEngine


def create_app(engine: CookingEngine, static_dir) -> FastAPI:
    """App factory; engine is owned by scripts/run_app.py's lifecycle."""
    app = FastAPI(title="ego-kitchen")
    app.state.engine = engine

    # ---- video ----

    def mjpeg_stream():
        """Yield annotated frames as JPEG parts (~10 fps)."""
        import time
        import cv2
        last = 0.0
        while True:
            frame = engine.annotated_frame()
            now = time.monotonic()
            if frame is None or now - last < 0.1:
                time.sleep(0.02)
                continue
            last = now
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

    @app.get("/video/stream")
    def video_stream():
        return StreamingResponse(mjpeg_stream(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    # ---- state / control ----

    @app.get("/api/state")
    def state():
        return engine.state()

    @app.post("/api/session/start")
    def session_start():
        engine.start()
        return {"ok": True}

    @app.post("/api/session/stop")
    def session_stop():
        engine.stop()
        return {"ok": True}

    @app.post("/api/scan/begin")
    def scan_begin(body: dict):
        engine.begin_ingredient_scan(body.get("ingredient", ""))
        return {"ok": True}

    @app.post("/api/scan/retake")
    def scan_retake(body: dict):
        engine.retake_ingredient(body.get("ingredient", ""))
        return {"ok": True}

    @app.post("/api/scan/skip")
    def scan_skip():
        engine.skip_ingredient()
        return {"ok": True}

    @app.post("/api/scan/finish")
    def scan_finish():
        engine.finish_scan()
        return {"ok": True}

    @app.post("/api/verb")
    def inject_verb(body: dict):
        engine.inject_verb(body.get("verb", ""))
        return {"ok": True}

    # ---- websocket events ----

    @app.websocket("/ws")
    async def ws_events(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_event(event):
            loop.call_soon_threadsafe(queue.put_nowait, event.to_json())

        engine.bus.subscribe(on_event)
        # catch-up: replay recent history so a refresh/reconnect loses nothing
        for event in engine.bus.history(64):
            await ws.send_text(event.to_json())
        try:
            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=20.0)
                await ws.send_text(msg)
        except asyncio.TimeoutError:
            await ws.send_text(json.dumps({"kind": "ping"}))
        except WebSocketDisconnect:
            pass

    # ---- static front end ----
    app.mount("/", StaticFiles(directory=str(static_dir), html=True),
              name="static")
    return app
