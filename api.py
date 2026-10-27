"""
NetraChat backend API (Python).

The Node.js frontend (frontend/) talks to this server. Python stays in
charge of the AI work (object detection, understanding, replies); Node.js
serves the web page.

Run:      python api.py                 (http://127.0.0.1:8000)
Options:  --blip            enable BLIP captioning + question answering
          --weights FILE    YOLO weights (default yolov8s.pt)
          --port 8000

Endpoints (all JSON):
  GET  /api/health            -> backend status
  POST /api/session           -> {"session_id": "..."}   start a conversation
  POST /api/image             -> upload an image (form fields: session_id, file)
  POST /api/chat              -> {"session_id": "...", "message": "..."}
  POST /api/reset             -> {"session_id": "..."}   forget image and chat
"""

from __future__ import annotations

import argparse
import io
import threading
import uuid
from collections import OrderedDict

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from netrachat import NetraChat
from netrachat.nlu.intent import IntentClassifier
from netrachat.vision.captioner import VisionLanguageModel
from netrachat.vision.detector import DEFAULT_WEIGHTS, ObjectDetector

MAX_IMAGE_BYTES = 15 * 1024 * 1024   # 15 MB
MAX_SESSIONS = 200                   # oldest conversations are forgotten first


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SessionRequest(BaseModel):
    session_id: str


def create_app(detector, vlm=None, intents=None) -> FastAPI:
    """Build the API. The heavy models are loaded once and shared by everyone;
    each conversation (session) gets its own NetraChat with its own memory."""
    intents = intents or IntentClassifier()
    app = FastAPI(title="NetraChat API")
    sessions: OrderedDict[str, NetraChat] = OrderedDict()
    lock = threading.Lock()   # the models are not safe to run twice at once

    def get_bot(session_id: str) -> NetraChat:
        bot = sessions.get(session_id)
        if bot is None:
            raise HTTPException(404, "Unknown session. Please reload the page.")
        sessions.move_to_end(session_id)
        return bot

    @app.get("/api/health")
    def health():
        return {"status": "ok", "blip": vlm is not None, "sessions": len(sessions)}

    @app.post("/api/session")
    def new_session():
        session_id = uuid.uuid4().hex
        sessions[session_id] = NetraChat(detector=detector, vlm=vlm,
                                         intent_classifier=intents)
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
        return {"session_id": session_id}

    @app.post("/api/image")
    def upload_image(session_id: str = Form(...), file: UploadFile = File(...)):
        bot = get_bot(session_id)
        data = file.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "The image is too large (maximum 15 MB).")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, ImportError):
            # (Ultralytics patches Image.open and may raise ImportError for
            # unknown files while looking for an optional HEIC plugin.)
            raise HTTPException(400, "That file is not an image I can open. "
                                     "Please upload a JPG or PNG.")
        with lock:
            message = bot.load_image(image)
        scene = bot.scene
        return {
            "message": message,
            "width": scene.width,
            "height": scene.height,
            "caption": scene.caption,
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                    "box": [round(v, 1) for v in d.box],
                    "color": d.color,
                    "position": scene.position(d),
                }
                for d in scene.detections
            ],
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        bot = get_bot(req.session_id)
        message = req.message.strip()[:500]
        with lock:
            reply = bot.reply(message)
        return {"reply": reply, "intent": bot.last_intent, "focus": bot.focus}

    @app.post("/api/reset")
    def reset(req: SessionRequest):
        get_bot(req.session_id).reset()
        return {"status": "ok"}

    return app


def main():
    parser = argparse.ArgumentParser(description="NetraChat backend API")
    parser.add_argument("--blip", action="store_true", help="enable BLIP models")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("Loading models, please wait…")
    app = create_app(ObjectDetector(args.weights),
                     VisionLanguageModel() if args.blip else None)
    print(f"NetraChat API ready on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
