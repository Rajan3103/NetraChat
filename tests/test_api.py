"""Tests for the backend API used by the Node.js frontend."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import create_app
from netrachat.nlu.intent import IntentClassifier
from test_chatbot import SCENE, FakeDetector


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(FakeDetector(SCENE), intents=IntentClassifier()))


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (900, 600), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_full_conversation(client):
    assert client.get("/api/health").json()["status"] == "ok"
    sid = client.post("/api/session").json()["session_id"]

    res = client.post("/api/image", data={"session_id": sid},
                      files={"file": ("photo.png", png_bytes(), "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert data["width"] == 900 and data["height"] == 600
    assert {d["label"] for d in data["detections"]} == {"dog", "cat", "couch", "person"}
    assert "two people" in data["message"]

    chat = client.post("/api/chat", json={"session_id": sid, "message": "how many dogs?"})
    assert chat.json() == {"reply": "There is one dog in the image.", "intent": "count",
                           "focus": ["dog"]}

    assert client.post("/api/reset", json={"session_id": sid}).status_code == 200
    again = client.post("/api/chat", json={"session_id": sid, "message": "how many dogs?"})
    assert "upload an image first" in again.json()["reply"].lower()


def test_rejects_files_that_are_not_images(client):
    sid = client.post("/api/session").json()["session_id"]
    res = client.post("/api/image", data={"session_id": sid},
                      files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_unknown_session(client):
    res = client.post("/api/chat", json={"session_id": "missing", "message": "hi"})
    assert res.status_code == 404
