# tests/test_health.py
from fastapi.testclient import TestClient
import os
import sys
# Ensure the app module is importable when running from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(PROJECT_ROOT))

from api import app

client = TestClient(app)

def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    # FastAPI returns JSON by default
    assert resp.headers.get("content-type", "").startswith("application/json")
    assert resp.json() == {"status": "ok"}
