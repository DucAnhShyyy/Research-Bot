# tests/test_api.py
from fastapi.testclient import TestClient
from src.api_main import app
from unittest.mock import MagicMock

client = TestClient(app)

def test_api_query(monkeypatch):
    monkeypatch.setattr("src.api_main.retriever.merge_and_rerank",
                        lambda q, top_k: [{"meta": {"source": "a.pdf", "chunk_id": 1, "text": "hello"}}])

    monkeypatch.setattr("src.api_main.generator.generate",
                        lambda q, retrieved: "test answer")

    res = client.post("/query", json={"question": "hi", "top_k": 3})
    assert res.status_code == 200
    assert res.json()["answer"] == "test answer"