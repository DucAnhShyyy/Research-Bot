# tests/test_vectorstore_qdrant.py
from unittest.mock import MagicMock
from src.vectorstore_qdrant import QdrantStore

def test_qdrant_upsert(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr("src.vectorstore_qdrant.QdrantClient", lambda host, port: mock_client)

    vs = QdrantStore(collection="test", vector_size=384)

    vs.upsert_points([{
        'id': None,
        'vector': [0.1]*384,
        'payload': {"text": "hello", "source": "a.pdf"}
    }])

    assert mock_client.upsert.called