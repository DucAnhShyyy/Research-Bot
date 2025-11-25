# tests/conftest.py
import pytest
from unittest.mock import MagicMock
import numpy as np

@pytest.fixture
def dummy_embedding():
    return np.ones(384, dtype='float32')

@pytest.fixture
def mock_embedding_model(monkeypatch, dummy_embedding):
    from src import embeddings

    def fake_embed_text(text):
        return dummy_embedding

    def fake_embed_texts(texts):
        return np.stack([dummy_embedding for _ in texts], axis=0)

    monkeypatch.setattr(embeddings, "SentenceTransformer", MagicMock())
    monkeypatch.setattr(embeddings.EmbeddingModel, "embed_text", lambda self, x: dummy_embedding)
    monkeypatch.setattr(embeddings.EmbeddingModel, "embed_texts", lambda self, x: np.stack([dummy_embedding]*len(x)))

    return embeddings.EmbeddingModel()