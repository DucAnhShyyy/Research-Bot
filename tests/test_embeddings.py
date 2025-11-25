# tests/test_embeddings.py
from src.embeddings import EmbeddingModel
import numpy as np

def test_embed_single(mock_embedding_model):
    emb = mock_embedding_model.embed_text("hello world")
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == 384

def test_embed_batch(mock_embedding_model):
    embs = mock_embedding_model.embed_texts(["a", "b", "c"])
    assert embs.shape == (3, 384)