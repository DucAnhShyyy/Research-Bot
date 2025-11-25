# src/embeddings.py
"""
embeddings.py
Wrapper around sentence-transformers for consistent embed interface.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

class EmbeddingModel:
    def __init__(self, model_name: str = MODEL):
        """
        Initialize the SentenceTransformer model.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        """
        Embed single text -> 1D numpy array (float32).
        """
        emb = self.model.encode(text, show_progress_bar=False)
        return np.array(emb, dtype='float32')

    def embed_texts(self, texts):
        """
        Embed list of texts -> 2D numpy array.
        """
        embs = self.model.encode(texts, show_progress_bar=True)
        return np.array(embs, dtype='float32')