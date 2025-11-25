# tests/test_retriever.py
from src.retriever_hybrid import HybridRetriever
from unittest.mock import MagicMock

def test_rerank(monkeypatch):
    r = HybridRetriever()

    # mock dense
    monkeypatch.setattr(r, "dense_search", lambda q, k: [
        {"text": "alpha", "meta": {"source": "a", "chunk_id": 1}}
    ])

    # mock bm25
    monkeypatch.setattr(r, "bm25_search", lambda q, k: [
        {"text": "beta", "meta": {"source": "b", "chunk_id": 2}}
    ])

    # mock reranker
    monkeypatch.setattr(r.reranker, "predict", lambda pairs: [0.9, 0.5])

    res = r.merge_and_rerank("test", top_k=2)

    assert len(res) == 2
    assert res[0]['text'] in ["alpha", "beta"]