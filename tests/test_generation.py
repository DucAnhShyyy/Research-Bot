# tests/test_generation.py
from src.generation_strict import StrictGenerator
from unittest.mock import MagicMock

def test_prompt_building():
    g = StrictGenerator()
    g.pipe = MagicMock(return_value=[{"generated_text": "Answer [DOC:a.pdf|chunk:1]"}])

    retrieved = [
        {"meta": {"source": "a.pdf", "chunk_id": 1, "text": "content"}}
    ]

    out = g.generate("What?", retrieved)
    assert "[DOC:a.pdf|chunk:1]" in out