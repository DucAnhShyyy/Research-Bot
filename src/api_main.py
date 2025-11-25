# src/api_main.py
"""
FastAPI production-style endpoints (ingest & query). This is a skeleton for extension.
Run with:
    uvicorn src.api_main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
from .ingest import index_folder
from .retriever_hybrid import HybridRetriever
from .generation_strict import StrictGenerator

app = FastAPI(title='Research Assistant API')
retriever = HybridRetriever()
generator = StrictGenerator()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post('/upload')
async def upload_pdf(file: UploadFile = File(...)):
    """
    Save uploaded PDF then call indexer immediately (blocking). In production, enqueue a background job.
    """
    out_path = os.path.join('sample_data', file.filename)
    with open(out_path, 'wb') as f:
        f.write(await file.read())
    index_folder('sample_data')
    return {'status': 'indexed', 'filename': file.filename}

@app.post('/query')
async def query(req: QueryRequest):
    """
    Query endpoint: returns answer + candidates list.
    """
    candidates = retriever.merge_and_rerank(req.question, top_k=req.top_k)
    answer = generator.generate(req.question, candidates)
    return {'answer': answer, 'candidates': candidates}