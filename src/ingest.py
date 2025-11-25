# # src/ingest.py
# """
# Ingest pipeline: PDF -> text extraction -> chunking -> embeddings -> Qdrant upsert
# Usage:
#     python -m src.ingest --data_dir sample_data --collection papers --model all-MiniLM-L6-v2
# """
# import os
# import argparse
# from pathlib import Path
# import logging
# from tqdm import tqdm

# import pdfplumber
# import fitz  # pymupdf

# from .embeddings import EmbeddingModel
# from .vectorstore_qdrant import QdrantStore

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# def extract_text_from_pdf(path: str) -> str:
#     """
#     Extract text using pdfplumber first, fallback to pymupdf.
#     Does not perform OCR; extend with pytesseract if needed.
#     """
#     parts = []
#     try:
#         with pdfplumber.open(path) as pdf:
#             for page in pdf.pages:
#                 txt = page.extract_text() or ""
#                 parts.append(txt)
#     except Exception:
#         try:
#             doc = fitz.open(path)
#             for page in doc:
#                 parts.append(page.get_text() or "")
#         except Exception as e:
#             logger.error("Failed to extract text from %s: %s", path, e)
#     return "\n\n".join(parts)

# # def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
# #     """
# #     Chunk by approximate token count using whitespace tokens.
# #     Returns list of dicts {'chunk_id': int, 'text': str}
# #     """
# #     words = text.split()
# #     i = 0
# #     chunks = []
# #     cid = 0
# #     while i < len(words):
# #         j = min(len(words), i + chunk_size)
# #         chunk = " ".join(words[i:j])
# #         chunks.append({"chunk_id": cid, "text": chunk})
# #         cid += 1
# #         i = j - overlap
# #         if i < 0:
# #             i = 0
# #         if i >= len(words):
# #             break
# #     return chunks

# def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
#     words = text.split()
#     chunks = []
#     cid = 0
#     i = 0

#     while i < len(words):
#         j = min(len(words), i + chunk_size)
#         chunk_words = words[i:j]
#         chunk = " ".join(chunk_words)

#         chunks.append({
#             "chunk_id": cid,
#             "text": chunk
#         })

#         cid += 1
#         # cập nhật i để tạo overlap
#         i = i + (chunk_size - overlap)
#         if i < 0:
#             break

#     return chunks

# def index_folder(data_dir: str, collection_name: str = None, model_name: str = None):
#     """
#     Index all PDFs in data_dir into the Qdrant collection.
#     """
#     collection_name = collection_name or os.getenv('COLLECTION_NAME', 'papers')
#     model_name = model_name or os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
#     em = EmbeddingModel(model_name=model_name)
#     vs = QdrantStore(collection=collection_name, vector_size=em.model.get_sentence_embedding_dimension())

#     p = Path(data_dir)
#     pdfs = list(p.glob('**/*.pdf'))
#     if not pdfs:
#         logger.warning('No PDF files found in %s', data_dir)
#         return

#     for pdf in tqdm(pdfs, desc='Indexing PDFs'):
#         text = extract_text_from_pdf(str(pdf))
#         if not text.strip():
#             logger.warning('No text for %s', pdf)
#             continue
#         chunks = chunk_text(text)
#         points = []
#         for c in chunks:
#             emb = em.embed_text(c['text'])
#             payload = {
#                 'source': pdf.name,
#                 'chunk_id': c['chunk_id'],
#                 'text': c['text'][:2000]
#             }
#             points.append({'id': None, 'vector': emb, 'payload': payload})
#         vs.upsert_points(points)
#     logger.info('Indexing finished')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str, default='sample_data')
#     parser.add_argument('--collection', type=str, default=None)
#     parser.add_argument('--model', type=str, default=None)
#     args = parser.parse_args()
#     index_folder(args.data_dir, collection_name=args.collection, model_name=args.model)

# src/ingest.py
import os
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

import pdfplumber
import fitz  # pymupdf

from .embeddings import EmbeddingModel
from .vectorstore_qdrant import QdrantStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_text_from_pdf(path: str) -> str:
    parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                parts.append(txt)
    except Exception:
        try:
            doc = fitz.open(path)
            for page in doc:
                parts.append(page.get_text() or "")
        except Exception as e:
            logger.error("Failed to extract text from %s: %s", path, e)
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
    words = text.split()
    chunks = []
    cid = 0
    i = 0

    while i < len(words):
        j = min(len(words), i + chunk_size)
        chunk_words = words[i:j]
        chunk = " ".join(chunk_words)

        chunks.append({
            "chunk_id": cid,
            "text": chunk
        })

        cid += 1
        i = i + (chunk_size - overlap)
        if i < 0:
            break

    return chunks


def index_folder(data_dir: str, collection_name: str = None, model_name: str = None):
    collection_name = collection_name or os.getenv("COLLECTION_NAME", "papers")
    model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    em = EmbeddingModel(model_name=model_name)
    vs = QdrantStore(
        collection=collection_name,
        vector_size=em.model.get_sentence_embedding_dimension()
    )

    p = Path(data_dir)
    pdfs = list(p.glob("**/*.pdf"))
    if not pdfs:
        logger.warning("No PDF found in %s", data_dir)
        return

    global_id = 0  # 🔥 Dùng ID tăng dần → không bao giờ None

    for pdf in tqdm(pdfs, desc="Indexing PDFs"):
        text = extract_text_from_pdf(str(pdf))
        if not text.strip():
            logger.warning("No text for %s", pdf)
            continue

        chunks = chunk_text(text)
        points = []

        for c in chunks:
            emb = em.embed_text(c["text"])
            payload = {
                "source": pdf.name,
                "chunk_id": c["chunk_id"],
                "text": c["text"][:2000]
            }

            points.append({
                "id": global_id,     # <=== FIX QUAN TRỌNG
                "vector": emb,
                "payload": payload
            })

            global_id += 1  # tăng ID

        vs.upsert_points(points)

    logger.info("Indexing finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="sample_data")
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    index_folder(args.data_dir, collection_name=args.collection, model_name=args.model)