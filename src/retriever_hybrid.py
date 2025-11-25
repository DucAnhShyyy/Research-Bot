# retriever_hybrid.py
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridRetriever:
    """
    A hybrid retriever combining BM25 + Dense vector search + Rerank merging.
    Compatible with any Qdrant version (old/new) thanks to safe tuple parsing.
    """

    def __init__(self, qdrant_collection: str = "papers", host="localhost", port=6333):
        self.collection = qdrant_collection
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ----------------------
    # SAFE TUPLE PARSER
    # ----------------------
    def _parse_hit(self, hit):
        """
        Convert Qdrant search result (tuple or ScoredPoint) into:
        { 'id': ..., 'score': ..., 'payload': {...}, 'vector': [...] }
        Works for ALL Qdrant versions.
        """

        # Case 1: New Qdrant — hit is ScoredPoint
        if hasattr(hit, "payload"):
            return {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {},
                "vector": hit.vector or None,
            }

        # Case 2: Old Qdrant — hit is tuple
        # Possible tuple formats:
        # (id, score)
        # (id, score, payload)
        # (id, score, payload, vector)
        if isinstance(hit, tuple):
            h = list(hit) + [None] * (4 - len(hit))  # pad to 4 fields
            doc_id, score, payload, vector = h[:4]
            return {
                "id": doc_id,
                "score": score,
                "payload": payload or {},
                "vector": vector,
            }

        raise ValueError(f"Unknown hit format from Qdrant: {hit}")

    # ----------------------
    # BM25 SEARCH
    # ----------------------
    def bm25_search(self, query: str, k=10):
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query,
            limit=k,
            using="bm25",
        )

        return [self._parse_hit(h) for h in hits]

    # ----------------------
    # DENSE SEARCH
    # ----------------------
    def dense_search(self, query: str, k=10):
        q_vec = self.model.encode(query).tolist()

        hits = self.client.query_points(
            collection_name=self.collection,
            query_vector=q_vec,
            limit=k,
        )

        return [self._parse_hit(h) for h in hits]

    # ----------------------
    # MERGE + RERANK
    # ----------------------
    def merge_and_rerank(self, query: str, top_k=5):
        bm25 = self.bm25_search(query, k=20)
        dense = self.dense_search(query, k=20)

        # scoring maps
        all_docs = {}
        for h in bm25:
            all_docs[h["id"]] = {"bm25": h["score"], "dense": 0, "payload": h["payload"]}
        for h in dense:
            if h["id"] not in all_docs:
                all_docs[h["id"]] = {"bm25": 0, "dense": h["score"], "payload": h["payload"]}
            else:
                all_docs[h["id"]]["dense"] = h["score"]

        # normalized score
        results = []
        for doc_id, v in all_docs.items():
            final_score = 0.55 * v["dense"] + 0.45 * v["bm25"]
            results.append({
                "id": doc_id,
                "score": final_score,
                "payload": v["payload"],
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def convert_for_generator(self, merged_results):
        """
        Convert results into the format required by StrictGenerator.
        """
        converted = []
        for item in merged_results:
            payload = item.get("payload", {})
            converted.append({
                "meta": {
                    "source": payload.get("source", "unknown"),
                    "chunk_id": payload.get("chunk_id", 0),
                    "text": payload.get("text", "")
                }
            })
        return converted

# # src/retriever_hybrid.py
# """
# Hybrid retriever using Whoosh (BM25) + Qdrant dense search and CrossEncoder rerank.
# """
# import os
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from whoosh import index, fields, qparser
# from whoosh.analysis import StemmingAnalyzer
# from whoosh.writing import AsyncWriter
# from .vectorstore_qdrant import QdrantStore

# WHOOSH_INDEX_DIR = 'whoosh_index'

# class HybridRetriever:
#     def __init__(self, qdrant_collection='papers', embed_model='all-MiniLM-L6-v2', rerank_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
#         """
#         Initialize dense embedder, Qdrant, and cross-encoder reranker.
#         """
#         self.embed = SentenceTransformer(embed_model)
#         self.qdrant = QdrantStore(collection=qdrant_collection, vector_size=self.embed.get_sentence_embedding_dimension())
#         self.reranker = CrossEncoder(rerank_model)
#         # ensure whoosh index exists
#         if not os.path.exists(WHOOSH_INDEX_DIR):
#             os.makedirs(WHOOSH_INDEX_DIR, exist_ok=True)
#             schema = fields.Schema(id=fields.ID(stored=True, unique=True),
#                                    source=fields.TEXT(stored=True),
#                                    title=fields.TEXT(stored=True),
#                                    text=fields.TEXT(stored=True, analyzer=StemmingAnalyzer()))
#             ix = index.create_in(WHOOSH_INDEX_DIR, schema)
#             ix.close()

#     def index_to_whoosh(self, doc_id: str, source: str, title: str, text: str):
#         """
#         Add a document/chunk into whoosh index (for BM25).
#         """
#         ix = index.open_dir(WHOOSH_INDEX_DIR)
#         writer = AsyncWriter(ix)
#         writer.update_document(id=doc_id, source=source, title=title, text=text)
#         writer.commit()

#     # def dense_search(self, query, k=50):
#     #     """
#     #     Dense ANN search via Qdrant.
#     #     """
#     #     qv = self.embed.encode(query)
#     #     hits = self.qdrant.search(qv, top_k=k)
#     #     results = []
#     #     for h in hits:
#     #         payload = h.payload
#     #         results.append({'id': getattr(h, 'id', None), 'text': payload.get('text'), 'meta': payload, 'score': h.score})
#     #     return results
#     def dense_search(self, query, k=50):
#         qv = self.embed.encode(query)
#         hits = self.qdrant.search(qv, top_k=k)

#         results = []
#         for hit in hits:
#             doc_id, score, payload, vector = hit

#             results.append({
#                 "id": doc_id,
#                 "text": payload.get("text", ""),
#                 "meta": payload,
#                 "score": score
#         })
#         return results

#     def bm25_search(self, query, k=50):
#         """
#         BM25 via whoosh.
#         """
#         ix = index.open_dir(WHOOSH_INDEX_DIR)
#         qp = qparser.MultifieldParser(['text', 'title', 'source'], schema=ix.schema)
#         q = qp.parse(query)
#         results = []
#         with ix.searcher() as s:
#             for r in s.search(q, limit=k):
#                 results.append({'id': r['id'], 'text': r['text'], 'meta': {'source': r['source']}, 'score': r.score})
#         return results

#     def merge_and_rerank(self, query, top_k=5):
#         """
#         Merge BM25 + dense, deduplicate, then rerank with cross-encoder.
#         Returns top_k candidates (each candidate has 'meta' with payload).
#         """
#         dense = self.dense_search(query, k=50)
#         bm25 = self.bm25_search(query, k=50)
#         candidates = []
#         seen = set()
#         # Merge preserving uniqueness
#         for c in dense + bm25:
#             key = c.get('id') or (c['meta'].get('source') + '_' + str(hash(c['text'])))
#             if key in seen:
#                 continue
#             seen.add(key)
#             candidates.append(c)
#         # rerank with CrossEncoder
#         pairs = [[query, c['text'][:512]] for c in candidates]
#         scores = self.reranker.predict(pairs) if pairs else []
#         for c, s in zip(candidates, scores):
#             c['score'] = float(s)
#         candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
#         return candidates[:top_k]