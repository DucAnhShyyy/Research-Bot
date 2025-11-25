# # src/vectorstore_qdrant.py
# """
# Qdrant wrapper with helper methods for upsert and search.
# """
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance
# import os
# import logging

# logger = logging.getLogger(__name__)

# QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
# QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
# COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")

# class QdrantStore:
#     def __init__(self, collection: str = COLLECTION_NAME, host: str = QDRANT_HOST, port: int = QDRANT_PORT, vector_size: int = 384):
#         """
#         Initialize Qdrant client and create collection if needed.
#         """
#         self.collection = collection
#         self.client = QdrantClient(host=host, port=port)
#         self._ensure_collection(vector_size)

#     def _ensure_collection(self, vector_size: int):
#         """
#         Create collection if it does not exist.
#         """
#         try:
#             self.client.get_collection(self.collection)
#         except Exception:
#             logger.info("Creating Qdrant collection: %s", self.collection)
#             self.client.recreate_collection(collection_name=self.collection,
#                                             vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))

#     def upsert_points(self, points):
#         """
#         Upsert points into Qdrant.
#         points: list of dicts {'id': optional, 'vector': list|ndarray, 'payload': dict}
#         """
#         formatted = []
#         for p in points:
#             vec = p['vector'].tolist() if hasattr(p['vector'], 'tolist') else p['vector']
#             formatted.append({
#                 'id': p.get('id'),
#                 'vector': vec,
#                 'payload': p['payload']
#             })
#         self.client.upsert(collection_name=self.collection, points=formatted)

#     def search(self, vector, top_k: int = 10, filter=None):
#         """
#         Query Qdrant returning hits.
#         """
#         vec = vector.tolist() if hasattr(vector, 'tolist') else vector
#         return self.client.search(collection_name=self.collection, query_vector=vec, limit=top_k, query_filter=filter)

# src/vectorstore_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
import logging

logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")

class QdrantStore:
    def __init__(
        self,
        collection: str = COLLECTION_NAME,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        vector_size: int = 384
    ):
        self.collection = collection
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            logger.info("Creating Qdrant collection: %s", self.collection)
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def upsert_points(self, points):
        """
        points: list of dicts {'id': int|str, 'vector': list, 'payload': dict}
        """
        formatted = []
        for p in points:
            vec = p['vector'].tolist() if hasattr(p['vector'], "tolist") else p["vector"]
            formatted.append(
                PointStruct(
                    id=p["id"],            # BẮT BUỘC: ID phải khác None
                    vector=vec,
                    payload=p["payload"]
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=formatted
        )

    def search(self, vector, top_k: int = 10, filter=None):
        vec = vector.tolist() if hasattr(vector, "tolist") else vector
        return self.client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=top_k,
            query_filter=filter
        )