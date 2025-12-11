# import os
# from typing import List, Dict, Any

# from dotenv import load_dotenv
# from qdrant_client import QdrantClient, models
# from sentence_transformers import SentenceTransformer

# load_dotenv()

# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# def get_embedder() -> SentenceTransformer:
#     # In a larger app you could cache this globally to avoid reloading
#     return SentenceTransformer(EMBEDDING_MODEL_NAME)


# def get_qdrant_client() -> QdrantClient:
#     """
#     Create a Qdrant client using environment variables.

#     Expected vars
#       QDRANT_URL default http://localhost:6333
#       QDRANT_API_KEY optional for local
#     """
#     url = os.getenv("QDRANT_URL", "http://localhost:6333")
#     api_key = os.getenv("QDRANT_API_KEY") or None

#     client = QdrantClient(
#         url=url,
#         api_key=api_key,
#         prefer_grpc=False,
#     )
#     return client


# def recreate_collection(
#     client: QdrantClient,
#     name: str,
#     vector_size: int,
#     distance: models.Distance = models.Distance.COSINE,
# ) -> None:
#     """
#     Drop and recreate a collection with a simple dense vector config.
#     """
#     client.recreate_collection(
#         collection_name=name,
#         vectors_config=models.VectorParams(
#             size=vector_size,
#             distance=distance,
#         ),
#     )


# def upsert_points(
#     client: QdrantClient,
#     collection_name: str,
#     vectors: List[List[float]],
#     payloads: List[Dict[str, Any]],
#     ids: List[int],
# ) -> None:
#     """
#     Upsert a batch of points into a collection.

#     This assumes vectors, payloads, and ids have the same length.
#     """
#     points = [
#         models.PointStruct(
#             id=id_val,
#             vector=vec,
#             payload=payload,
#         )
#         for id_val, vec, payload in zip(ids, vectors, payloads)
#     ]

#     client.upsert(
#         collection_name=collection_name,
#         points=points,
#         wait=True,
#     )
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder() -> SentenceTransformer:
    # In a bigger app we could cache this more smartly
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_qdrant_client() -> QdrantClient:
    """
    Create a Qdrant client using environment variables.

    Expected vars
      QDRANT_URL default http://localhost:6333
      QDRANT_API_KEY optional for local
    """
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None

    client = QdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=False,
    )
    return client


def recreate_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """
    Drop and recreate a collection with a simple dense vector config.
    """
    client.recreate_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    ids: List[int],
) -> None:
    """
    Upsert a batch of points into a collection.

    This assumes vectors, payloads, and ids have the same length.
    """
    points = [
        models.PointStruct(
            id=id_val,
            vector=vec,
            payload=payload,
        )
        for id_val, vec, payload in zip(ids, vectors, payloads)
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
