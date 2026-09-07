import os

# ============================================================
# SSL CERTIFICATE CONFIGURATION
# ============================================================

SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"

os.environ["REQUESTS_CA_BUNDLE"] = SYSTEM_CA
os.environ["SSL_CERT_FILE"] = SYSTEM_CA
os.environ["CURL_CA_BUNDLE"] = SYSTEM_CA

print("Using certificate bundle:", SYSTEM_CA)


# ============================================================
# IMPORTS
# ============================================================

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate normalized embeddings for document chunks.
    """

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """
    Generate a normalized embedding for a user query.
    """

    model = get_embedding_model()

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.astype("float32")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Loading embedding model...")

    model = get_embedding_model()

    test_embedding = embed_query(
        "What are the working hours at BlueOrbit?"
    )

    print("Model loaded successfully.")
    print("Embedding shape:", test_embedding.shape)