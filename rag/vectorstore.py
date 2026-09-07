import json
from pathlib import Path

import faiss

from .ingestion import load_policy_chunks
from .embeddings import embed_texts


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "faiss_policy"

INDEX_FILE = INDEX_DIR / "policy.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store():

    print("Loading policy chunks...")

    chunks = load_policy_chunks()

    if not chunks:
        raise ValueError("No chunks found from the PDF files.")

    print(f"Number of chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = embed_texts(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # Create FAISS index
    # Since embeddings are normalized,
    # Inner Product = Cosine Similarity

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    print("Adding embeddings to FAISS index...")

    index.add(embeddings)

    print(f"Number of vectors in FAISS: {index.ntotal}")

    # Create output directory
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save chunk metadata
    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nVector store created successfully!")

    print(f"FAISS index: {INDEX_FILE}")

    print(f"Chunk metadata: {CHUNKS_FILE}")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    build_vector_store()