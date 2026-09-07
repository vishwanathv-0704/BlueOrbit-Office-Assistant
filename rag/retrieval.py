import json
from pathlib import Path

import faiss

from .embeddings import embed_query


BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "faiss_policy"

INDEX_FILE = INDEX_DIR / "policy.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"


TOP_K = 5
SIMILARITY_THRESHOLD = 0.50


class PolicyRetriever:

  def __init__(
    self,
    index_file: Path = INDEX_FILE,
    chunks_file: Path = CHUNKS_FILE,
  ):
    """Load the persisted FAISS index and metadata."""

    if not index_file.exists():

      raise FileNotFoundError(
        f"FAISS index not found: {index_file}. "
        "Run `python -m rag.vectorstore` first."
      )

    if not chunks_file.exists():

      raise FileNotFoundError(
        f"Chunk metadata not found: {chunks_file}. "
        "Run `python -m rag.vectorstore` first."
      )

    self.index = faiss.read_index(
      str(index_file)
    )

    with open(
      chunks_file,
      "r",
      encoding="utf-8"
    ) as file:

      self.chunks = json.load(file)


  def search(
    self,
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
  ) -> list[dict]:
    """
    Retrieve the most relevant policy chunks.
    """

    query_embedding = embed_query(query)

    scores, indices = self.index.search(
      query_embedding,
      top_k
    )

    results = []

    for score, index_id in zip(
      scores[0],
      indices[0]
    ):

      if index_id == -1:
        continue

      score = float(score)

      if score < threshold:
        continue

      chunk = self.chunks[index_id]

      results.append(
        {
          "score": score,
          "document": chunk["document"],
          "page": chunk["page"],
          "chunk_id": chunk["chunk_id"],
          "text": chunk["text"],
        }
      )

    return results


  def search_policy(
    self,
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
  ) -> dict:
    """
    Main structured RAG interface.

    Returns:
      {
        "context": "...",
        "sources": [...]
      }
    """

    results = self.search(
      query=query,
      top_k=top_k,
      threshold=threshold
    )

    if not results:

      return {
        "context": "",
        "sources": []
      }

    context_parts = []
    sources = []

    for result in results:

      context_parts.append(
        f"[Source: {result['document']} | "
        f"Page: {result['page']}]\n"
        f"{result['text']}"
      )

      sources.append(
        {
          "document": result["document"],
          "page": result["page"],
          "chunk_id": result["chunk_id"],
          "score": result["score"],
        }
      )

    return {
      "context": "\n\n".join(context_parts),
      "sources": sources
    }


# Lazy singleton
_retriever = None


def get_retriever() -> PolicyRetriever:
  """
  Initialize the retriever once and reuse it.
  """

  global _retriever

  if _retriever is None:

    _retriever = PolicyRetriever()

  return _retriever


def search_policy(query: str) -> dict:
  """
  Public RAG function.

  This is the function the agent will call.
  """

  return get_retriever().search_policy(query)

if __name__ == "__main__":

    print("Testing policy retrieval...")

    retriever = get_retriever()

    result = retriever.search_policy(
       "What is the capital of France?"
    )

    print(json.dumps(result, indent=2))