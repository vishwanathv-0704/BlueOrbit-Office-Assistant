# ============================================================
# BLUE ORBIT OFFICE ASSISTANT
# Policy RAG Retrieval using FAISS
# ============================================================

import json
from pathlib import Path

import faiss

from .embeddings import embed_query


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "faiss_policy"

INDEX_FILE = INDEX_DIR / "policy.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

TOP_K = 5

# Minimum similarity score required for a chunk
# to be considered relevant.
SIMILARITY_THRESHOLD = 0.50


# ============================================================
# POLICY RETRIEVER
# ============================================================

class PolicyRetriever:
    """
    Loads the persisted FAISS policy index and retrieves
    relevant company-policy chunks for a user query.
    """

    def __init__(
        self,
        index_file: Path = INDEX_FILE,
        chunks_file: Path = CHUNKS_FILE,
    ):
        """
        Load the persisted FAISS index and chunk metadata.
        """

        # ----------------------------------------------------
        # Check FAISS index
        # ----------------------------------------------------

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_file}. "
                "Run `python -m rag.vectorstore` first."
            )

        # ----------------------------------------------------
        # Check chunk metadata
        # ----------------------------------------------------

        if not chunks_file.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {chunks_file}. "
                "Run `python -m rag.vectorstore` first."
            )

        # ----------------------------------------------------
        # Load FAISS index
        # ----------------------------------------------------

        self.index = faiss.read_index(
            str(index_file)
        )

        # ----------------------------------------------------
        # Load chunks
        # ----------------------------------------------------

        with open(
            chunks_file,
            "r",
            encoding="utf-8",
        ) as file:

            self.chunks = json.load(file)

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if not isinstance(self.chunks, list):
            raise ValueError(
                "chunks.json must contain a list of chunks."
            )

        if self.index.ntotal == 0:
            raise ValueError(
                "FAISS index contains no vectors."
            )

        if self.index.ntotal > len(self.chunks):
            raise ValueError(
                "FAISS index contains more vectors than "
                "the number of chunk metadata entries."
            )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Retrieve the most relevant policy chunks.

        Parameters
        ----------
        query:
            User's policy question.

        top_k:
            Maximum number of chunks to retrieve.

        threshold:
            Minimum similarity score.

        Returns
        -------
        list[dict]
            Relevant policy chunks with scores and metadata.
        """

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not query or not str(query).strip():
            return []

        query = str(query).strip()

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        query_embedding = embed_query(query)

        # ----------------------------------------------------
        # Search FAISS
        # ----------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        # ----------------------------------------------------
        # Process search results
        # ----------------------------------------------------

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):

            # FAISS can return -1 when no result exists.
            if index_id == -1:
                continue

            score = float(score)

            # Ignore low-confidence matches.
            if score < threshold:
                continue

            # Safety check for metadata.
            if index_id >= len(self.chunks):
                continue

            chunk = self.chunks[index_id]

            # Make sure the chunk is a dictionary.
            if not isinstance(chunk, dict):
                continue

            results.append(
                {
                    "score": score,
                    "document": chunk.get(
                        "document",
                        "Unknown document",
                    ),
                    "page": chunk.get(
                        "page",
                        "Unknown",
                    ),
                    "chunk_id": chunk.get(
                        "chunk_id",
                        index_id,
                    ),
                    "text": chunk.get(
                        "text",
                        "",
                    ),
                }
            )

        return results


    # ========================================================
    # STRUCTURED POLICY SEARCH
    # ========================================================

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

        The employee agent uses the "context" field to
        generate the final policy answer.
        """

        results = self.search(
            query=query,
            top_k=top_k,
            threshold=threshold,
        )

        # ----------------------------------------------------
        # No relevant policy found
        # ----------------------------------------------------

        if not results:
            return {
                "context": "",
                "sources": [],
            }

        context_parts = []
        sources = []

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        for result in results:

            document = result.get(
                "document",
                "Unknown document",
            )

            page = result.get(
                "page",
                "Unknown",
            )

            chunk_id = result.get(
                "chunk_id",
                "",
            )

            text = result.get(
                "text",
                "",
            )

            # Ignore empty chunks.
            if not str(text).strip():
                continue

            context_parts.append(
                f"[Source: {document} | Page: {page}]\n"
                f"{text}"
            )

            sources.append(
                {
                    "document": document,
                    "page": page,
                    "chunk_id": chunk_id,
                    "score": result["score"],
                }
            )

        # ----------------------------------------------------
        # If all chunks were empty
        # ----------------------------------------------------

        if not context_parts:
            return {
                "context": "",
                "sources": [],
            }

        # ----------------------------------------------------
        # Final structured response
        # ----------------------------------------------------

        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
        }


# ============================================================
# LAZY SINGLETON
# ============================================================

_retriever = None


def get_retriever() -> PolicyRetriever:
    """
    Initialize the PolicyRetriever only when required.

    This avoids loading the FAISS index when the module
    is merely imported.
    """

    global _retriever

    if _retriever is None:
        _retriever = PolicyRetriever()

    return _retriever


# ============================================================
# PUBLIC RAG FUNCTION
# ============================================================

def retrieve_policy(
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Public RAG function used by employee_agent.py.

    Example:

        result = retrieve_policy(
            "What is the work from home policy?"
        )

    Returns:

        {
            "context": "...",
            "sources": [...]
        }
    """

    return get_retriever().search_policy(
        query=query,
        top_k=top_k,
        threshold=threshold,
    )


# ============================================================
# BACKWARD-COMPATIBLE FUNCTION
# ============================================================

def search_policy(
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Backward-compatible alias.

    Older code may call:

        search_policy(...)

    Newer code should use:

        retrieve_policy(...)
    """

    return retrieve_policy(
        query=query,
        top_k=top_k,
        threshold=threshold,
    )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BLUE ORBIT POLICY RAG - RETRIEVAL TEST")
    print("=" * 70)

    try:

        print("\nLoading FAISS policy retriever...")

        retriever = get_retriever()

        print("FAISS index loaded successfully.")
        print(f"Vectors: {retriever.index.ntotal}")
        print(f"Chunks: {len(retriever.chunks)}")

        # ----------------------------------------------------
        # Test 1
        # ----------------------------------------------------

        question = "What is the work from home policy?"

        print("\n" + "-" * 70)
        print(f"QUERY: {question}")

        result = retrieve_policy(question)

        print("\nRESULT:")

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        # ----------------------------------------------------
        # Test 2
        # ----------------------------------------------------

        question = "What is the leave policy?"

        print("\n" + "-" * 70)
        print(f"QUERY: {question}")

        result = retrieve_policy(question)

        print("\nRESULT:")

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        # ----------------------------------------------------
        # Test 3 - unrelated question
        # ----------------------------------------------------

        question = "What is the capital of France?"

        print("\n" + "-" * 70)
        print(f"QUERY: {question}")

        result = retrieve_policy(question)

        print("\nRESULT:")

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 70)
        print("RAG RETRIEVAL TEST COMPLETE")
        print("=" * 70)

    except Exception as exc:

        print("\n" + "=" * 70)
        print("RAG RETRIEVAL TEST FAILED")
        print("=" * 70)

        print(f"\nError: {exc}")