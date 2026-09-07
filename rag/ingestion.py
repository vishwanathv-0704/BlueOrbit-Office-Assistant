from pathlib import Path
import re

from pypdf import PdfReader


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Normalize extracted PDF text."""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks.

    Each chunk is approximately CHUNK_SIZE characters,
    with CHUNK_OVERLAP characters shared between chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# LOAD POLICY CHUNKS
# ============================================================

def load_policy_chunks(pdf_dir: Path = PDF_DIR) -> list[dict]:
    """
    Load policy PDFs and split them into chunks.

    Page-level metadata is preserved so retrieved
    information can be cited accurately.
    """

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {pdf_dir}"
        )

    chunks = []
    chunk_id = 0

    for pdf_path in pdf_files:

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            raw_text = page.extract_text() or ""

            text = clean_text(raw_text)

            if not text:
                continue

            page_chunks = split_text(text)

            for chunk_text in page_chunks:

                chunks.append(
                    {
                        "document": pdf_path.name,
                        "page": page_number,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )

                chunk_id += 1

    return chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    chunks = load_policy_chunks()

    print(f"PDF directory: {PDF_DIR}")
    print(f"Number of chunks: {len(chunks)}")

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0])