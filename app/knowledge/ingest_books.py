import os
from pathlib import Path
from pypdf import PdfReader

from app.knowledge.embeddings import embed
from app.knowledge.vector_store import save_vector_store

BOOKS_DIR = Path("data/books")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x0c", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 700, overlap: int = 120):
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if len(chunk) > 80:
            chunks.append(chunk)

        if end == text_len:
            break

        start = max(0, end - overlap)

    return chunks


def load_pdf_chunks(pdf_path: str):
    reader = PdfReader(pdf_path)
    all_chunks = []
    book_name = Path(pdf_path).name

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()

        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        page_chunks = split_into_chunks(cleaned, chunk_size=700, overlap=120)

        for chunk in page_chunks:
            all_chunks.append({
                "book_name": book_name,
                "page": page_num,
                "text": chunk,
            })

    return all_chunks


def ingest_all_books():
    if not BOOKS_DIR.exists():
        raise FileNotFoundError(f"Books directory not found: {BOOKS_DIR}")

    pdf_files = sorted([p for p in BOOKS_DIR.iterdir() if p.suffix.lower() == ".pdf"])

    if not pdf_files:
        raise FileNotFoundError("No PDF files found inside data/books/")

    all_metadata = []

    print("Found books:")
    for pdf in pdf_files:
        print(f" - {pdf.name}")

    for pdf in pdf_files:
        print(f"\nReading: {pdf.name}")
        chunks = load_pdf_chunks(str(pdf))
        print(f"Extracted {len(chunks)} chunks from {pdf.name}")
        all_metadata.extend(chunks)

    if not all_metadata:
        raise ValueError("No usable chunks extracted from the books.")

    texts = [item["text"] for item in all_metadata]

    print("\nCreating embeddings...")
    vectors = embed(texts)

    print("Saving FAISS index + metadata...")
    save_vector_store(vectors, all_metadata)

    print(f"Done! Stored {len(all_metadata)} chunks from {len(pdf_files)} books.")


if __name__ == "__main__":
    ingest_all_books()