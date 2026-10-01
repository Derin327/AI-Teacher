import os
import re
import pickle
from typing import List, Dict

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BOOKS_DIR = "data/books"
VECTOR_DIR = "data/vector_index"
INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
META_PATH = os.path.join(VECTOR_DIR, "meta.pkl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Skip front matter for this book. Real content starts around Foundation.
SKIP_PAGES_BY_BOOK = {
    "computer networks.pdf": 29,
}

MIN_SENTENCE_LEN = 20
MIN_CHUNK_CHARS = 220
MAX_CHUNK_CHARS = 700
TARGET_SENTENCES_PER_CHUNK = 4
OVERLAP_SENTENCES = 1


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x0c", " ")
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")

    # Fix common PDF ligatures and punctuation
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "’": "'",
        "“": '"',
        "”": '"',
        "—": "-",
        "–": "-",
        "■": " ",
        "◮": " ",
        "￾": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Join hyphen-broken words across spaces/newlines: "hard- ware" -> "hardware"
    text = re.sub(r"(\w)\s*-\s+(\w)", r"\1\2", text)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_front_matter(text: str) -> bool:
    t = text.lower()

    bad_patterns = [
        "foreword",
        "preface",
        "contents",
        "acknowledgments",
        "further reading",
        "road map and course use",
        "pedagogy and features",
        "supplemental materials",
        "approach for an area that's as dynamic",
        "the systems approach has several implications",
        "this page intentionally left blank",
    ]
    return any(p in t for p in bad_patterns)


def split_into_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []

    for part in parts:
        s = normalize_text(part)

        if len(s) < MIN_SENTENCE_LEN:
            continue

        # Skip obvious broken sentence starts
        if s and s[0].islower() and not s.lower().startswith(("tcp", "ip", "atm", "dns", "http", "wifi")):
            continue

        sentences.append(s)

    return sentences


def is_good_chunk(text: str) -> bool:
    t = normalize_text(text)
    if len(t) < MIN_CHUNK_CHARS:
        return False

    if looks_like_front_matter(t):
        return False

    # Avoid broken starts like "etwork ..."
    if t and t[0].islower():
        return False

    # Prefer chunks ending like real prose
    if t[-1] not in ".!?":
        return False

    return True


def sentence_window_chunks(sentences: List[str]) -> List[str]:
    chunks = []
    i = 0

    while i < len(sentences):
        current = []
        current_len = 0
        j = i

        while j < len(sentences):
            s = sentences[j]
            trial = " ".join(current + [s]).strip()

            if len(trial) > MAX_CHUNK_CHARS:
                break

            current.append(s)
            current_len = len(trial)

            if len(current) >= TARGET_SENTENCES_PER_CHUNK and current_len >= MIN_CHUNK_CHARS:
                break

            j += 1

        chunk = normalize_text(" ".join(current))

        if is_good_chunk(chunk):
            chunks.append(chunk)

        # move with overlap
        step = max(1, TARGET_SENTENCES_PER_CHUNK - OVERLAP_SENTENCES)
        i += step

    return chunks


def load_pdf_chunks(pdf_path: str) -> List[Dict]:
    pdf_name = os.path.basename(pdf_path)
    skip_before_page = SKIP_PAGES_BY_BOOK.get(pdf_name.lower(), 0)

    reader = PdfReader(pdf_path)
    all_chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        if page_num <= skip_before_page:
            continue

        raw_text = page.extract_text() or ""
        raw_text = normalize_text(raw_text)

        if len(raw_text) < 120:
            continue

        if looks_like_front_matter(raw_text):
            continue

        sentences = split_into_sentences(raw_text)
        if len(sentences) < 2:
            continue

        chunks = sentence_window_chunks(sentences)

        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "page": page_num,
                    "book_name": pdf_name,
                    "chunk_id": f"{pdf_name}-p{page_num}-c{chunk_id}",
                }
            )

    return all_chunks


def build_vector_store():
    os.makedirs(VECTOR_DIR, exist_ok=True)

    pdf_files = [
        os.path.join(BOOKS_DIR, f)
        for f in os.listdir(BOOKS_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {BOOKS_DIR}")

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"[TRAIN] Reading: {os.path.basename(pdf_path)}")
        chunks = load_pdf_chunks(pdf_path)
        print(f"[TRAIN] Clean chunks from {os.path.basename(pdf_path)}: {len(chunks)}")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No clean chunks were created from the PDFs.")

    texts = [item["text"] for item in all_chunks]
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[TRAIN] Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"[TRAIN] Saved FAISS index to: {INDEX_PATH}")
    print(f"[TRAIN] Saved metadata to: {META_PATH}")
    print(f"[TRAIN] Total clean chunks indexed: {len(all_chunks)}")

    # quick preview
    print("\n[TRAIN] Sample chunks:")
    for i, item in enumerate(all_chunks[:5], start=1):
        print(f"{i}. [{item['book_name']} | page {item['page']}] {item['text'][:180]}...")


if __name__ == "__main__":
    build_vector_store()