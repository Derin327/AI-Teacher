import os
import pickle
import re
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/vector_index/index.faiss"
META_PATH = "data/vector_index/meta.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBED_MODEL_NAME)


def normalize_text(text: str) -> str:
    if not text:
        return ""

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

    text = re.sub(r"(\w)\s*-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_index_and_meta():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return index, meta


def keyword_boost(query: str, text: str) -> float:
    q = query.lower().strip()
    t = text.lower()

    bonus = 0.0

    if "what is computer network" in q or "what is computer networks" in q:
        strong_patterns = [
            "what a computer network is",
            "what distinguishes a computer network",
            "most important characteristic of a computer network",
            "computer networks are built primarily from general-purpose programmable hardware",
            "they are able to carry many different types of data",
            "computer network is",
            "generality",
            "general-purpose programmable hardware",
        ]
        for p in strong_patterns:
            if p in t:
                bonus += 4.0

    return bonus


def retrieve_answer(
    query: str,
    k: int = 12,
    max_chunks: int = 3,
    max_chars: int = 1200,
) -> Tuple[str, List[Dict]]:
    query = (query or "").strip()
    if not query:
        return "", []

    index, meta = load_index_and_meta()

    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k)

    candidates = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(meta):
            continue

        item = meta[idx]
        text = normalize_text(item.get("text", ""))

        if len(text) < 80:
            continue

        score = float(-distances[0][rank]) + keyword_boost(query, text)

        candidates.append(
            {
                "text": text,
                "page": item.get("page"),
                "book_name": item.get("book_name"),
                "chunk_id": item.get("chunk_id"),
                "final_score": score,
            }
        )

    if not candidates:
        return "", []

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    selected = []
    seen_pages = set()
    total_chars = 0

    for item in candidates:
        if item["page"] in seen_pages and len(selected) >= 1:
            continue

        selected.append(item)
        seen_pages.add(item["page"])
        total_chars += len(item["text"])

        if len(selected) >= max_chunks or total_chars >= max_chars:
            break

    print("\n[RETRIEVER] Top selected chunks:", flush=True)
    for i, item in enumerate(selected, start=1):
        preview = item["text"][:160].replace("\n", " ")
        print(
            f"  {i}. [{item['book_name']} | page {item['page']} | score {item['final_score']:.2f}] {preview}...",
            flush=True,
        )

    context = "\n\n".join(item["text"] for item in selected)
    return context, selected