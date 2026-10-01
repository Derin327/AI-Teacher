import os
import pickle
import re
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_PATH = "data/vector_index/index.faiss"
META_PATH = "data/vector_index/meta.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBED_MODEL_NAME)

# Cache for BM25 to avoid rebuilding on every query
_cached_meta_len = -1
_bm25_instance = None
_cached_corpus = None

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
        "": "",
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


def get_bm25_instance(meta):
    global _cached_meta_len, _bm25_instance, _cached_corpus
    
    # Rebuild BM25 if meta size changes (e.g., new book uploaded)
    if _bm25_instance is None or len(meta) != _cached_meta_len:
        _cached_corpus = [normalize_text(item.get("text", "")) for item in meta]
        tokenized_corpus = [doc.lower().split() for doc in _cached_corpus]
        _bm25_instance = BM25Okapi(tokenized_corpus)
        _cached_meta_len = len(meta)
        
    return _bm25_instance


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
    
    if len(meta) == 0:
        return "", []

    # 1. FAISS Vector Search
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")
    distances, vector_indices = index.search(query_embedding, k)
    
    vector_rank_map = {}
    for rank, idx in enumerate(vector_indices[0]):
        if 0 <= idx < len(meta):
            vector_rank_map[idx] = rank + 1

    # 2. BM25 Keyword Search
    bm25 = get_bm25_instance(meta)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top k BM25 indices
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]
    
    bm25_rank_map = {}
    for rank, idx in enumerate(bm25_top_indices):
        if bm25_scores[idx] > 0: # Only count if there's an actual keyword match
            bm25_rank_map[idx] = rank + 1

    # 3. Reciprocal Rank Fusion (RRF)
    all_candidate_indices = set(vector_rank_map.keys()).union(set(bm25_rank_map.keys()))
    
    candidates = []
    rrf_k = 60 # Standard constant for RRF
    
    for idx in all_candidate_indices:
        item = meta[idx]
        text = normalize_text(item.get("text", ""))
        
        if len(text) < 80:
            continue
            
        v_rank = vector_rank_map.get(idx, 1000) # Penalize if missing
        b_rank = bm25_rank_map.get(idx, 1000)
        
        # RRF Score formula
        rrf_score = 0.0
        if v_rank != 1000:
            rrf_score += 1.0 / (rrf_k + v_rank)
        if b_rank != 1000:
            rrf_score += 1.0 / (rrf_k + b_rank)

        candidates.append(
            {
                "text": text,
                "page": item.get("page"),
                "book_name": item.get("book_name"),
                "chunk_id": item.get("chunk_id"),
                "final_score": rrf_score,
                "v_rank": v_rank,
                "b_rank": b_rank
            }
        )

    if not candidates:
        return "", []

    # Sort by RRF score descending
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # 4. Selection based on max constraints
    selected = []
    seen_pages = set()
    total_chars = 0

    for item in candidates:
        # Penalize pulling multiple chunks from the exact same page unless needed
        if item["page"] in seen_pages and len(selected) >= 1:
            continue

        selected.append(item)
        seen_pages.add(item["page"])
        total_chars += len(item["text"])

        if len(selected) >= max_chunks or total_chars >= max_chars:
            break

    print("\n[RETRIEVER] Top selected chunks (Hybrid RRF):", flush=True)
    for i, item in enumerate(selected, start=1):
        preview = item["text"][:120].replace("\n", " ")
        v_str = f"V:{item['v_rank']}" if item['v_rank'] != 1000 else "V:-"
        b_str = f"B:{item['b_rank']}" if item['b_rank'] != 1000 else "B:-"
        
        print(
            f"  {i}. [{item['book_name']} | p.{item['page']} | RRF:{item['final_score']:.4f} | {v_str} | {b_str}] {preview}...",
            flush=True,
        )

    context = "\n\n".join(item["text"] for item in selected)
    return context, selected