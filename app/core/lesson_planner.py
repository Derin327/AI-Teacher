# app/core/lesson_planner.py

import re


def split_into_chunks(answer: str):
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    chunks = []

    if not sentences:
        return []

    if len(sentences) <= 2:
        chunks.append({
            "title": "Explanation",
            "text": " ".join(sentences)
        })
        return chunks

    # basic chunking
    chunks.append({
        "title": "Basic Idea",
        "text": sentences[0]
    })

    mid = " ".join(sentences[1:-1])
    if mid:
        chunks.append({
            "title": "How It Works",
            "text": mid
        })

    chunks.append({
        "title": "Conclusion",
        "text": sentences[-1]
    })

    return chunks