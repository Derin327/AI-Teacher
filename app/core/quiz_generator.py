# app/core/quiz_generator.py

import random
import re
from typing import List, Dict


def _clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _split_sentences(text: str) -> List[str]:
    text = _clean_sentence(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _make_distractors(answer: str) -> List[str]:
    distractors = [
        "It creates a permanent dedicated connection between devices.",
        "It stores the full message in one switch without forwarding packets.",
        "It prevents multiple users from sharing the same network resources.",
        "It sends all data through a single fixed path without splitting it.",
        "It works only for voice calls and not for data transmission.",
        "It blocks packets instead of routing them across the network.",
    ]

    answer_clean = _clean_sentence(answer).lower()
    filtered = []
    for d in distractors:
        if _clean_sentence(d).lower() != answer_clean:
            filtered.append(d)

    random.shuffle(filtered)
    return filtered[:3]


def build_quiz(question: str, answer: str) -> Dict:
    sentences = _split_sentences(answer)

    correct = sentences[0] if sentences else answer
    correct = _clean_sentence(correct)

    if correct and correct[-1] not in ".!?":
        correct += "."

    options = [correct] + _make_distractors(correct)
    random.shuffle(options)

    answer_index = options.index(correct)

    return {
        "question": question.strip() if question else "Choose the correct answer.",
        "options": options,
        "answer_index": answer_index,
        "correct_explanation": answer.strip() if answer else correct,
    }