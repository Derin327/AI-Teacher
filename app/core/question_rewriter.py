# app/core/question_rewriter.py

def rewrite_question(question: str, answer: str) -> str:
    if not question:
        return "Let us understand this concept clearly."

    q = question.strip().rstrip("?")

    # simple intelligent rewrites
    patterns = [
        f"Let us understand {q.lower()} in a simple way.",
        f"How does {q.lower()} actually work?",
        f"What is the real concept behind {q.lower()}?",
    ]

    return patterns[0]