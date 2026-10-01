import re

def plan_teaching_steps(question: str, answer_text: str):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    steps = []

    steps.append({"type": "heading", "text": question})

    for sentence in sentences:
        steps.append({"type": "write", "text": sentence})

        lower = sentence.lower()
        if any(word in lower for word in ["network", "connected", "router", "server", "database", "cloud"]):
            steps.append({"type": "diagram", "kind": "network_basic"})

        long_words = [w.strip(".,!?") for w in sentence.split() if len(w.strip(".,!?")) > 7]
        if long_words:
            steps.append({"type": "highlight", "words": long_words[:2]})

    steps.append({"type": "quiz"})
    return steps