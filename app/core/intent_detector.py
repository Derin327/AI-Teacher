def detect_intent(query: str):
    q = query.lower()
    if "again" in q or "doubt" in q:
        return "doubt"
    if "simple" in q or "easy" in q:
        return "simplify"
    return "teach"
