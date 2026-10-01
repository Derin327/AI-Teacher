import math


def reveal_words_by_time(full_text: str, t: float, total: float) -> str:
    """
    Reveal text word by word based on time progress.
    """
    if total <= 0:
        return full_text

    words = full_text.split()
    if not words:
        return full_text

    progress = max(0.0, min(1.0, t / total))
    visible_count = max(1, int(len(words) * progress))
    return " ".join(words[:visible_count])


def reveal_word_count(full_text: str, t: float, total: float) -> int:
    if total <= 0:
        return len(full_text.split())

    words = full_text.split()
    if not words:
        return 0

    progress = max(0.0, min(1.0, t / total))
    return max(1, int(len(words) * progress))


def chalk_pulse(t: float) -> float:
    return 0.8 + 0.2 * abs(math.sin(t * 5.0))


def subtle_float(t: float):
    x = math.sin(t * 0.8) * 1.5
    y = math.cos(t * 1.0) * 1.5
    return int(x), int(y)