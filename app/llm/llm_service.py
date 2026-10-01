import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:0.5b"


def post_clean_llm_answer(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove textbook transitions
    bad_starts = [
        r"^similarly,\s*",
        r"^another kind of\s*",
        r"^this chapter introduces\s*",
        r"^to start us on the road.*?\.\s*",
        r"^as we saw earlier,\s*",
        r"^in this chapter,\s*",
    ]

    for pattern in bad_starts:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def refine_answer(question: str, context: str) -> str:
    prompt = f"""
You are an expert computer science teacher.

STRICT RULES:
1. First sentence must directly answer the question.
2. No textbook style phrases like:
   - "Similarly"
   - "Another kind of"
   - "This chapter introduces"
3. No chapter references.
4. Explain clearly in simple teaching style.
5. Keep it 4 to 6 sentences.
6. No bullet points.

QUESTION:
{question}

CONTEXT:
{context}

Now give a clean explanation.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        result = response.json()
        answer = result.get("response", "").strip()

        return post_clean_llm_answer(answer)

    except Exception as e:
        print("⚠️ Ollama failed, fallback triggered:", e)
        return fallback_answer(context)


# fallback if ollama fails
def fallback_answer(context: str) -> str:
    sentences = context.split(".")
    cleaned = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not cleaned:
        return "I could not generate a clear explanation."

    return ". ".join(cleaned[:4]) + "."