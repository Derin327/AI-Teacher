# app/routes/chat.py

import os
import re
import time
import uuid
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.schemas.chat import ChatRequest, ChatResponse
from app.db.database import get_db
from app.db.crud import save_chat
from app.media.tts_engine import text_to_speech
from app.media.video_renderer_v2 import render_teaching_video
from app.knowledge.retriever import retrieve_answer
from app.llm.llm_service import refine_answer
from app.core.quiz_generator import build_quiz

load_dotenv()

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
VIDEO_DIR = (BASE_DIR / os.getenv("VIDEO_DIR", "data/video")).resolve()
AUDIO_DIR = (BASE_DIR / os.getenv("AUDIO_DIR", "data/audio")).resolve()

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHUNKS = 3
MAX_ANSWER_CHARS = 700

TEACHER_AVATAR_PATH = (BASE_DIR / "assets" / "teacher.png").resolve()
USE_AVATAR = TEACHER_AVATAR_PATH.exists()


# =========================
# NEW: QUESTION REWRITE
# =========================
def rewrite_question(question: str, answer: str) -> str:
    if not question:
        return "Let us understand this concept clearly."

    q = question.strip().rstrip("?")

    patterns = [
        f"Let us understand {q.lower()} in a simple way.",
        f"How does {q.lower()} actually work?",
        f"What is the real concept behind {q.lower()}?",
    ]

    return patterns[0]


# =========================
# NEW: CHUNK SPLIT
# =========================
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


# =========================
# EXISTING CLEANERS
# =========================
def clean_chunk_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("■", " ")
    text = re.sub(r"\b\d+(\.\d+)+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_good_context_chunk(text: str) -> bool:
    if not text:
        return False

    t = text.strip().lower()

    if len(t) < 80:
        return False

    if any(x in t for x in ["preface", "foreword", "contents", "index"]):
        return False

    return True


def build_context(results):
    cleaned = []
    total_chars = 0

    for item in results[:MAX_CHUNKS]:
        text = clean_chunk_text(item.get("text", ""))

        if not is_good_context_chunk(text):
            continue

        cleaned.append(text)
        total_chars += len(text)

        if total_chars >= 800:
            break

    if not cleaned:
        for item in results[:MAX_CHUNKS]:
            text = clean_chunk_text(item.get("text", ""))
            if text:
                cleaned.append(text)

    return "\n\n".join(cleaned)


def clean_answer_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()

    if text:
        text = text[0].upper() + text[1:]

    return text


def generate_video_filename() -> str:
    return f"lesson_{uuid.uuid4().hex[:8]}.mp4"


def generate_audio_url(audio_filename: str) -> str:
    return f"/audio/{audio_filename}"


def generate_video_url(video_filename: str) -> str:
    return f"/video/{video_filename}"


# =========================
# API
# =========================
@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    start = time.perf_counter()

    try:
        print("\n===== CHAT REQUEST =====", flush=True)

        user_text = (payload.message or payload.query or "").strip()
        if not user_text:
            raise HTTPException(status_code=422, detail="Empty input")

        print("Question:", user_text, flush=True)

        # STEP 1: RETRIEVE
        print("\n[STEP 1] Retrieving...", flush=True)
        _, results = retrieve_answer(
            user_text,
            k=12,
            max_chunks=MAX_CHUNKS,
            max_chars=1200,
        )

        # STEP 2: CONTEXT
        print("[STEP 2] Building context...", flush=True)
        context = build_context(results)

        if not context:
            raise HTTPException(status_code=500, detail="No valid context")

        # STEP 3: LLM
        print("[STEP 3] LLM...", flush=True)
        answer = refine_answer(user_text, context)

        # STEP 4: CLEAN
        print("[STEP 4] Cleaning...", flush=True)
        answer = clean_answer_text(answer)

        if not answer:
            answer = "I could not generate a clear explanation from the uploaded books."

        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0].strip()
            if answer and answer[-1] not in ".!?":
                answer += "."

        print("Answer:", answer, flush=True)

        # =========================
        # NEW: TEACHER STYLE
        # =========================
        teaching_question = rewrite_question(user_text, answer)
        chunks = split_into_chunks(answer)

        final_answer_text = " ".join([c["text"] for c in chunks])

        print("Teaching Question:", teaching_question, flush=True)
        print("Chunks:", chunks, flush=True)

        # STEP 5: TTS
        print("[STEP 5] TTS...", flush=True)
        tts = text_to_speech(final_answer_text)

        if not isinstance(tts, dict):
            raise HTTPException(status_code=500, detail="Invalid TTS response format")

        audio_filename = tts.get("audio_filename")
        audio_path = tts.get("audio_path")

        if not audio_filename or not audio_path:
            raise HTTPException(status_code=500, detail="TTS did not return valid audio data")

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail=f"Audio file not found: {audio_path}")

        # STEP 6: VIDEO (DISABLED FOR SPEED)
        # print("[STEP 6] Video render...", flush=True)
        # video_filename = generate_video_filename()
        # video_path = str(VIDEO_DIR / video_filename)
        #
        # render_teaching_video(
        #     question_text=teaching_question,
        #     answer_text=final_answer_text,
        #     output_path=video_path,
        #     full_audio_path=audio_path,
        #     avatar_path=str(TEACHER_AVATAR_PATH) if USE_AVATAR else None,
        # )
        #
        # if not os.path.exists(video_path):
        #     raise HTTPException(status_code=500, detail=f"Video file not found: {video_path}")

        # STEP 7: QUIZ
        print("[STEP 7] Build quiz...", flush=True)
        quiz = build_quiz(user_text, answer)

        # STEP 8: DB SAVE
        print("[STEP 8] Save DB...", flush=True)
        saved = save_chat(
            db=db,
            question=user_text,
            answer=answer,
            audio_filename=audio_filename,
            video_filename=None,
        )

        total_time = time.perf_counter() - start
        print(f"\nDONE in {total_time:.2f}s", flush=True)

        return ChatResponse(
            id=saved.id,
            question=teaching_question,
            answer=answer,
            audio_url=generate_audio_url(audio_filename),
            video_url=None,
            quiz=quiz,
        )

    except HTTPException:
        raise
    except Exception as e:
        print("\nERROR:", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))