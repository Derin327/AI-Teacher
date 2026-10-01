# app/schemas/chat.py

from typing import List
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str | None = None
    query: str | None = None


class QuizData(BaseModel):
    question: str
    options: List[str]
    answer_index: int
    correct_explanation: str


class ChatResponse(BaseModel):
    id: int
    question: str
    answer: str
    audio_url: str | None = None
    video_url: str | None = None
    quiz: QuizData | None = None