from sqlalchemy.orm import Session
from app.db.models import ChatHistory


def save_chat(
    db: Session,
    question: str,
    answer: str,
    audio_filename: str = None,
    video_filename: str = None,
):
    chat = ChatHistory(
        question=question,
        answer=answer,
        audio_filename=audio_filename,
        video_filename=video_filename,
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat