from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.db.database import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    audio_filename = Column(String, nullable=True)
    video_filename = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)