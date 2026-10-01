from sqlalchemy import text
from app.db.database import engine

with engine.connect() as conn:
    print("✅ PostgreSQL Connected Successfully!")

    conn.execute(text("""
        ALTER TABLE chat_history
        ADD COLUMN IF NOT EXISTS video_filename VARCHAR;
    """))

    conn.commit()

    print("✅ video_filename column added successfully")