import os
import shutil
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.knowledge.ingest_books import ingest_all_books

router = APIRouter()

BOOKS_DIR = Path("data/books")

@router.post("/upload")
async def upload_book(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # 1. Clear existing books
        if BOOKS_DIR.exists():
            for p in BOOKS_DIR.iterdir():
                if p.is_file():
                    p.unlink()
        else:
            BOOKS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 2. Save the uploaded file
        file_path = BOOKS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Trigger ingestion
        print("Ingesting new book...")
        ingest_all_books()
        
        return {"message": f"Successfully uploaded and ingested {file.filename}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
