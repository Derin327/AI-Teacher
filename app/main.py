import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.chat import router as chat_router
from app.routes.upload import router as upload_router
from fastapi.responses import FileResponse
# Load environment
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]

AUDIO_DIR = (BASE_DIR / os.getenv("AUDIO_DIR", "data/audio")).resolve()
VIDEO_DIR = (BASE_DIR / os.getenv("VIDEO_DIR", "data/video")).resolve()

# Create folders if missing
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI app
app = FastAPI(
    title="AI Teaching Assistant API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

# Static files
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")
app.mount("/video", StaticFiles(directory=str(VIDEO_DIR)), name="video")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))