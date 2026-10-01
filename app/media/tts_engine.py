import os
import re
import uuid
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
AUDIO_DIR = (BASE_DIR / os.getenv("AUDIO_DIR", "data/audio")).resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
POLLY_VOICE_ID = os.getenv("POLLY_VOICE_ID", "Joanna")
POLLY_ENGINE = os.getenv("POLLY_ENGINE", "neural")
POLLY_OUTPUT_FORMAT = os.getenv("POLLY_OUTPUT_FORMAT", "mp3")
POLLY_LANGUAGE_CODE = os.getenv("POLLY_LANGUAGE_CODE", "en-US")
POLLY_SAMPLE_RATE = os.getenv("POLLY_SAMPLE_RATE", "24000")

_polly = None


def get_polly_client():
    global _polly
    if _polly is None:
        _polly = boto3.client("polly", region_name=AWS_REGION)
    return _polly


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("•", " ")
    text = text.replace("|", " ")
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def teacher_style_text(text: str) -> str:
    text = clean_text(text)

    if not text:
        return ""

    replacements = {
        " i.e. ": ", that is, ",
        " e.g. ": ", for example, ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace(";", ". ")
    text = text.replace(":", ". ")

    transition_words = [
        "Therefore",
        "However",
        "So",
        "Now",
        "Next",
        "Also",
        "Finally",
        "For example",
        "In conclusion",
        "As a result",
    ]

    for word in transition_words:
        text = re.sub(rf"\b{re.escape(word)}\b(?!,)", f"{word},", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_ssml(text: str) -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"<speak>{escaped}</speak>"


def text_to_speech(
    text: str,
    voice_id: Optional[str] = None,
    engine: Optional[str] = None,
):
    polly = get_polly_client()

    prepared_text = teacher_style_text(text)
    if not prepared_text:
        raise ValueError("Empty text received for TTS generation.")

    selected_voice = voice_id or POLLY_VOICE_ID
    selected_engine = engine or POLLY_ENGINE

    file_ext = "mp3" if POLLY_OUTPUT_FORMAT == "mp3" else POLLY_OUTPUT_FORMAT
    filename = f"{uuid.uuid4()}.{file_ext}"
    output_path = (AUDIO_DIR / filename).resolve()

    ssml_text = build_ssml(prepared_text)

    try:
        response = polly.synthesize_speech(
            Text=ssml_text,
            TextType="ssml",
            OutputFormat=POLLY_OUTPUT_FORMAT,
            VoiceId=selected_voice,
            Engine=selected_engine,
            LanguageCode=POLLY_LANGUAGE_CODE,
            SampleRate=POLLY_SAMPLE_RATE,
        )
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Amazon Polly synthesis failed: {str(e)}") from e

    audio_stream = response.get("AudioStream")
    if audio_stream is None:
        raise RuntimeError("Amazon Polly did not return an audio stream.")

    with open(output_path, "wb") as f:
        f.write(audio_stream.read())

    if not output_path.exists():
        raise FileNotFoundError(f"Audio file was not created: {output_path}")

    return {
        "audio_filename": filename,
        "audio_path": str(output_path),
        "sample_rate": int(POLLY_SAMPLE_RATE),
        "voice_mode": "amazon_polly",
        "speaker": selected_voice,
        "engine": selected_engine,
        "region": AWS_REGION,
    }


# optional backward compatibility
def text_to_audio(text: str):
    result = text_to_speech(text)
    return result["audio_path"]