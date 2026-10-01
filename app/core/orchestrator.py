from app.knowledge.retriever import retrieve_context
from app.teaching.script_generator import generate_script
from app.media.tts_engine import text_to_audio
from app.media.slate_engine import create_frame
from app.media.video_renderer_v1 import render_video

def teach_topic(query: str):
    context = retrieve_context(query)
    script = generate_script(context)

    frames = []
    audio_files = []

    for step in script:
        audio = text_to_audio(step["speech"])
        frame = create_frame(step["draw"])
        audio_files.append(audio)
        frames.append(frame)

    video_path = render_video(frames, audio_files[0])
    return video_path
