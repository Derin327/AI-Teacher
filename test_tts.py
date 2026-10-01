import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to(device)
tts.tts_to_file(text="Hello, I am your AI teacher.", file_path="output.wav")
print("Saved output.wav")
