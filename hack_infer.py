import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="/home/suryasss/Hack_dearborn_3/WhatsApp Ptt 2024-10-12 at 3.11.05 PM.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="Nishaant is an Idiot and Kaushik is a stupid", speaker_wav="/home/suryasss/Hack_dearborn_3/WhatsApp Ptt 2024-10-12 at 3.11.05 PM.wav", language="en", file_path="output_divya.wav")