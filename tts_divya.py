# -*- coding: utf-8 -*-
"""TTS Divya.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vBQqS0ufOroIAQbGq51r1oZur07kq2dF
"""

pip install TTS

!pip install TTS

! git clone https://github.com/coqui-ai/TTS.git

! pip install -r TTS/requirements.txt

import torch
from TTS.api import TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
print(TTS().list_models())
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(
    text="Hello something something something hack hack dearborn",
    speaker_wav="C:/Users/DIVYA TR/OneDrive/Desktop/hackathon/audio.wav",
    language="en",
    file_path="output.wav"
)

from google.colab import files
uploaded = files.upload()  # This will prompt you to upload your file

file_path = "audio.wav"  # Replace with the actual filename you uploaded
tts.tts_to_file(
    text="Hello something something something hack hack dearborn",
    speaker_wav=file_path,
    language="en",
    file_path="output.wav"
)

from google.colab import files
files.download("output.wav")

!ls

!find / -name "output.wav"

from google.colab import files
files.download("/content/output.wav")