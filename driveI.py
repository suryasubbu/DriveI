import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper

STS_MODEL = whisper.load_model("turbo")  
# def record(filename: str, duration):
#     print(f"Recording for {duration} seconds...")
    
#     audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float64')
#     sd.wait()  
#     wav.write(filename, 44100, audio)

def sts(path: str):
    result = STS_MODEL.transcribe(path) 
    print(result["text"]) 
    return result["text"]

def llm(text: str):
    #######llm code here########
    model_load = None
    text = None
    return text
def tts(text: str, ref_audio_path: str):
    #######tts code here#########
    model_load = None
    output_wav_path = None
    return output_wav_path

def play(path: str):
    ###simply plays the output tts audio
    return None


def driveI():
    input_path = "/home/suryasss/Hack_dearborn_3/WhatsApp Ptt 2024-10-12 at 3.11.05 PM.wav"
    input_text = sts(input_path)
    output_text = llm(input_text)
    ref_audio_path = "/home/suryasss/Hack_dearborn_3/WhatsApp Ptt 2024-10-12 at 3.11.05 PM.wav"
    output_path = tts(output_text, ref_audio_path)
    play(output_path)

if __name__ == "__main__":
    driveI()

