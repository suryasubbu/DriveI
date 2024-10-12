import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper


def record_audio(filename, duration):
    print(f"Recording for {duration} seconds...")
    
    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float64')
    sd.wait()  
    wav.write(filename, 44100, audio)


def transcribe_audio(filename):
    model = whisper.load_model("turbo")  
    result = model.transcribe(filename)  
    return result["text"]


def main():
    
    filename = "/Users/nishaantmadhan/Desktop/Visual Studio/recorded_audio.wav"
    duration = 5  

    
    record_audio(filename, duration)

   
    transcription = transcribe_audio(filename)

    
    with open("/Users/nishaantmadhan/Desktop/Visual Studio/transcription.txt", "w") as text_file:
        text_file.write(transcription)

    print("Transcription saved to transcription.txt")
    
if __name__ == "__main__":
    main()
