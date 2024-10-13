from uagents import Agent, Context
####### tts_package
import torch
from TTS.api import TTS
TTS_MODEL =  TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
driveI_TTS = Agent(name="driveI_TTS", seed="driveI_TTS recovery phrase")

@driveI_TTS.on_interval(period=1.0)
async def tts(ctx: Context ):
    output_wav_path = "output_divya.wav"
    text_input = "Hello World, I am Fetch AI agent"
    ref_audio_path = "/home/suryasss/Hack_dearborn_3/WhatsApp Ptt 2024-10-12 at 3.11.05 PM.wav"
    TTS_MODEL.tts_to_file(text=text_input, speaker_wav=ref_audio_path, language="en", file_path=output_wav_path)
    ctx.logger.info(f'hello, the audio is {output_wav_path} for {text_input}')
    return output_wav_path
    

if __name__ == "__main__":
    driveI_TTS.run()