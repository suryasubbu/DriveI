from uagents import Agent, Context
####### tts_package
import whisper

STS_MODEL = whisper.load_model("turbo")
driveI_STT = Agent(name="driveI_STT", seed="driveI_STT recovery phrase")

@driveI_STT.on_interval(period=1.0)
async def STT(ctx: Context ):
    result = STS_MODEL.transcribe("/home/suryasss/Hack_dearborn_3/WhatsApp Ptt 2024-10-12 at 3.11.05 PM.wav") 
    ctx.logger.info(f'hello, the transcribed text is {result["text"]}')
    return result["text"]
    

if __name__ == "__main__":
    driveI_STT.run()