import gradio as gr
import whisper
import torch
import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
STS_MODEL = whisper.load_model("turbo")
TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(checkpoints), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator

ckpt_dir = "/home/suryasss/Hack_dearborn_3/pyllama/pyllama_data/7B"
tokenizer_path = "/home/suryasss/Hack_dearborn_3/pyllama/pyllama_data/tokenizer.model"

GENERATOR = load(
    ckpt_dir, tokenizer_path, local_rank=0, world_size=1, max_seq_len=1024, max_batch_size=1
)

def sts(path: str):
    result = STS_MODEL.transcribe(path) 
    print(result["text"]) 
    return result["text"]

def llm(text: str, temperature: float = 0.8, top_p: float = 0.95):
    prompts = [text]
    print("Prompt:", prompts)
    results = GENERATOR.generate(
        prompts, max_gen_len=50, temperature=temperature, top_p=top_p
    )
    res = []
    print(results)
    for result in results[0:1]:
        res = result.strip()
    return res

def tts(text_input: str, ref_audio_path: str):
    output_wav_path = "output_divya.wav"
    TTS_MODEL.tts_to_file(text=text_input, speaker_wav=ref_audio_path, language="en", file_path=output_wav_path)
    return output_wav_path

def driveI(image_path,input_audio_path: str, ref_audio_path: str):
    input_text = sts(input_audio_path)
    output_text = llm(input_text)
    output_audio_path = tts(output_text, ref_audio_path)
    return output_audio_path, output_text



def display_interface():
    with gr.Blocks(gr.themes.Soft(),fill_height=True) as demo:
        gr.Markdown("![](file/logo.png)")
        gr.Markdown("# Intelligent Co-Pilot and Driver == Quality travel")
        # gr.Markdown("This application takes an input audio file and a reference audio file. The input audio is transcribed to text, processed with a language model, and then converted to speech using the reference audio for voice characteristics. Use the interface below to get started.")
        gr.Interface(
        fn=driveI,
        inputs=[
            gr.Image(type = "filepath"),
            gr.Audio(type="filepath", label="Input Audio"),
            gr.Audio(type="filepath", label="Reference Audio")
        ],
        outputs=[
            gr.Audio(type="filepath", label="Output Audio"),
            gr.Textbox(label="Generated Text")
        ],
        title="",
        description="This application takes an input audio file and a reference audio file. \
    The input audio is transcribed to text, processed with a language model, and then converted to speech \
    using the reference audio for voice characteristics."
    )
    demo.launch()

if __name__ == "__main__":
    display_interface()