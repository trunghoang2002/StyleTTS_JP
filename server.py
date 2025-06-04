from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import yaml
import argparse
from scipy.io.wavfile import write

import os
from dotenv import load_dotenv

load_dotenv()
print("using devices: ", os.getenv("CUDA_VISIBLE_DEVICES"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES")

from inference import load_styletts, load_hifigan, TextCleaner, compute_style, synthesize_speech

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Load configuration
model_config_path = "./Models/JSUT/config.yml"
config = yaml.safe_load(open(model_config_path))
print("Loaded config file...")

speed = 0.9

# Load models
model_path = "./Models/JSUT/epoch_2nd_00100.pth"
model = load_styletts(config, model_path, device)
generator = load_hifigan(device)
textcleaner = TextCleaner()
print("Loaded models...")

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def tts(request: TTSRequest):
    try:
        converted_samples = synthesize_speech(
            request.text, reference_embeddings, speed, model, generator, textcleaner, device
        )
        audios = []
        for key, audio in converted_samples.items():
            # Convert numpy array to list for JSON serialization
            audios.append(audio.tolist())
        return {"audios": audios}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleTTS API")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7000, help="Server port (default: 7000)")
    parser.add_argument("--output_dir", default="../DINet", help="Output directory for audio files (default: ../DINet)")
    parser.add_argument("--reference_audio", default="ref_audio.wav", help="Path to the reference audio")
    args = parser.parse_args()

    # Compute style embeddings from reference samples
    reference_samples = [
        args.reference_audio,
    ]
    ref_dicts = {path.split('/')[-1].replace('.wav', ''): path for path in reference_samples}
    reference_embeddings = compute_style(model, ref_dicts, device)
    # Chạy FastAPI server
    import uvicorn
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)