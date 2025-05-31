import fastapi
import uvicorn
from pydantic import BaseModel
from typing import Optional, Tuple
import torch
from chatterbox.tts import ChatterboxTTS
import io
import soundfile as sf
import sys
from fastapi.responses import Response
from fastapi import HTTPException
import argparse # Import argparse

app = fastapi.FastAPI()

# --- Model Loading and Device Detection ---
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    # Apply patch for MPS if necessary (from Chatterbox example_for_mac.py)
    torch_load_original = torch.load
    def patched_torch_load(*args_load, **kwargs_load):
        if "map_location" not in kwargs_load:
            kwargs_load["map_location"] = torch.device(device)
        return torch_load_original(*args_load, **kwargs_load)
    torch.load = patched_torch_load
print(f"Using device: {device}", file=sys.stderr)

model: Optional[ChatterboxTTS] = None
try:
    model = ChatterboxTTS.from_pretrained(device=device)
    print("ChatterboxTTS model loaded successfully.", file=sys.stderr)
except Exception as e:
    print(f"Error loading ChatterboxTTS model: {e}", file=sys.stderr)
    model = None

class SpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    output_format: Optional[str] = "wav"
    exaggeration: Optional[float] = 0.5
    temperature: Optional[float] = 0.5

async def generate_speech_audio(text: str, ref_voice_path: Optional[str], exaggeration: float, temperature: float) -> Optional[Tuple[bytes, int]]:
    if model is None:
        print("TTS model is not loaded. Cannot generate speech.", file=sys.stderr)
        return None
    try:
        print(f"Synthesizing: '{text}' with voice: {ref_voice_path}, exag: {exaggeration}, temp: {temperature}", file=sys.stderr)
        wav_tensor = model.generate(
            text,
            audio_prompt_path=ref_voice_path,
            exaggeration=exaggeration,
            temperature=temperature,
        )
        audio_data = wav_tensor.squeeze(0).cpu().numpy()
        sample_rate = model.sr
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        print(f"Generated audio for: '{text}'", file=sys.stderr)
        return buffer.getvalue(), sample_rate
    except Exception as e:
        print(f"Error during TTS generation for '{text}': {e}", file=sys.stderr)
        return None

@app.get("/")
async def read_root():
    return {"message": "Chatterbox TTS API"}

@app.post("/speech")
async def api_create_speech(request: SpeechRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="TTS model is not available.")
    if request.output_format.lower() != "wav":
        raise HTTPException(status_code=400, detail=f"Unsupported output format: {request.output_format}. Only 'wav' is currently supported.")
    audio_result = await generate_speech_audio(
        text=request.text,
        ref_voice_path=request.voice,
        exaggeration=request.exaggeration,
        temperature=request.temperature
    )
    if audio_result is None:
        raise HTTPException(status_code=500, detail="Failed to generate speech.")
    audio_bytes, _ = audio_result
    return Response(content=audio_bytes, media_type="audio/wav")

if __name__ == "__main__":
    if model is None:
        print("Exiting: TTS model failed to load.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Chatterbox TTS FastAPI Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the API server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server (default: 8000)"
    )
    # TTS specific parameters like ref-voice, exaggeration, temperature are per-request via SpeechRequest.
    # A --debug flag could be added here if more granular control over logging is needed in the future.

    args = parser.parse_args()

    print(f"Starting server on {args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
