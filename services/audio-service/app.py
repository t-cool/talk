import os
import io
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from engine import AudioEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("audio-service")

# Global Audio Engine instance
audio_engine: Optional[AudioEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global audio_engine
    whisper_model = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-medium")
    kokoro_voice = os.getenv("KOKORO_VOICE", "jf_alpha")
    device = os.getenv("DEVICE", None) # None autodetects CUDA

    logger.info("Starting Audio Service on GPU 1...")
    audio_engine = AudioEngine(
        whisper_model_id=whisper_model,
        default_voice=kokoro_voice,
        device=device,
    )
    yield
    logger.info("Shutting down Audio Service...")

app = FastAPI(
    title="Realtime Voice Audio Service (GPU 1)",
    description="OpenAI-compatible STT (faster-whisper) and TTS (Kokoro-82M) for LiveKit Agents",
    version="1.0.0",
    lifespan=lifespan,
)

# ----------------- Models -----------------
class SpeechRequest(BaseModel):
    model: str = "hexgrad/Kokoro-82M"
    input: str
    voice: Optional[str] = "jf_alpha"
    response_format: Optional[str] = "wav" # "wav" or "mp3"
    speed: Optional[float] = 1.0

# ----------------- Endpoints -----------------

@app.get("/health")
async def health():
    if not audio_engine:
        raise HTTPException(status_code=503, detail="Audio Engine is initializing")
    stats = audio_engine.get_stats()
    return JSONResponse(content={"status": "healthy", **stats})

@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": audio_engine.whisper_model_id if audio_engine else "faster-whisper",
                    "object": "model",
                    "type": "stt",
                },
                {
                    "id": "hexgrad/Kokoro-82M",
                    "object": "model",
                    "type": "tts",
                },
            ],
        }
    )

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: Optional[str] = Form("Systran/faster-whisper-medium"),
    language: Optional[str] = Form("ja"),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    response_format: Optional[str] = Form("json"),
):
    """
    OpenAI-compatible 音声文字起こしエンドポイント (STT)
    """
    if not audio_engine:
        raise HTTPException(status_code=503, detail="Audio Engine not ready")

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            return JSONResponse(content={"text": ""})

        text = await audio_engine.transcribe(
            audio_bytes,
            language=language or "ja",
            temperature=temperature or 0.0,
        )
        return JSONResponse(content={"text": text})
    except Exception as e:
        logger.error(f"Error in transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    OpenAI-compatible 音声合成エンドポイント (TTS)
    """
    if not audio_engine:
        raise HTTPException(status_code=503, detail="Audio Engine not ready")

    try:
        fmt = (request.response_format or "wav").lower()
        if fmt not in ["wav", "mp3"]:
            fmt = "wav"

        audio_bytes = await audio_engine.synthesize(
            text=request.input,
            voice=request.voice,
            speed=request.speed or 1.0,
            output_format=fmt,
        )

        media_type = "audio/wav" if fmt == "wav" else "audio/mpeg"
        return Response(content=audio_bytes, media_type=media_type)
    except Exception as e:
        logger.error(f"Error in speech synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
