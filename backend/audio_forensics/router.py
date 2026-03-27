from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from shared.schemas import AudioForensicsResponse
import logging

router = APIRouter()
logger = logging.getLogger("truthscan.audio.router")

@router.post("/analyze", response_model=AudioForensicsResponse)
async def analyze_audio(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/") and not any(file.filename.endswith(ext) for ext in [".wav", ".mp3", ".aac", ".ogg"]):
         raise HTTPException(status_code=400, detail="Invalid audio file format. Supported: WAV, MP3, AAC, OGG.")
    try:
        content = await file.read()
        analyzer = request.app.state.audio_analyzer
        result = await analyzer.analyze(content, file.filename)
        return result
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
