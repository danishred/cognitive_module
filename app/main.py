import os
from pathlib import Path
from typing import List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.config import Settings, get_settings
from app.schemas import HealthResponse, TranscribeUrlRequest, TranscriptionResponse
from app.services.deepgram_service import DeepgramService, get_deepgram_service

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_AUDIO_PATH = BASE_DIR.parent / "cost_of_care.mp3"

app = FastAPI(
    title="Cognitive Audio STT API",
    description=(
        "High-performance Speech-to-Text API powered by Deepgram with native "
        "support for English-Hindi (Hinglish) code-switching, speaker diarization, "
        "and audio file transcription."
    ),
    version="1.0.0",
)

# Enable CORS for cross-origin API clients and web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def serve_ui():
    """Serve the single-page interactive transcription Web UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="UI template not found",
        )
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint to verify system status and API key configuration."""
    api_key_set = settings.is_api_key_configured
    msg = (
        "Server is running and Deepgram API key is configured."
        if api_key_set
        else "Server is running, but DEEPGRAM_API_KEY is missing or set to placeholder in .env."
    )
    return HealthResponse(
        status="healthy",
        api_key_configured=api_key_set,
        default_model=settings.DEFAULT_MODEL,
        default_language=settings.DEFAULT_LANGUAGE,
        message=msg,
    )


@app.get("/sample-audio", tags=["Sample Data"])
async def get_sample_audio():
    """Stream the sample audio file (cost_of_care.mp3) for testing and UI playback."""
    if not SAMPLE_AUDIO_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample audio file not found on server",
        )
    return FileResponse(
        path=str(SAMPLE_AUDIO_PATH),
        media_type="audio/mpeg",
        filename="cost_of_care.mp3",
    )


@app.post(
    "/api/v1/transcribe",
    response_model=TranscriptionResponse,
    tags=["Transcription"],
    summary="Transcribe Audio File (Multipart Form)",
)
async def transcribe_audio_file(
    file: UploadFile = File(..., description="Audio or video file (.mp3, .wav, .m4a, .mp4, .ogg, .flac, .webm, .aac)"),
    model: Optional[str] = Form(None, description="Deepgram model (e.g. 'nova-3', 'nova-3-medical', 'nova-2')"),
    language: Optional[str] = Form(None, description="Language code (use 'multi' for Hinglish, 'en', or 'hi')"),
    diarize: Optional[bool] = Form(None, description="Identify and separate speakers (e.g. Speaker 0, Speaker 1)"),
    smart_format: Optional[bool] = Form(None, description="Format numbers, dates, and punctuation cleanly"),
    punctuate: Optional[bool] = Form(None, description="Add punctuation"),
    keyterms: Optional[str] = Form(None, description="Comma-separated custom keywords/terms to boost"),
    include_raw: bool = Form(False, description="Include raw Deepgram JSON payload in the response"),
    include_words: bool = Form(False, description="Include word-level timestamps in the response"),
    service: DeepgramService = Depends(get_deepgram_service),
):
    """Receive an audio or video file via multipart/form-data and transcribe it using Deepgram.
    
    Default configuration uses `model=nova-3` and `language=multi` to recognize both
    English and Hindi seamlessly within the same conversation (Hinglish code-switching).
    If a video container (like MP4) is uploaded, the audio stream is automatically extracted.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename provided")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    keyterm_list = [k.strip() for k in keyterms.split(",") if k.strip()] if keyterms else None

    return await service.transcribe_file(
        audio_bytes=audio_bytes,
        filename=file.filename or "",
        model=model,
        language=language,
        diarize=diarize,
        smart_format=smart_format,
        punctuate=punctuate,
        keyterms=keyterm_list,
        include_raw=include_raw,
        include_words=include_words,
    )


@app.post(
    "/api/v1/transcribe/raw",
    response_model=TranscriptionResponse,
    tags=["Transcription"],
    summary="Transcribe Raw Audio Binary Stream",
)
async def transcribe_raw_audio(
    request: Request,
    filename: Optional[str] = Query("", description="Optional filename hint for format detection"),
    model: Optional[str] = Query(None, description="Deepgram model (default: nova-3)"),
    language: Optional[str] = Query(None, description="Language (default: multi for Hinglish)"),
    diarize: Optional[bool] = Query(None, description="Enable speaker diarization"),
    smart_format: Optional[bool] = Query(None, description="Enable smart formatting"),
    punctuate: Optional[bool] = Query(None, description="Enable punctuation"),
    keyterms: Optional[str] = Query(None, description="Comma-separated keywords"),
    include_raw: bool = Query(False, description="Include raw response"),
    include_words: bool = Query(False, description="Include word timestamps"),
    service: DeepgramService = Depends(get_deepgram_service),
):
    """Accepts raw audio binary bytes in the HTTP request body.
    
    Useful for clients and microservices streaming binary audio buffers directly.
    """
    audio_bytes = await request.body()
    if not audio_bytes or len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body contains no audio binary data",
        )

    keyterm_list = [k.strip() for k in keyterms.split(",") if k.strip()] if keyterms else None

    return await service.transcribe_file(
        audio_bytes=audio_bytes,
        filename=filename or "",
        model=model,
        language=language,
        diarize=diarize,
        smart_format=smart_format,
        punctuate=punctuate,
        keyterms=keyterm_list,
        include_raw=include_raw,
        include_words=include_words,
    )


@app.post(
    "/api/v1/transcribe/url",
    response_model=TranscriptionResponse,
    tags=["Transcription"],
    summary="Transcribe Audio from Remote URL",
)
async def transcribe_audio_url(
    payload: TranscribeUrlRequest,
    service: DeepgramService = Depends(get_deepgram_service),
):
    """Transcribes an audio file hosted at an accessible URL.
    
    Compatible with the sample code pattern, but with full support for
    multilingual code-switching, speaker diarization, and custom parameters.
    """
    return await service.transcribe_url(
        url=payload.url,
        model=payload.model,
        language=payload.language,
        diarize=payload.diarize,
        smart_format=payload.smart_format,
        punctuate=payload.punctuate,
        keyterms=payload.keyterms,
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
