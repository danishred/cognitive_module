from typing import Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    api_key_configured: bool = Field(..., description="Whether a valid Deepgram API key is present")
    default_model: str = Field(..., json_schema_extra={"example": "nova-3"})
    default_language: str = Field(..., json_schema_extra={"example": "multi"})
    message: str


class TranscribeUrlRequest(BaseModel):
    url: str = Field(..., description="Public or pre-signed URL of the audio file to transcribe")
    model: Optional[str] = Field(None, description="Deepgram model (e.g. 'nova-3', 'nova-3-medical', 'nova-2')")
    language: Optional[str] = Field(None, description="Language code (use 'multi' for English-Hindi code-switching, 'en', or 'hi')")
    diarize: Optional[bool] = Field(None, description="Enable speaker diarization")
    smart_format: Optional[bool] = Field(None, description="Enable smart formatting (dates, currencies, punctuation)")
    punctuate: Optional[bool] = Field(None, description="Enable punctuation")
    keyterms: Optional[List[str]] = Field(None, description="Custom keyterms to boost recognition")


class DiarizedUtterance(BaseModel):
    speaker: Optional[int] = Field(default=0, description="Speaker identifier (e.g., 0, 1)")
    start: float = Field(default=0.0, description="Start timestamp in seconds")
    end: float = Field(default=0.0, description="End timestamp in seconds")
    text: str = Field(default="", description="Spoken text in this utterance")
    confidence: Optional[float] = Field(default=1.0, description="Confidence score between 0.0 and 1.0")


class WordTimestamp(BaseModel):
    word: str
    start: float = Field(default=0.0)
    end: float = Field(default=0.0)
    confidence: Optional[float] = Field(default=None)
    speaker: Optional[int] = Field(default=None)


class TranscriptionResponse(BaseModel):
    status: str = "success"
    model_used: str
    language_mode: str
    duration_seconds: Optional[float] = None
    transcript: str = Field(default="", description="Complete transcription text")
    confidence: Optional[float] = None
    word_count: int = 0
    diarized_utterances: List[DiarizedUtterance] = Field(default_factory=list)
    words: Optional[List[WordTimestamp]] = None
    raw_deepgram: Optional[dict] = None
