import io
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.config import Settings
from app.services.deepgram_service import DeepgramService, extract_audio_if_video
from app.schemas import TranscriptionResponse, DiarizedUtterance

client = TestClient(app)


def test_health_check_endpoint():
    """Verify /health returns system status and configuration."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "api_key_configured" in data
    assert data["default_model"] == "nova-3"
    assert data["default_language"] == "multi"


def test_serve_ui_endpoint():
    """Verify GET / returns the HTML user interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cognitive Audio STT" in response.text
    assert "Hinglish" in response.text


def test_get_sample_audio_endpoint():
    """Verify /sample-audio streams the sample mp3 file."""
    response = client.get("/sample-audio")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0


def test_transcribe_empty_file():
    """Verify /api/v1/transcribe rejects empty files."""
    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcribe_missing_key_error():
    """Verify that attempting transcription with placeholder key raises a clean 400."""
    dummy_settings = Settings(DEEPGRAM_API_KEY="YOUR_DEEPGRAM_API_KEY_HERE")
    service = DeepgramService(settings=dummy_settings)

    with pytest.raises(Exception) as exc_info:
        service._get_client()
    assert "Deepgram API key is not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_deepgram_service_parser_with_null_speaker():
    """Verify parsing when Deepgram returns speaker: None (the bug reported in cost_of_care.mp3)."""
    service = DeepgramService(settings=Settings(DEEPGRAM_API_KEY="test_key"))

    payload_with_none_speaker = {
        "metadata": {"duration": 10.0},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello this is a test without confident speaker.",
                            "confidence": 0.95,
                            "words": [
                                {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.9, "speaker": None},
                                {"word": "this", "start": 0.6, "end": 0.8, "confidence": None, "speaker": None},
                            ],
                        }
                    ]
                }
            ],
            "utterances": [
                {
                    "speaker": None,  # Deepgram sometimes returns None when speaker is unknown
                    "start": 0.0,
                    "end": 2.5,
                    "transcript": "Hello this is a test without confident speaker.",
                    "confidence": None,
                }
            ],
        },
    }

    # Must NOT raise ValidationError!
    result = service._parse_response(
        response=payload_with_none_speaker,
        model_used="nova-3",
        language_mode="multi",
        include_words=True,
    )
    assert result.status == "success"
    assert len(result.diarized_utterances) == 1
    assert result.diarized_utterances[0].speaker == 0
    assert result.diarized_utterances[0].confidence == 1.0
    assert len(result.words) == 2
    assert result.words[0].speaker is None
    assert result.words[1].confidence is None


@pytest.mark.asyncio
async def test_deepgram_service_parser():
    """Verify parsing of Deepgram response with Hinglish transcript and diarized utterances."""
    service = DeepgramService(settings=Settings(DEEPGRAM_API_KEY="test_key"))

    mock_payload = {
        "metadata": {
            "duration": 14.5,
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello doctor, mujhe kal se fever aur headache ho raha hai. Did you take any medicine?",
                            "confidence": 0.96,
                            "words": [
                                {"word": "Hello", "start": 0.2, "end": 0.5, "confidence": 0.99, "speaker": 0},
                                {"word": "doctor,", "start": 0.6, "end": 1.0, "confidence": 0.98, "speaker": 0},
                                {"word": "mujhe", "start": 1.1, "end": 1.4, "confidence": 0.95, "speaker": 0},
                                {"word": "kal", "start": 1.5, "end": 1.7, "confidence": 0.97, "speaker": 0},
                                {"word": "se", "start": 1.8, "end": 1.9, "confidence": 0.99, "speaker": 0},
                                {"word": "fever", "start": 2.0, "end": 2.3, "confidence": 0.96, "speaker": 0},
                                {"word": "aur", "start": 2.4, "end": 2.6, "confidence": 0.94, "speaker": 0},
                                {"word": "headache", "start": 2.7, "end": 3.1, "confidence": 0.96, "speaker": 0},
                                {"word": "ho", "start": 3.2, "end": 3.3, "confidence": 0.97, "speaker": 0},
                                {"word": "raha", "start": 3.4, "end": 3.6, "confidence": 0.98, "speaker": 0},
                                {"word": "hai.", "start": 3.7, "end": 4.0, "confidence": 0.95, "speaker": 0},
                                {"word": "Did", "start": 4.5, "end": 4.7, "confidence": 0.97, "speaker": 1},
                                {"word": "you", "start": 4.8, "end": 4.9, "confidence": 0.98, "speaker": 1},
                                {"word": "take", "start": 5.0, "end": 5.2, "confidence": 0.96, "speaker": 1},
                                {"word": "any", "start": 5.3, "end": 5.5, "confidence": 0.95, "speaker": 1},
                                {"word": "medicine?", "start": 5.6, "end": 6.0, "confidence": 0.99, "speaker": 1},
                            ],
                        }
                    ]
                }
            ],
            "utterances": [
                {
                    "speaker": 0,
                    "start": 0.2,
                    "end": 4.0,
                    "transcript": "Hello doctor, mujhe kal se fever aur headache ho raha hai.",
                    "confidence": 0.96,
                },
                {
                    "speaker": 1,
                    "start": 4.5,
                    "end": 6.0,
                    "transcript": "Did you take any medicine?",
                    "confidence": 0.97,
                },
            ],
        },
    }

    result = service._parse_response(
        response=mock_payload,
        model_used="nova-3",
        language_mode="multi",
        include_raw=True,
        include_words=True,
    )

    assert result.status == "success"
    assert result.model_used == "nova-3"
    assert result.language_mode == "multi"
    assert result.duration_seconds == 14.5
    assert "mujhe kal se fever" in result.transcript
    assert len(result.diarized_utterances) == 2
    assert result.diarized_utterances[0].speaker == 0
    assert result.diarized_utterances[0].text == "Hello doctor, mujhe kal se fever aur headache ho raha hai."
    assert result.diarized_utterances[1].speaker == 1
    assert result.diarized_utterances[1].text == "Did you take any medicine?"
    assert len(result.words) == 16


def test_transcribe_endpoint_with_mock():
    """Verify POST /api/v1/transcribe end-to-end with mocked Deepgram service."""
    mock_response = TranscriptionResponse(
        status="success",
        model_used="nova-3",
        language_mode="multi",
        duration_seconds=5.0,
        transcript="Test transcript",
        confidence=0.98,
        word_count=2,
        diarized_utterances=[
            DiarizedUtterance(speaker=0, start=0.0, end=2.5, text="Test transcript", confidence=0.98)
        ],
    )

    with patch.object(DeepgramService, "transcribe_file", AsyncMock(return_value=mock_response)):
        audio_content = b"FAKE_AUDIO_DATA_FOR_TESTING"
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"model": "nova-3", "language": "multi", "diarize": "true"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_used"] == "nova-3"
        assert data["transcript"] == "Test transcript"
        assert len(data["diarized_utterances"]) == 1
        assert data["diarized_utterances"][0]["speaker"] == 0


def test_extract_audio_if_video_passthrough_for_audio():
    """Verify audio files like MP3 pass through unchanged."""
    raw_audio = b"FAKE_AUDIO_DATA_FOR_PASSTHROUGH"
    out = extract_audio_if_video(raw_audio, filename="sample.mp3")
    assert out == raw_audio
