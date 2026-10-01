import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException, status
from deepgram import AsyncDeepgramClient

from app.config import Settings, get_settings
from app.schemas import DiarizedUtterance, TranscriptionResponse, WordTimestamp

logger = logging.getLogger(__name__)

# Common video file extensions that usually contain video streams
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".flv", ".wmv", ".webm"}


def extract_audio_if_video(audio_bytes: bytes, filename: str = "") -> bytes:
    """If the uploaded file is a video container (like MP4), extract the audio track.
    
    This dramatically reduces file size (often 80-95% smaller), preventing upload timeouts
    and optimizing network transfer to Deepgram.
    """
    ext = Path(filename).suffix.lower() if filename else ""
    is_video_ext = ext in VIDEO_EXTENSIONS
    is_mp4_magic = len(audio_bytes) > 8 and audio_bytes[4:8] in (b"ftyp", b"moov")

    # If it's not a video format or ffmpeg isn't installed, return as is
    if not (is_video_ext or is_mp4_magic) or not shutil.which("ffmpeg"):
        return audio_bytes

    temp_dir = tempfile.mkdtemp(prefix="audio_extract_")
    try:
        input_suffix = ext if ext else ".mp4"
        input_path = os.path.join(temp_dir, f"input{input_suffix}")
        output_path = os.path.join(temp_dir, "extracted.m4a")

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # Attempt 1: Copy audio stream directly without re-encoding (instantaneous, < 0.1s)
        cmd_copy = [
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-c:a", "copy",
            output_path,
        ]
        proc = subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("Extracted audio stream from video (%s -> %s bytes)", len(audio_bytes), os.path.getsize(output_path))
            with open(output_path, "rb") as f:
                return f.read()

        # Attempt 2: Re-encode audio to compressed MP3 if copy fails
        output_mp3 = os.path.join(temp_dir, "extracted.mp3")
        cmd_reencode = [
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-c:a", "libmp3lame", "-b:a", "128k",
            output_mp3,
        ]
        proc2 = subprocess.run(cmd_reencode, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc2.returncode == 0 and os.path.exists(output_mp3) and os.path.getsize(output_mp3) > 0:
            logger.info("Re-encoded video audio to mp3 (%s -> %s bytes)", len(audio_bytes), os.path.getsize(output_mp3))
            with open(output_mp3, "rb") as f:
                return f.read()

    except Exception as exc:
        logger.warning("Audio extraction failed (%s), passing original bytes", exc)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return audio_bytes


class DeepgramService:
    """Service to interact with Deepgram API for Speech-to-Text transcription."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def _get_client(self) -> AsyncDeepgramClient:
        """Create and return an authenticated AsyncDeepgramClient with extended timeouts."""
        if not self.settings.is_api_key_configured:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Deepgram API key is not configured or still set to placeholder. "
                    "Please set a valid DEEPGRAM_API_KEY in your .env file or environment."
                ),
            )

        # Generous timeout for large files (prevents httpx.WriteTimeout / ReadTimeout)
        custom_timeout = httpx.Timeout(
            timeout=self.settings.DEEPGRAM_TIMEOUT_SECONDS,
            connect=60.0,
            read=self.settings.DEEPGRAM_TIMEOUT_SECONDS,
            write=self.settings.DEEPGRAM_TIMEOUT_SECONDS,
            pool=60.0,
        )
        custom_httpx = httpx.AsyncClient(timeout=custom_timeout)

        return AsyncDeepgramClient(
            api_key=self.settings.DEEPGRAM_API_KEY,
            httpx_client=custom_httpx,
            timeout=self.settings.DEEPGRAM_TIMEOUT_SECONDS,
        )

    def _build_options(
        self,
        model: Optional[str] = None,
        language: Optional[str] = None,
        diarize: Optional[bool] = None,
        smart_format: Optional[bool] = None,
        punctuate: Optional[bool] = None,
        paragraphs: Optional[bool] = None,
        utterances: Optional[bool] = None,
        keyterms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Construct options payload for Deepgram API."""
        selected_model = model or self.settings.DEFAULT_MODEL
        selected_language = language if language is not None else self.settings.DEFAULT_LANGUAGE

        opts: Dict[str, Any] = {
            "model": selected_model,
            "smart_format": smart_format if smart_format is not None else self.settings.DEFAULT_SMART_FORMAT,
            "punctuate": punctuate if punctuate is not None else self.settings.DEFAULT_PUNCTUATE,
            "diarize": diarize if diarize is not None else self.settings.DEFAULT_DIARIZE,
            "paragraphs": paragraphs if paragraphs is not None else self.settings.DEFAULT_PARAGRAPHS,
            "utterances": utterances if utterances is not None else self.settings.DEFAULT_UTTERANCES,
        }

        # Language selection:
        # If language is "multi", Nova-3 enables multilingual code-switching (e.g. Hindi + English).
        # If language is "auto", we omit language and set detect_language=True.
        if selected_language:
            if selected_language.lower() in ("auto", "detect"):
                opts["detect_language"] = True
            else:
                opts["language"] = selected_language

        if keyterms:
            opts["keyterm"] = keyterms

        return opts

    def _parse_response(
        self,
        response: Any,
        model_used: str,
        language_mode: str,
        include_raw: bool = False,
        include_words: bool = False,
    ) -> TranscriptionResponse:
        """Parse Deepgram API response object into standardized TranscriptionResponse schema."""
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif isinstance(response, dict):
            data = response
        else:
            data = {}

        metadata = data.get("metadata") or {}
        duration = metadata.get("duration")

        results = data.get("results") or {}
        channels = results.get("channels") or []

        transcript_text = ""
        overall_confidence: Optional[float] = None
        words_list: List[WordTimestamp] = []

        if channels and len(channels) > 0:
            primary_channel = channels[0] or {}
            alternatives = primary_channel.get("alternatives") or []
            if alternatives and len(alternatives) > 0:
                best_alt = alternatives[0] or {}
                transcript_text = (best_alt.get("transcript") or "").strip()
                raw_conf = best_alt.get("confidence")
                try:
                    overall_confidence = float(raw_conf) if raw_conf is not None else None
                except (ValueError, TypeError):
                    overall_confidence = None

                raw_words = best_alt.get("words") or []
                for w in raw_words:
                    word_str = w.get("word", "")
                    if not word_str:
                        continue
                    
                    raw_spk = w.get("speaker")
                    try:
                        spk_val = int(raw_spk) if raw_spk is not None else None
                    except (ValueError, TypeError):
                        spk_val = None

                    raw_w_start = w.get("start")
                    try:
                        w_start = float(raw_w_start) if raw_w_start is not None else 0.0
                    except (ValueError, TypeError):
                        w_start = 0.0

                    raw_w_end = w.get("end")
                    try:
                        w_end = float(raw_w_end) if raw_w_end is not None else 0.0
                    except (ValueError, TypeError):
                        w_end = 0.0

                    raw_w_conf = w.get("confidence")
                    try:
                        w_conf = float(raw_w_conf) if raw_w_conf is not None else None
                    except (ValueError, TypeError):
                        w_conf = None

                    wt = WordTimestamp(
                        word=word_str,
                        start=w_start,
                        end=w_end,
                        confidence=w_conf,
                        speaker=spk_val,
                    )
                    words_list.append(wt)

        # Parse diarized utterances with safe handling of None / missing speaker
        diarized_utterances: List[DiarizedUtterance] = []
        raw_utterances = results.get("utterances") or []

        if raw_utterances:
            for utt in raw_utterances:
                text = (utt.get("transcript") or "").strip()
                if not text:
                    continue

                raw_spk = utt.get("speaker")
                try:
                    speaker_id = int(raw_spk) if raw_spk is not None else 0
                except (ValueError, TypeError):
                    speaker_id = 0

                raw_start = utt.get("start")
                try:
                    start_val = float(raw_start) if raw_start is not None else 0.0
                except (ValueError, TypeError):
                    start_val = 0.0

                raw_end = utt.get("end")
                try:
                    end_val = float(raw_end) if raw_end is not None else 0.0
                except (ValueError, TypeError):
                    end_val = 0.0

                raw_conf = utt.get("confidence")
                try:
                    conf_val = float(raw_conf) if raw_conf is not None else 1.0
                except (ValueError, TypeError):
                    conf_val = 1.0

                diarized_utterances.append(
                    DiarizedUtterance(
                        speaker=speaker_id,
                        start=start_val,
                        end=end_val,
                        text=text,
                        confidence=conf_val,
                    )
                )
        elif words_list and any(w.speaker is not None for w in words_list):
            # Fallback: Group sequential words by speaker if utterances array was omitted
            current_speaker: Optional[int] = None
            current_words: List[str] = []
            curr_start = 0.0
            curr_end = 0.0
            confidences: List[float] = []

            for w in words_list:
                spk = w.speaker if w.speaker is not None else 0
                if current_speaker is None:
                    current_speaker = spk
                    curr_start = w.start
                    curr_end = w.end
                    current_words = [w.word]
                    if w.confidence is not None:
                        confidences = [w.confidence]
                elif spk == current_speaker:
                    current_words.append(w.word)
                    curr_end = w.end
                    if w.confidence is not None:
                        confidences.append(w.confidence)
                else:
                    avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
                    diarized_utterances.append(
                        DiarizedUtterance(
                            speaker=current_speaker,
                            start=curr_start,
                            end=curr_end,
                            text=" ".join(current_words),
                            confidence=round(avg_conf, 3),
                        )
                    )
                    current_speaker = spk
                    curr_start = w.start
                    curr_end = w.end
                    current_words = [w.word]
                    confidences = [w.confidence] if w.confidence is not None else []

            if current_words and current_speaker is not None:
                avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
                diarized_utterances.append(
                    DiarizedUtterance(
                        speaker=current_speaker,
                        start=curr_start,
                        end=curr_end,
                        text=" ".join(current_words),
                        confidence=round(avg_conf, 3),
                    )
                )

        word_count = len(transcript_text.split()) if transcript_text else 0

        return TranscriptionResponse(
            status="success",
            model_used=model_used,
            language_mode=language_mode,
            duration_seconds=duration,
            transcript=transcript_text,
            confidence=overall_confidence,
            word_count=word_count,
            diarized_utterances=diarized_utterances,
            words=words_list if include_words else None,
            raw_deepgram=data if include_raw else None,
        )

    async def transcribe_file(
        self,
        audio_bytes: bytes,
        filename: str = "",
        model: Optional[str] = None,
        language: Optional[str] = None,
        diarize: Optional[bool] = None,
        smart_format: Optional[bool] = None,
        punctuate: Optional[bool] = None,
        paragraphs: Optional[bool] = None,
        utterances: Optional[bool] = None,
        keyterms: Optional[List[str]] = None,
        include_raw: bool = False,
        include_words: bool = False,
    ) -> TranscriptionResponse:
        """Transcribe an in-memory audio byte payload using Deepgram."""
        client = self._get_client()
        options = self._build_options(
            model=model,
            language=language,
            diarize=diarize,
            smart_format=smart_format,
            punctuate=punctuate,
            paragraphs=paragraphs,
            utterances=utterances,
            keyterms=keyterms,
        )

        # Preprocess video files (e.g. MP4) to extract audio and prevent WriteTimeout
        processed_bytes = extract_audio_if_video(audio_bytes, filename=filename)

        try:
            response = await client.listen.v1.media.transcribe_file(
                request=processed_bytes,
                **options,
            )
            return self._parse_response(
                response=response,
                model_used=options.get("model", "unknown"),
                language_mode=options.get("language", "auto"),
                include_raw=include_raw,
                include_words=include_words,
            )
        except Exception as exc:
            logger.error("Deepgram file transcription error: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Deepgram transcription error: {str(exc)}",
            )

    async def transcribe_url(
        self,
        url: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        diarize: Optional[bool] = None,
        smart_format: Optional[bool] = None,
        punctuate: Optional[bool] = None,
        paragraphs: Optional[bool] = None,
        utterances: Optional[bool] = None,
        keyterms: Optional[List[str]] = None,
        include_raw: bool = False,
        include_words: bool = False,
    ) -> TranscriptionResponse:
        """Transcribe an audio file hosted at an accessible URL using Deepgram."""
        client = self._get_client()
        options = self._build_options(
            model=model,
            language=language,
            diarize=diarize,
            smart_format=smart_format,
            punctuate=punctuate,
            paragraphs=paragraphs,
            utterances=utterances,
            keyterms=keyterms,
        )

        try:
            response = await client.listen.v1.media.transcribe_url(
                url=url,
                **options,
            )
            return self._parse_response(
                response=response,
                model_used=options.get("model", "unknown"),
                language_mode=options.get("language", "auto"),
                include_raw=include_raw,
                include_words=include_words,
            )
        except Exception as exc:
            logger.error("Deepgram URL transcription error: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Deepgram transcription error: {str(exc)}",
            )


def get_deepgram_service() -> DeepgramService:
    """Dependency injector for DeepgramService."""
    return DeepgramService()
