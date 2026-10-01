import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DEEPGRAM_API_KEY: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Speech-to-Text defaults
    # For code-switching (English + Hindi), "nova-3" + "multi" is the recommended pairing.
    DEFAULT_MODEL: str = "nova-3"
    DEFAULT_LANGUAGE: str = "multi"

    DEFAULT_DIARIZE: bool = True
    DEFAULT_SMART_FORMAT: bool = True
    DEFAULT_PUNCTUATE: bool = True
    DEFAULT_PARAGRAPHS: bool = True
    DEFAULT_UTTERANCES: bool = True

    # Network timeout for large audio/video file uploads (seconds)
    DEEPGRAM_TIMEOUT_SECONDS: float = 300.0

    @property
    def is_api_key_configured(self) -> bool:
        """Check if DEEPGRAM_API_KEY is present and not a dummy/placeholder."""
        key = self.DEEPGRAM_API_KEY.strip()
        return bool(key) and key != "YOUR_DEEPGRAM_API_KEY_HERE" and key != "YOUR_SECRET"


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
