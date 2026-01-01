import os
import logging

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

# Locate .env file (search upward from current working directory)
dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    # Load .env and override environment variables
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Configuration loaded from {dotenv_path}")
else:
    logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Required (only for OpenAI mode)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed

    # Local S2S configuration (Qwen3-Omni)
    LOCAL_S2S_ENABLED = os.getenv("LOCAL_S2S_ENABLED", "false").lower() in ("true", "1", "yes")
    LOCAL_S2S_MODEL = os.getenv("LOCAL_S2S_MODEL", "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit")
    LOCAL_S2S_SPEAKER = os.getenv("LOCAL_S2S_SPEAKER", "Ethan")  # Ethan, Chelsie, or Aiden
    LOCAL_S2S_VAD_THRESHOLD = float(os.getenv("LOCAL_S2S_VAD_THRESHOLD", "0.02"))  # RMS threshold for speech detection
    LOCAL_S2S_SKIP_GREETING = os.getenv("LOCAL_S2S_SKIP_GREETING", "false").lower() in ("true", "1", "yes")  # Skip greeting generation

    # Latency tuning options
    LOCAL_S2S_CHUNK_SIZE = int(os.getenv("LOCAL_S2S_CHUNK_SIZE", "100"))  # Streaming chunk size (smaller = faster first audio)
    LOCAL_S2S_THINKER_TOKENS = int(os.getenv("LOCAL_S2S_THINKER_TOKENS", "256"))  # Max "thinking" tokens (reduce for faster response)
    LOCAL_S2S_TALKER_TOKENS = int(os.getenv("LOCAL_S2S_TALKER_TOKENS", "1024"))  # Max speech tokens
    LOCAL_S2S_USE_SILERO_VAD = os.getenv("LOCAL_S2S_USE_SILERO_VAD", "true").lower() in ("true", "1", "yes")  # Use Silero VAD vs energy-based
    LOCAL_S2S_MODEL_PERSISTENCE = os.getenv("LOCAL_S2S_MODEL_PERSISTENCE", "true").lower() in ("true", "1", "yes")  # Keep model in memory across sessions
    LOCAL_S2S_PREFIX_CACHING = os.getenv("LOCAL_S2S_PREFIX_CACHING", "true").lower() in ("true", "1", "yes")  # Pre-compute system prompt KV cache

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")
    logger.debug(f"Local S2S: enabled={LOCAL_S2S_ENABLED}, model={LOCAL_S2S_MODEL}, speaker={LOCAL_S2S_SPEAKER}")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
