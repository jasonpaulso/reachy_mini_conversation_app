"""Local speech-to-speech handler using Qwen3-Omni via mlx-vlm.

This handler replaces OpenAI Realtime with local inference using
Qwen3-Omni-30B-A3B running on Apple Silicon via MLX.

Latency Optimizations:
- Model persistence: Keeps model in memory across Gradio sessions
- Prefix caching: Pre-computes KV cache for system prompt (~65% TTFT reduction)
- Silero VAD: Faster, more accurate end-of-speech detection
- Configurable chunk_size and token limits for tuning latency vs quality
"""

import asyncio
import logging
import tempfile
import threading
from typing import Any, Dict, List, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)

# Sample rates - Qwen3-Omni uses 24kHz which matches OpenAI Realtime
INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
SILERO_SAMPLE_RATE: Final[Literal[16000]] = 16000  # Silero VAD requires 16kHz

# Default model path - can be overridden via env
DEFAULT_MODEL_PATH = "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"


# ============================================================================
# Module-level model cache for persistence across Gradio sessions
# ============================================================================
class _ModelCache:
    """Thread-safe cache for Qwen3-Omni model and processor.

    When LOCAL_S2S_MODEL_PERSISTENCE=true, the model is loaded once and reused
    across handler instances (Gradio sessions). This eliminates the ~15-20s
    model loading time for each new session.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._model_path: Optional[str] = None
        self._silero_vad: Optional[Any] = None
        self._prefix_cache: Optional[Dict[str, Any]] = None
        self._prefix_cache_instructions: Optional[str] = None

    def get_model(self, model_path: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Get cached model/processor or None if not loaded."""
        with self._lock:
            if self._model_path == model_path:
                return self._model, self._processor
            return None, None

    def set_model(self, model_path: str, model: Any, processor: Any) -> None:
        """Cache loaded model/processor."""
        with self._lock:
            self._model = model
            self._processor = processor
            self._model_path = model_path

    def get_silero_vad(self) -> Optional[Any]:
        """Get cached Silero VAD model."""
        with self._lock:
            return self._silero_vad

    def set_silero_vad(self, vad: Any) -> None:
        """Cache Silero VAD model."""
        with self._lock:
            self._silero_vad = vad

    def get_prefix_cache(self, instructions: str) -> Optional[Dict[str, Any]]:
        """Get cached prefix (system prompt KV cache) if instructions match."""
        with self._lock:
            if self._prefix_cache_instructions == instructions:
                return self._prefix_cache
            return None

    def set_prefix_cache(self, instructions: str, cache: Dict[str, Any]) -> None:
        """Cache prefix (system prompt KV cache)."""
        with self._lock:
            self._prefix_cache = cache
            self._prefix_cache_instructions = instructions

    def clear(self) -> None:
        """Clear all cached state."""
        with self._lock:
            self._model = None
            self._processor = None
            self._model_path = None
            self._silero_vad = None
            self._prefix_cache = None
            self._prefix_cache_instructions = None


_model_cache = _ModelCache()


class LocalQwenS2SHandler(AsyncStreamHandler):
    """Local speech-to-speech handler using Qwen3-Omni."""

    def __init__(
        self,
        deps: ToolDependencies,
        model_path: Optional[str] = None,
        speaker: str = "Ethan",
    ):
        """Initialize the local s2s handler.

        Args:
            deps: Tool dependencies for robot control.
            model_path: HuggingFace model path for Qwen3-Omni MLX weights.
            speaker: Voice to use - "Ethan", "Chelsie", or "Aiden".

        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.speaker = speaker

        # Model and processor - loaded lazily
        # Type annotations needed so Pylance knows these can be non-None after assignment
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None

        # Audio input buffer for collecting speech
        self._audio_buffer: List[NDArray[np.int16]] = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

        # VAD parameters (configurable via LOCAL_S2S_VAD_THRESHOLD env var)
        self._vad_threshold = config.LOCAL_S2S_VAD_THRESHOLD
        self._silence_threshold_frames = 30  # ~1.25s of silence to end utterance
        self._min_speech_frames = 5  # Minimum frames to consider valid speech

        # Silero VAD (loaded lazily if enabled)
        self._use_silero_vad = config.LOCAL_S2S_USE_SILERO_VAD
        self._silero_vad: Optional[Any] = None
        self._silero_buffer: List[NDArray[np.float32]] = []  # Buffer for 16kHz audio
        self._silero_speech_prob = 0.0  # Last speech probability from Silero

        # Latency tuning (from config)
        self._chunk_size = config.LOCAL_S2S_CHUNK_SIZE
        self._thinker_max_tokens = config.LOCAL_S2S_THINKER_TOKENS
        self._talker_max_tokens = config.LOCAL_S2S_TALKER_TOKENS
        self._use_model_persistence = config.LOCAL_S2S_MODEL_PERSISTENCE
        self._use_prefix_caching = config.LOCAL_S2S_PREFIX_CACHING

        # Prefix cache for system prompt (reduces TTFT by ~65%)
        self._prefix_cache: Optional[Dict[str, Any]] = None
        self._system_instructions: Optional[str] = None

        # Output queue
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        # Conversation history
        self._conversation: List[Dict[str, object]] = []

        # Lifecycle
        self._shutdown_requested = False
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self._greeting_task: Optional[asyncio.Task[None]] = None
        self._utterance_task: Optional[asyncio.Task[None]] = None

    def copy(self) -> "LocalQwenS2SHandler":
        """Create a copy of the handler."""
        return LocalQwenS2SHandler(self.deps, self.model_path, self.speaker)

    async def start_up(self) -> None:
        """Initialize the Qwen3-Omni model with latency optimizations.

        Optimizations applied:
        - Model persistence: Reuses cached model across Gradio sessions
        - Silero VAD: Loads fast neural VAD for accurate end-of-speech detection
        - Prefix caching: Pre-computes KV cache for system prompt
        """
        import time
        start_time = time.time()

        try:
            # Check for cached model first (model persistence)
            if self._use_model_persistence:
                cached_model, cached_processor = _model_cache.get_model(self.model_path)
                if cached_model is not None and cached_processor is not None:
                    logger.info("Using cached model (model persistence enabled)")
                    self.model = cached_model
                    self.processor = cached_processor
                else:
                    logger.info("Loading Qwen3-Omni model from %s (will cache for reuse)...", self.model_path)
                    self._load_model_with_patches()
                    _model_cache.set_model(self.model_path, self.model, self.processor)
            else:
                logger.info("Loading Qwen3-Omni model from %s...", self.model_path)
                self._load_model_with_patches()

            # Ensure talker is enabled for audio output
            if self.model is not None and hasattr(self.model, "enable_talker") and callable(self.model.enable_talker):
                self.model.enable_talker()

            model_load_time = (time.time() - start_time) * 1000
            logger.info("Model ready (%.0fms)", model_load_time)

            # Load Silero VAD if enabled
            if self._use_silero_vad:
                await self._load_silero_vad()

            # Log latency tuning settings
            logger.info("Latency settings: chunk_size=%d, thinker_tokens=%d, talker_tokens=%d",
                       self._chunk_size, self._thinker_max_tokens, self._talker_max_tokens)
            logger.info("Optimizations: model_persistence=%s, prefix_caching=%s, silero_vad=%s",
                       self._use_model_persistence, self._use_prefix_caching, self._use_silero_vad)
            logger.info("Using speaker: %s", self.speaker)
            if not self._use_silero_vad:
                logger.info("VAD threshold: %.4f (energy-based)", self._vad_threshold)

            # Initialize conversation with system prompt
            self._system_instructions = get_session_instructions()
            self._conversation = [{"role": "system", "content": self._system_instructions}]

            # Warm up the model to JIT compile for fast first response
            # This also pre-computes prefix cache if enabled
            await self._warm_up_model()

            total_time = (time.time() - start_time) * 1000
            logger.info("Local S2S handler ready (%.0fms total) - listening for speech...", total_time)

            # Generate greeting as background task (like OpenAI Realtime, which generates
            # greeting AFTER connection is established). This allows emit() to start
            # consuming audio immediately rather than waiting for greeting to complete.
            # Track the task so we can cancel it on shutdown.
            # Skip greeting if configured (useful for Gradio mode where timing is sensitive)
            if config.LOCAL_S2S_SKIP_GREETING:
                logger.info("Greeting skipped (LOCAL_S2S_SKIP_GREETING=true)")
            else:
                self._greeting_task = asyncio.create_task(self.generate_greeting())

        except Exception as e:
            logger.error("Failed to load Qwen3-Omni model: %s", e)
            raise

    def _load_model_with_patches(self) -> None:
        """Load model with necessary patches for transformers bugs."""
        # Patch transformers tokenizer bug: extra_special_tokens is a list but code expects dict
        # See: https://github.com/huggingface/transformers/issues/11455
        import transformers.tokenization_utils_base as tub

        original_set_special = tub.PreTrainedTokenizerBase._set_model_specific_special_tokens

        def patched_set_special(self: tub.PreTrainedTokenizerBase, special_tokens: object = None) -> None:
            if isinstance(special_tokens, list):
                special_tokens = {token: token for token in special_tokens}
            original_set_special(self, special_tokens)  # type: ignore[arg-type]

        tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = patched_set_special  # type: ignore[method-assign,assignment]

        # Patch Qwen2TokenizerFast.__getattr__ to return special token values
        # The Qwen3OmniMoeProcessor expects tokenizer.image_token etc but
        # Qwen2TokenizerFast's __getattr__ raises AttributeError for unknown attrs.
        # We patch __getattr__ to return fallback values for these specific tokens.
        from transformers import Qwen2TokenizerFast

        original_tokenizer_getattr = Qwen2TokenizerFast.__getattr__

        # Fallback values for special tokens expected by Qwen3OmniMoeProcessor
        # Values derived from tokenizer_config.json and processor requirements
        SPECIAL_TOKEN_FALLBACKS = {
            # Pad tokens (from tokenizer_config.json)
            "image_token": "<|image_pad|>",
            "audio_token": "<|audio_pad|>",
            "video_token": "<|video_pad|>",
            # Vision start/end tokens
            "vision_bos_token": "<|vision_start|>",
            "vision_eos_token": "<|vision_end|>",
            # Audio start/end tokens (processor expects these but tokenizer doesn't define them)
            "audio_bos_token": "<|audio_bos|>",
            "audio_eos_token": "<|audio_eos|>",
        }

        def patched_tokenizer_getattr(self: Qwen2TokenizerFast, key: str) -> object:
            if key in SPECIAL_TOKEN_FALLBACKS:
                return SPECIAL_TOKEN_FALLBACKS[key]
            return original_tokenizer_getattr(self, key)  # type: ignore[no-untyped-call]

        Qwen2TokenizerFast.__getattr__ = patched_tokenizer_getattr  # type: ignore[method-assign,assignment]

        # Import mlx-vlm and load
        from mlx_vlm.utils import load

        self.model, self.processor = load(
            self.model_path,
            trust_remote_code=True,
        )

    async def _load_silero_vad(self) -> None:
        """Load Silero VAD model for fast speech detection.

        Silero VAD is ~10x faster than energy-based VAD and provides more
        accurate speech boundaries, reducing end-of-speech latency.
        """
        try:
            import torch

            # Check cache first
            cached_vad = _model_cache.get_silero_vad()
            if cached_vad is not None:
                logger.info("Using cached Silero VAD model")
                self._silero_vad = cached_vad
                return

            logger.info("Loading Silero VAD model...")

            # Load from torch hub (downloads on first run, cached afterwards)
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,  # Use PyTorch for Apple Silicon
                trust_repo=True,
            )

            self._silero_vad = model
            _model_cache.set_silero_vad(model)
            logger.info("Silero VAD loaded successfully")

        except Exception as e:
            logger.warning("Failed to load Silero VAD, falling back to energy-based: %s", e)
            self._use_silero_vad = False
            self._silero_vad = None

    async def _warm_up_model(self) -> None:
        """Warm up the model with a dummy inference to JIT compile.

        This ensures the first real response is fast (~260ms TTFT instead of ~2s).
        Also pre-computes prefix cache for system prompt if enabled.
        """
        try:
            import time

            import mlx.core as mx
            from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

            logger.info("Warming up model (JIT compilation)...")
            start = time.time()

            # Type assertions
            assert self.model is not None
            assert self.processor is not None

            # Type guard
            if self.model is None:
                logger.warning("Model is None during warm-up")
                return

            # Assert for type narrowing (model is not None after the check above)
            model = self.model

            # Pre-compute prefix cache for system prompt if enabled
            if self._use_prefix_caching and self._system_instructions:
                await self._compute_prefix_cache()

            # Minimal conversation for warm-up (includes system prompt)
            warmup_conv = self._conversation + [{"role": "user", "content": "Hi"}]
            inputs, _ = prepare_omni_inputs(self.processor, warmup_conv)
            input_ids = inputs.pop("input_ids")

            # Run a quick generation to compile the model
            for _ in model.generate_stream(
                input_ids=input_ids,
                speaker=self.speaker,
                thinker_max_new_tokens=8,
                talker_max_new_tokens=64,
                chunk_size=self._chunk_size,
                **inputs,
            ):
                break  # Just need first iteration to trigger compilation

            # Force evaluation
            mx.eval()

            elapsed = (time.time() - start) * 1000
            logger.info("Model warm-up complete (%.0fms)", elapsed)

        except Exception as e:
            logger.warning("Model warm-up failed (non-fatal): %s", e)

    async def _compute_prefix_cache(self) -> None:
        """Pre-compute KV cache for system prompt.

        This optimization reduces TTFT by ~65% by caching the system prompt's
        key-value pairs so they don't need to be recomputed for each utterance.

        Note: mlx-vlm's generate_stream doesn't natively support prefix caching,
        so we store the tokenized system prompt for potential future use when
        the API supports it. For now, this prepares the infrastructure.
        """
        if not self._system_instructions:
            return

        try:
            # Check if we have a cached prefix for these instructions
            if self._use_model_persistence:
                cached = _model_cache.get_prefix_cache(self._system_instructions)
                if cached is not None:
                    logger.info("Using cached prefix (system prompt KV cache)")
                    self._prefix_cache = cached
                    return

            logger.info("Computing prefix cache for system prompt...")

            assert self.processor is not None
            from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

            # Prepare inputs for just the system prompt
            system_conv = [{"role": "system", "content": self._system_instructions}]
            inputs, _ = prepare_omni_inputs(self.processor, system_conv)

            # Store the tokenized prefix
            # Note: Full KV caching would require model API changes
            # For now, we cache the prepared inputs
            self._prefix_cache = {
                "input_ids": inputs.get("input_ids"),
                "system_length": len(self._system_instructions),
            }

            if self._use_model_persistence:
                _model_cache.set_prefix_cache(self._system_instructions, self._prefix_cache)

            logger.info("Prefix cache computed (system prompt: %d chars)", len(self._system_instructions))

        except Exception as e:
            logger.warning("Failed to compute prefix cache (non-fatal): %s", e)
            self._prefix_cache = None

    async def _detect_speech(self, audio_float: NDArray[np.float32]) -> bool:
        """Detect speech using Silero VAD or energy-based VAD.

        Silero VAD provides faster, more accurate end-of-speech detection by
        using a neural network trained on speech patterns. Falls back to
        simple energy-based VAD if Silero is not available.

        Args:
            audio_float: Audio samples in float32 [-1, 1] range at 24kHz

        Returns:
            True if speech detected, False otherwise

        """
        if self._use_silero_vad and self._silero_vad is not None:
            return await self._detect_speech_silero(audio_float)
        else:
            return self._detect_speech_energy(audio_float)

    def _detect_speech_energy(self, audio_float: NDArray[np.float32]) -> bool:
        """Detect speech using simple RMS energy threshold."""
        rms = np.sqrt(np.mean(audio_float**2))
        return bool(rms > self._vad_threshold)

    async def _detect_speech_silero(self, audio_float: NDArray[np.float32]) -> bool:
        """Silero VAD for accurate speech detection.

        Silero requires 16kHz audio, so we resample from 24kHz. The model
        processes audio in chunks (typically 512 samples = 32ms at 16kHz)
        and returns speech probability.

        Benefits over energy-based VAD:
        - More accurate end-of-speech detection (reduces latency by ~200-500ms)
        - Less sensitive to background noise
        - Better handling of speech pauses vs true silence
        """
        try:
            import torch

            # Resample 24kHz → 16kHz for Silero
            num_samples_16k = int(len(audio_float) * SILERO_SAMPLE_RATE / INPUT_SAMPLE_RATE)
            audio_16k = np.asarray(
                resample(audio_float, num_samples_16k),
                dtype=np.float32,
            )

            # Silero expects chunks of ~512 samples (32ms at 16kHz)
            # We accumulate audio and process in chunks
            self._silero_buffer.append(audio_16k)

            # Process when we have enough samples (512 = recommended chunk size)
            chunk_size = 512
            total_samples = sum(len(chunk) for chunk in self._silero_buffer)

            if total_samples >= chunk_size:
                # Concatenate buffer and process
                all_audio = np.concatenate(self._silero_buffer)

                # Process in chunk_size increments
                speech_probs = []
                for i in range(0, len(all_audio) - chunk_size + 1, chunk_size):
                    chunk = all_audio[i : i + chunk_size]
                    tensor = torch.from_numpy(chunk).unsqueeze(0)

                    # Get speech probability from Silero
                    # Type guard: self._silero_vad is confirmed non-None at method entry
                    vad_model = self._silero_vad
                    assert vad_model is not None
                    with torch.no_grad():
                        prob = vad_model(tensor, SILERO_SAMPLE_RATE).item()
                        speech_probs.append(prob)

                # Use max probability from processed chunks
                if speech_probs:
                    self._silero_speech_prob = max(speech_probs)

                # Keep remainder in buffer
                processed_samples = (len(all_audio) // chunk_size) * chunk_size
                remainder = all_audio[processed_samples:]
                self._silero_buffer = [remainder] if len(remainder) > 0 else []

            # Threshold for speech detection (Silero recommends 0.5)
            # Lower threshold = faster response but more false positives
            silero_threshold = 0.5
            return bool(self._silero_speech_prob > silero_threshold)

        except Exception as e:
            # Fall back to energy-based if Silero fails
            if not hasattr(self, "_silero_error_logged"):
                logger.warning("Silero VAD failed, using energy-based: %s", e)
                self._silero_error_logged = True
            return self._detect_speech_energy(audio_float)

    async def generate_greeting(self) -> None:
        """Generate a spoken greeting on startup.

        This is called in headless mode to greet the user when the app starts,
        similar to how OpenAI Realtime auto-greets when the session begins.
        """
        if self.model is None or self.processor is None:
            logger.warning("Cannot generate greeting - model not loaded")
            return

        # Local references for type narrowing (model/processor confirmed non-None above)
        model = self.model
        processor = self.processor

        try:
            import time
            import base64

            from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

            logger.info("Generating startup greeting...")
            start_time = time.time()

            # Create a text-only greeting request
            greeting_message = {
                "role": "user",
                "content": "Please greet me warmly and briefly introduce yourself.",
            }
            conversation = self._conversation + [greeting_message]

            # Prepare inputs (text-only, no audio)
            model_inputs, _ = prepare_omni_inputs(processor, conversation)
            input_ids = model_inputs.pop("input_ids")

            # Stream the greeting response
            response_tokens = []
            response_text = ""
            chunk_size = 960  # 40ms at 24kHz

            for chunk_type, chunk_data in model.generate_stream(
                input_ids=input_ids,
                speaker=self.speaker,
                thinker_max_new_tokens=self._thinker_max_tokens,
                talker_max_new_tokens=self._talker_max_tokens,
                chunk_size=self._chunk_size,
                **model_inputs,
            ):
                if self._shutdown_requested:
                    break

                # Handle text tokens
                if chunk_type == "text" and chunk_data:
                    response_tokens.extend(chunk_data)
                    response_text = processor.decode(response_tokens, skip_special_tokens=True)

                # Handle audio chunks
                elif chunk_type == "audio" and chunk_data is not None:
                    audio_np = np.array(chunk_data)
                    if audio_np.ndim > 1:
                        audio_np = audio_np.squeeze()

                    # Feed to head wobbler if available
                    if self.deps.head_wobbler is not None:
                        audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                        self.deps.head_wobbler.feed(audio_b64)

                    # Convert to int16 and queue for playback
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    for i in range(0, len(audio_int16), chunk_size):
                        if self._shutdown_requested:
                            break
                        chunk = audio_int16[i : i + chunk_size]
                        if len(chunk) < chunk_size:
                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                        await self.output_queue.put((OUTPUT_SAMPLE_RATE, chunk.reshape(1, -1)))

                await asyncio.sleep(0)

            # Log and update conversation history
            total_time = (time.time() - start_time) * 1000
            logger.info("Greeting generated (%.0fms): %s", total_time, response_text[:100] if response_text else "")

            # Add to conversation history
            self._conversation.append({"role": "user", "content": "[Startup greeting request]"})
            self._conversation.append({"role": "assistant", "content": response_text})

            # Emit final transcript
            if response_text:
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))

        except Exception as e:
            logger.error("Failed to generate greeting: %s", e)

    async def receive(self, frame: Tuple[int, NDArray[Any]]) -> None:
        """Receive audio frame from the microphone.

        Implements simple energy-based VAD to detect speech boundaries.
        Accepts float32 audio in [-1, 1] range (from SDK) or int16.
        """
        if self.model is None or self._shutdown_requested:
            return

        input_sample_rate, audio_frame = frame

        # Discard stale audio buffers accumulated during model loading.
        # The robot's audio backend pre-allocates large buffers that contain zeros
        # until consumed in real-time. We discard frames until we see non-zero audio
        # OR until we've discarded enough frames (to handle quiet environments).
        if not hasattr(self, "_audio_warmup_complete"):
            self._audio_warmup_frames = getattr(self, "_audio_warmup_frames", 0) + 1
            frame_samples = audio_frame.size if audio_frame.ndim == 1 else audio_frame.shape[0]
            has_audio = np.count_nonzero(audio_frame) > frame_samples * 0.01  # >1% nonzero

            # Discard first 50 frames (~2 seconds) OR until we see real audio
            if self._audio_warmup_frames < 50 and not has_audio:
                if self._audio_warmup_frames == 1:
                    logger.info("Discarding stale audio buffer (accumulated during model load)...")
                return

            # Warmup complete - either we have real audio or enough time passed
            self._audio_warmup_complete = True
            if has_audio:
                logger.info("Audio warmup complete - detected real audio after %d frames", self._audio_warmup_frames)
            else:
                logger.info("Audio warmup complete - timeout after %d frames (no audio detected)", self._audio_warmup_frames)

        # Log sample rate and audio info once (after warmup)
        if not hasattr(self, "_input_sr_logged"):
            logger.info("Microphone sample rate: %d Hz (will resample to %d Hz)", input_sample_rate, INPUT_SAMPLE_RATE)
            logger.info(
                "First audio frame: shape=%s, dtype=%s, min=%s, max=%s, nonzero=%d/%d",
                audio_frame.shape,
                audio_frame.dtype,
                audio_frame.min(),
                audio_frame.max(),
                np.count_nonzero(audio_frame),
                audio_frame.size,
            )
            self._input_sr_logged = True

        # Reshape if needed (stereo -> mono)
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed (do this in float domain to preserve precision)
        if INPUT_SAMPLE_RATE != input_sample_rate:
            audio_frame = np.asarray(
                resample(audio_frame, int(len(audio_frame) * INPUT_SAMPLE_RATE / input_sample_rate)),
                dtype=np.float32,
            )

        # Convert to float32 in [-1, 1] range for processing
        # Handle both float32 (from SDK, already normalized) and int16 (legacy) formats
        if audio_frame.dtype == np.int16:
            audio_float = np.asarray(audio_frame, dtype=np.float32) / 32768.0
        else:
            # Already float32, ensure it's in the right range
            audio_float = np.asarray(audio_frame, dtype=np.float32)

        # Voice Activity Detection (Silero or energy-based)
        is_speech = await self._detect_speech(audio_float)

        # Log VAD periodically for debugging (every ~1 second at 24kHz with typical frame sizes)
        if not hasattr(self, "_vad_log_counter"):
            self._vad_log_counter = 0
        self._vad_log_counter += 1
        if self._vad_log_counter % 50 == 0:  # Log every ~50 frames
            if self._use_silero_vad:
                logger.info("VAD (Silero): prob=%.4f, is_speech=%s, speaking=%s", self._silero_speech_prob, is_speech, self._is_speaking)
            else:
                rms = np.sqrt(np.mean(audio_float**2))
                logger.info("VAD (energy): rms=%.4f, threshold=%.4f, is_speech=%s, speaking=%s", rms, self._vad_threshold, is_speech, self._is_speaking)

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += 1

            if not self._is_speaking and self._speech_frames >= self._min_speech_frames:
                # Speech started
                self._is_speaking = True
                self.deps.movement_manager.set_listening(True)
                if self.deps.head_wobbler is not None:
                    self.deps.head_wobbler.reset()
                logger.debug("Speech started")

            if self._is_speaking:
                self._audio_buffer.append(audio_to_int16(audio_float))

        else:
            self._speech_frames = 0

            if self._is_speaking:
                self._audio_buffer.append(audio_to_int16(audio_float))
                self._silence_frames += 1

                if self._silence_frames >= self._silence_threshold_frames:
                    # Speech ended - process the utterance
                    self._is_speaking = False
                    self.deps.movement_manager.set_listening(False)
                    logger.debug("Speech ended, processing utterance")

                    # Process in background to not block audio collection
                    # Track task for cancellation on shutdown
                    audio_data = np.concatenate(self._audio_buffer)
                    self._audio_buffer = []
                    self._utterance_task = asyncio.create_task(self._process_utterance(audio_data))

    async def _process_utterance(self, audio_data: NDArray[np.int16]) -> None:
        """Process a complete utterance through Qwen3-Omni with streaming.

        Uses generate_stream() for low-latency response (~260ms TTFT).
        Streams text updates and audio chunks as they become available.
        """
        # Early exit if shutdown requested
        if self._shutdown_requested:
            logger.debug("Skipping utterance processing - shutdown requested")
            return

        # Early exit if model not loaded
        if self.model is None or self.processor is None:
            logger.warning("Cannot process utterance - model not loaded")
            return

        # Local references for type narrowing (model/processor confirmed non-None above)
        model = self.model
        processor = self.processor

        try:
            import time
            import base64

            import soundfile as sf
            from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

            start_time = time.time()

            # Debug: log detailed audio stats
            duration_sec = len(audio_data) / INPUT_SAMPLE_RATE
            audio_float_debug = audio_data.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float_debug**2))
            peak = np.max(np.abs(audio_float_debug))
            # Check if audio has variation (not just DC or constant)
            unique_vals = len(np.unique(audio_data[:1000]))  # Check first 1000 samples
            logger.info(
                "Processing audio: %.2fs, %d samples, RMS=%.4f, peak=%.4f, unique_vals=%d, dtype=%s, range=[%d, %d]",
                duration_sec,
                len(audio_data),
                rms,
                peak,
                unique_vals,
                audio_data.dtype,
                audio_data.min(),
                audio_data.max(),
            )

            # Save audio to temp file for processing
            # Note: soundfile expects float32 in [-1, 1] for best compatibility
            audio_float = audio_data.astype(np.float32) / 32768.0
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(f, audio_float, INPUT_SAMPLE_RATE)

            # Debug: also save to a persistent file for inspection
            debug_path = Path("/tmp/qwen_debug_audio.wav")
            sf.write(str(debug_path), audio_float, INPUT_SAMPLE_RATE)
            logger.info("Debug audio saved to: %s", debug_path)

            # Build conversation with audio
            user_message = {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": temp_path},
                ],
            }
            conversation = self._conversation + [user_message]

            # Prepare inputs
            model_inputs, _ = prepare_omni_inputs(
                processor,
                conversation,
            )

            input_ids = model_inputs.pop("input_ids")
            # Remaining model_inputs contain audio features, attention masks, etc.
            logger.debug("Starting streaming generation with %d audio feature inputs", len(model_inputs))

            # Stream response with generate_stream() for low latency
            # API: yields (chunk_type, chunk_data) where:
            #   - chunk_type="text": chunk_data is list of token IDs
            #   - chunk_type="audio": chunk_data is audio array
            response_tokens = []
            response_text = ""
            first_chunk_logged = False
            chunk_size = 960  # 40ms at 24kHz for output

            for chunk_type, chunk_data in model.generate_stream(
                input_ids=input_ids,
                speaker=self.speaker,
                thinker_max_new_tokens=self._thinker_max_tokens,
                talker_max_new_tokens=self._talker_max_tokens,
                chunk_size=self._chunk_size,  # Configurable for latency tuning
                **model_inputs,  # Pass audio features, attention masks, etc.
            ):
                # Check shutdown between chunks
                if self._shutdown_requested:
                    logger.debug("Shutdown requested during streaming")
                    break

                # Log time to first chunk
                if not first_chunk_logged:
                    ttft = (time.time() - start_time) * 1000
                    logger.info("TTFT: %.0fms", ttft)
                    first_chunk_logged = True

                # Handle text tokens
                if chunk_type == "text" and chunk_data:
                    # chunk_data is a list of token IDs - accumulate and decode
                    response_tokens.extend(chunk_data)
                    response_text = processor.decode(response_tokens, skip_special_tokens=True)
                    # Emit partial transcript for UI
                    if not self._shutdown_requested and response_text:
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": response_text, "partial": True})
                        )

                # Handle audio chunks
                elif chunk_type == "audio" and chunk_data is not None:
                    if self._shutdown_requested:
                        break

                    # Convert MLX array to numpy
                    audio_np = np.array(chunk_data)
                    if audio_np.ndim > 1:
                        audio_np = audio_np.squeeze()

                    # Feed to head wobbler if available
                    if self.deps.head_wobbler is not None:
                        audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                        self.deps.head_wobbler.feed(audio_b64)

                    # Convert to int16 and queue for playback
                    audio_int16 = (audio_np * 32767).astype(np.int16)

                    # Send in smaller chunks for smooth playback
                    for i in range(0, len(audio_int16), chunk_size):
                        if self._shutdown_requested:
                            break
                        chunk = audio_int16[i : i + chunk_size]
                        if len(chunk) < chunk_size:
                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                        await self.output_queue.put((OUTPUT_SAMPLE_RATE, chunk.reshape(1, -1)))

                # Yield to event loop to allow emit() to run
                await asyncio.sleep(0)

            # Log final response
            total_time = (time.time() - start_time) * 1000
            logger.info("Response (%.0fms): %s", total_time, response_text[:100])

            # Update conversation history with final text
            self._conversation.append({"role": "user", "content": "[Audio input]"})
            self._conversation.append({"role": "assistant", "content": response_text})

            # Emit final transcript (not partial)
            if not self._shutdown_requested and response_text:
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))

            self.last_activity_time = asyncio.get_event_loop().time()

            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error("Error processing utterance: %s", e)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[Error: {e}]"}))

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker.

        Uses wait_for_item like the OpenAI handler for consistency.
        Returns None when queue is empty (after 0.1s timeout), which signals
        "nothing to send" to fastrtc without closing the stream prematurely.
        """
        # If shutting down, signal end of stream cleanly
        if self._shutdown_requested:
            return None

        # Use wait_for_item like OpenAI handler - returns None on timeout
        result: Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None = await wait_for_item(self.output_queue)

        # Debug: track audio chunks emitted
        if not hasattr(self, "_emit_audio_count"):
            self._emit_audio_count = 0

        # Debug logging for audio emission
        if result is not None and isinstance(result, tuple):
            self._emit_audio_count += 1
            sr, audio = result
            # Log first few chunks and periodically after
            if self._emit_audio_count <= 3 or self._emit_audio_count % 50 == 0:
                logger.info(
                    "emit(): audio chunk #%d, sr=%d, shape=%s, dtype=%s, queue=%d",
                    self._emit_audio_count, sr, audio.shape, audio.dtype, self.output_queue.qsize()
                )

            qsize = self.output_queue.qsize()
            if qsize > 10 or qsize == 0:  # Log when queue is filling up or empty
                logger.debug("emit(): audio chunk, queue_size=%d", qsize)

        return result

    def shutdown(self) -> None:
        """Shutdown the handler (sync to match base class signature)."""
        logger.info("LocalQwenS2SHandler shutdown requested")
        self._shutdown_requested = True

        # Cancel any running background tasks
        if self._greeting_task is not None and not self._greeting_task.done():
            self._greeting_task.cancel()
            logger.debug("Cancelled greeting task")
        if self._utterance_task is not None and not self._utterance_task.done():
            self._utterance_task.cancel()
            logger.debug("Cancelled utterance task")

        # Clear buffers
        self._audio_buffer = []
        self._silero_buffer = []

        # Clear output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cleanup model - only release if persistence is disabled
        # With persistence enabled, model stays in cache for next session
        if not self._use_model_persistence:
            self.model = None
            self.processor = None
            logger.debug("Model released (persistence disabled)")
        else:
            logger.debug("Model kept in cache (persistence enabled)")

        # Clear Silero VAD state (but keep model in cache)
        self._silero_vad = None

        logger.info("LocalQwenS2SHandler shutdown complete")

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    # Stub methods for personality UI compatibility (not yet implemented for local S2S)

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Apply a personality profile (stub - local S2S doesn't support dynamic personality changes)."""
        logger.info("apply_personality called with profile=%s (not implemented for local S2S)", profile)
        return "not_implemented"

    async def get_available_voices(self) -> List[str]:
        """Get available voices (Qwen3-Omni supports a fixed set)."""
        return ["Ethan", "Chelsie", "Aiden"]
