"""Local speech-to-speech handler using Qwen3-Omni via mlx-vlm.

This handler replaces OpenAI Realtime with local inference using
Qwen3-Omni-30B-A3B running on Apple Silicon via MLX.
"""

import asyncio
import logging
import tempfile
from typing import Any, Dict, List, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)

# Sample rates - Qwen3-Omni uses 24kHz which matches OpenAI Realtime
INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000

# Default model path - can be overridden via env
DEFAULT_MODEL_PATH = "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"


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

        # VAD parameters
        self._vad_threshold = 0.02  # RMS threshold for speech detection
        self._silence_threshold_frames = 30  # ~1.25s of silence to end utterance
        self._min_speech_frames = 5  # Minimum frames to consider valid speech

        # Output queue
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        # Conversation history
        self._conversation: List[Dict[str, object]] = []

        # Lifecycle
        self._shutdown_requested = False
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()

    def copy(self) -> "LocalQwenS2SHandler":
        """Create a copy of the handler."""
        return LocalQwenS2SHandler(self.deps, self.model_path, self.speaker)

    async def start_up(self) -> None:
        """Initialize the Qwen3-Omni model."""
        logger.info("Loading Qwen3-Omni model from %s...", self.model_path)

        try:
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

            # Import mlx-vlm
            from mlx_vlm.utils import load

            # Load model and processor
            self.model, self.processor = load(
                self.model_path,
                trust_remote_code=True,
            )

            # Ensure talker is enabled for audio output
            if self.model is not None and hasattr(self.model, "enable_talker") and callable(self.model.enable_talker):
                self.model.enable_talker()

            logger.info("Qwen3-Omni model loaded successfully")
            logger.info("Using speaker: %s", self.speaker)

            # Initialize conversation with system prompt
            system_instructions = get_session_instructions()
            self._conversation = [{"role": "system", "content": system_instructions}]

            # Warm up the model to JIT compile for fast first response
            await self._warm_up_model()

            # Generate a greeting to the user (like OpenAI Realtime does on session start)
            await self.generate_greeting()

        except Exception as e:
            logger.error("Failed to load Qwen3-Omni model: %s", e)
            raise

    async def _warm_up_model(self) -> None:
        """Warm up the model with a dummy inference to JIT compile.

        This ensures the first real response is fast (~260ms TTFT instead of ~2s).
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

            # Minimal conversation for warm-up
            warmup_conv = [{"role": "user", "content": "Hi"}]
            inputs, _ = prepare_omni_inputs(self.processor, warmup_conv)
            input_ids = inputs.get("input_ids")

            # Type guard
            if self.model is None:
                logger.warning("Model is None during warm-up")
                return

            # Assert for type narrowing (model is not None after the check above)
            model = self.model

            # Run a quick generation to compile the model
            for _ in model.generate_stream(
                input_ids=input_ids,
                speaker=self.speaker,
                thinker_max_new_tokens=8,
                talker_max_new_tokens=64,
            ):
                break  # Just need first iteration to trigger compilation

            # Force evaluation
            mx.eval()

            elapsed = (time.time() - start) * 1000
            logger.info("Model warm-up complete (%.0fms)", elapsed)

        except Exception as e:
            logger.warning("Model warm-up failed (non-fatal): %s", e)

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
                thinker_max_new_tokens=256,
                talker_max_new_tokens=1024,
                chunk_size=200,
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

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone.

        Implements simple energy-based VAD to detect speech boundaries.
        """
        if self.model is None or self._shutdown_requested:
            return

        input_sample_rate, audio_frame = frame

        # Log sample rate once
        if not hasattr(self, "_input_sr_logged"):
            logger.info("Microphone sample rate: %d Hz (will resample to %d Hz)", input_sample_rate, INPUT_SAMPLE_RATE)
            self._input_sr_logged = True

        # Reshape if needed
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if INPUT_SAMPLE_RATE != input_sample_rate:
            audio_frame = np.asarray(
                resample(audio_frame, int(len(audio_frame) * INPUT_SAMPLE_RATE / input_sample_rate)), dtype=np.int16
            )

        # Convert to float for processing
        audio_float = np.asarray(audio_frame, dtype=np.float32) / 32768.0

        # Simple energy-based VAD
        rms = np.sqrt(np.mean(audio_float**2))
        is_speech = rms > self._vad_threshold

        # Log RMS periodically for debugging (every ~1 second at 24kHz with typical frame sizes)
        if not hasattr(self, "_rms_log_counter"):
            self._rms_log_counter = 0
        self._rms_log_counter += 1
        if self._rms_log_counter % 50 == 0:  # Log every ~50 frames
            logger.info("VAD: rms=%.4f, threshold=%.4f, is_speech=%s, speaking=%s", rms, self._vad_threshold, is_speech, self._is_speaking)

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
                    audio_data = np.concatenate(self._audio_buffer)
                    self._audio_buffer = []
                    asyncio.create_task(self._process_utterance(audio_data))

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
                thinker_max_new_tokens=512,
                talker_max_new_tokens=2048,
                chunk_size=200,  # Smaller chunks for faster streaming
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

        # Debug logging for audio emission (only log periodically to avoid spam)
        if result is not None and isinstance(result, tuple):
            qsize = self.output_queue.qsize()
            if qsize > 10 or qsize == 0:  # Log when queue is filling up or empty
                logger.debug("emit(): audio chunk, queue_size=%d", qsize)

        return result

    def shutdown(self) -> None:
        """Shutdown the handler (sync to match base class signature)."""
        self._shutdown_requested = True

        # Clear buffers
        self._audio_buffer = []

        # Clear output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cleanup model (free memory)
        self.model = None
        self.processor = None

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
