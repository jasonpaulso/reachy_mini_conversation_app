"""Custom AudioInterface for ElevenLabs Conversational AI.

Bridges ElevenLabs' blocking audio I/O model to fastrtc's async queue-based system.
"""

import asyncio
import logging
import threading
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample


logger = logging.getLogger(__name__)

ELEVENLABS_SAMPLE_RATE = 16000


class ReachyAudioInterface:
    """Bridge ElevenLabs blocking I/O to fastrtc async queues.

    ElevenLabs SDK expects:
    - start(input_callback): Begin receiving mic audio via callback
    - stop(): Stop audio capture
    - output(audio_data): Called by SDK with agent audio output

    This implementation:
    - Reads from an async input queue (mic frames from handler.receive())
    - Writes to an async output queue (for handler.emit() to consume)
    - Feeds audio to HeadWobbler for speech-reactive head movement
    """

    def __init__(self, sample_rate: int = ELEVENLABS_SAMPLE_RATE, head_wobbler: Any = None):
        """Initialize the audio interface.

        Args:
            sample_rate: Target sample rate for ElevenLabs (default 16kHz)
            head_wobbler: Optional HeadWobbler for speech-reactive head movement

        """
        self._sample_rate = sample_rate
        self._head_wobbler = head_wobbler
        self._input_queue: asyncio.Queue[bytes] | None = None
        self._output_queue: asyncio.Queue[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input_callback: Callable[[bytes], None] | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def set_queues(
        self,
        input_queue: asyncio.Queue[bytes],
        output_queue: asyncio.Queue[Any],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Set the async queues and event loop for cross-thread communication.

        Args:
            input_queue: Queue of PCM bytes from microphone (handler.receive())
            output_queue: Queue for agent audio output (consumed by handler.emit())
            loop: Event loop for scheduling async operations from threads

        """
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._loop = loop

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        """Start audio capture. Called by ElevenLabs SDK when conversation starts.

        Args:
            input_callback: Function to call with PCM audio bytes from microphone

        """
        self._input_callback = input_callback
        self._running = True
        self._thread = threading.Thread(target=self._audio_input_loop, daemon=True, name="elevenlabs-audio-input")
        self._thread.start()
        logger.info("ElevenLabs audio interface started")

    def stop(self) -> None:
        """Stop audio capture. Called by ElevenLabs SDK when conversation ends."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("ElevenLabs audio interface stopped")

    def output(self, audio_data: bytes) -> None:
        """Handle agent audio output. Called by ElevenLabs SDK with PCM audio.

        Args:
            audio_data: PCM int16 audio bytes from agent at 16kHz

        """
        if not self._output_queue or not self._loop:
            logger.warning("Output queue not configured, dropping audio")
            return

        # Feed audio to HeadWobbler for speech-reactive head movement
        if self._head_wobbler is not None:
            try:
                self._head_wobbler.feed_bytes(audio_data, self._sample_rate)
            except Exception as e:
                logger.debug("HeadWobbler feed error: %s", e)

        audio_np = np.frombuffer(audio_data, dtype=np.int16).reshape(1, -1)

        def put_audio() -> None:
            try:
                self._output_queue.put_nowait((self._sample_rate, audio_np))
            except asyncio.QueueFull:
                logger.warning("Output queue full, dropping audio frame")

        self._loop.call_soon_threadsafe(put_audio)

    def interrupt(self) -> None:
        """Handle interruption. Called by ElevenLabs SDK when user interrupts agent."""
        logger.debug("ElevenLabs conversation interrupted")

    def _audio_input_loop(self) -> None:
        """Background thread reading from input queue and calling SDK callback."""
        if not self._input_queue or not self._loop:
            logger.error("Input queue or loop not configured")
            return

        while self._running:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    asyncio.wait_for(self._input_queue.get(), timeout=0.1),
                    self._loop,
                )
                audio_bytes = future.result(timeout=0.2)

                if self._input_callback and audio_bytes:
                    self._input_callback(audio_bytes)

            except asyncio.TimeoutError:
                continue
            except TimeoutError:
                continue
            except Exception as e:
                if self._running:
                    logger.debug("Audio input loop error (expected during shutdown): %s", e)
                continue


def convert_audio_for_elevenlabs(
    audio_frame: NDArray[Any],
    input_sample_rate: int,
    target_sample_rate: int = ELEVENLABS_SAMPLE_RATE,
) -> bytes:
    """Convert audio frame to PCM bytes for ElevenLabs.

    Args:
        audio_frame: Audio data (float32 or int16)
        input_sample_rate: Sample rate of input audio
        target_sample_rate: Target sample rate (default 16kHz for ElevenLabs)

    Returns:
        PCM int16 bytes at target sample rate

    """
    if audio_frame.ndim == 2:
        if audio_frame.shape[1] > audio_frame.shape[0]:
            audio_frame = audio_frame.T
        if audio_frame.shape[1] > 1:
            audio_frame = audio_frame[:, 0]
        audio_frame = audio_frame.flatten()

    if input_sample_rate != target_sample_rate:
        num_samples = int(len(audio_frame) * target_sample_rate / input_sample_rate)
        audio_frame = np.asarray(resample(audio_frame, num_samples), dtype=np.float32)

    if audio_frame.dtype == np.float32 or audio_frame.dtype == np.float64:
        audio_int16 = (audio_frame * 32767).astype(np.int16)
    else:
        audio_int16 = audio_frame.astype(np.int16)

    return audio_int16.tobytes()
