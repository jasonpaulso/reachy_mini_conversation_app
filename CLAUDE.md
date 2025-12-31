# Reachy Mini Conversation App

<!-- AUTO-MANAGED: project-description -->
Conversational app for the Reachy Mini robot combining real-time speech-to-speech APIs (OpenAI Realtime or local Qwen3-Omni), vision pipelines, and choreographed motion libraries. Supports both cloud-based and on-device inference.
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: build-commands -->
## Build Commands

**Setup:**
```bash
uv venv --python 3.12.1
source .venv/bin/activate
uv sync
```

**Development:**
```bash
uv run pytest                    # Run tests
uv run ruff check src/          # Lint
uv run ruff format src/         # Format
uv run mypy src/                # Type check
```

**Run:**
```bash
reachy-mini-conversation-app           # Headless mode
reachy-mini-conversation-app --gradio  # With web UI
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Architecture

```
src/reachy_mini_conversation_app/
├── main.py                    # Entry point, handler selection
├── config.py                  # Environment config (OpenAI/local S2S)
├── console.py                 # LocalStream for headless mode
├── headless_personality_ui.py # REST API for personality management
├── openai_realtime.py         # OpenAI Realtime handler
├── local_qwen_s2s.py         # Local Qwen3-Omni S2S handler
├── moves.py                   # MovementManager, motion queuing
├── prompts.py                 # System instructions
├── camera_worker.py           # Thread-safe camera with face tracking
├── audio/                     # HeadWobbler, speech reactivity
├── vision/                    # Vision processing (SmolVLM2, YOLO)
├── tools/                     # Tool calling (dance, camera, etc)
├── profiles/                  # Personality profiles with tools.txt
└── static/                    # Settings UI
```

**Handler Selection:**
- `LOCAL_S2S_ENABLED=true` → `LocalQwenS2SHandler` (on-device MLX inference)
- `LOCAL_S2S_ENABLED=false` → `OpenaiRealtimeHandler` (cloud API)

**Personality System:**
- Profiles in `profiles/` directory with `instructions.txt`, `tools.txt`, `voice.txt`
- Headless REST API at `/personalities/*` for runtime personality switching
- Tools loaded dynamically from profile's `tools.txt` (profile-local or shared fallback)
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: conventions -->
## Conventions

**Configuration:**
- `.env` file for API keys and feature flags
- `LOCAL_S2S_ENABLED`, `LOCAL_S2S_MODEL`, `LOCAL_S2S_SPEAKER` control local inference
- `LOCAL_S2S_VAD_THRESHOLD` (default 0.02) - RMS threshold for speech detection in local mode
- `LOCAL_S2S_SKIP_GREETING` (default false) - Skip greeting generation, useful for Gradio mode
- `OPENAI_API_KEY` only required when `LOCAL_S2S_ENABLED=false` (cloud mode), auto-downloaded from HuggingFace if missing
- Settings UI and API key download skipped entirely when local S2S is enabled

**Audio Processing:**
- 24kHz sample rate for both OpenAI and Qwen3-Omni
- Energy-based VAD in local mode (configurable thresholds)
- Resample via `scipy.signal.resample` with explicit `np.asarray` cast for type safety
- Debug audio files saved to `/tmp/qwen_debug_audio.wav` for troubleshooting

**Handler Interface:**
- Both handlers implement `fastrtc.AsyncStreamHandler`
- `receive()` for incoming audio, `output_queue` for outgoing audio
- `copy()` method required for handler instantiation

**Type Safety:**
- Use `assert` statements after optional type loads to satisfy type checkers
- Explicit `np.asarray(..., dtype=np.int16)` for scipy operations returning Any
- Type assertions before accessing dynamically loaded models/processors
- `TYPE_CHECKING` block for forward references to avoid circular imports (e.g., LocalQwenS2SHandler in console.py)
- Union types for dual-mode handlers: `Union[OpenaiRealtimeHandler, LocalQwenS2SHandler]`
- String type annotations for forward references: `"Union[...]"` when class not yet defined

**Tool System:**
- Tools loaded from profile's `tools.txt` (one tool name per line, comments start with `#`)
- Profile-local tools in `profiles/{profile}/{tool_name}.py` override shared tools in `tools/{tool_name}.py`
- Tool registration happens at import via `_initialize_tools()` → finds all `Tool` subclasses
- Tools receive `ToolDependencies` dataclass with robot, movement_manager, camera_worker, etc

**Import Organization (ruff isort):**
- Standard library, third-party, local (no blank lines between groups)
- 2 blank lines after imports
- Known first-party: `reachy_mini`, `reachy_mini_dances_library`, `reachy_mini_toolbox`
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: patterns -->
## Patterns

**Monkey Patching for Upstream Bugs:**
```python
# Patch transformers tokenizer bug: extra_special_tokens is a list but code expects dict
import transformers.tokenization_utils_base as tub

original_set_special = tub.PreTrainedTokenizerBase._set_model_specific_special_tokens

def patched_set_special(self: tub.PreTrainedTokenizerBase, special_tokens: object = None) -> None:
    if isinstance(special_tokens, list):
        special_tokens = {token: token for token in special_tokens}
    return original_set_special(self, special_tokens)

tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = patched_set_special  # type: ignore[method-assign]

# Patch Qwen2TokenizerFast.__getattr__ to provide fallback special token values
from transformers import Qwen2TokenizerFast

original_tokenizer_getattr = Qwen2TokenizerFast.__getattr__

SPECIAL_TOKEN_FALLBACKS = {
    "image_token": "<|image_pad|>",
    "audio_token": "<|audio_pad|>",
    "video_token": "<|video_pad|>",
    "vision_bos_token": "<|vision_start|>",
    "vision_eos_token": "<|vision_end|>",
    "audio_bos_token": "<|audio_bos|>",
    "audio_eos_token": "<|audio_eos|>",
}

def patched_tokenizer_getattr(self: Qwen2TokenizerFast, key: str) -> object:
    if key in SPECIAL_TOKEN_FALLBACKS:
        return SPECIAL_TOKEN_FALLBACKS[key]
    return original_tokenizer_getattr(self, key)

Qwen2TokenizerFast.__getattr__ = patched_tokenizer_getattr  # type: ignore[method-assign]
```
Applied during `start_up()` in LocalQwenS2SHandler to work around transformers bugs. First patch converts extra_special_tokens list to dict. Second patch provides fallback values for Qwen3OmniMoeProcessor's expected special tokens (pad tokens, vision start/end, audio start/end) that Qwen2TokenizerFast doesn't define.

**Conditional Imports:**
```python
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from reachy_mini_conversation_app.local_qwen_s2s import LocalQwenS2SHandler

# Runtime conditional import
if config.LOCAL_S2S_ENABLED:
    from reachy_mini_conversation_app.local_qwen_s2s import LocalQwenS2SHandler

# Usage in type hints
def __init__(self, handler: "Union[OpenaiRealtimeHandler, LocalQwenS2SHandler]"):
    ...
```
Lazy-load heavy dependencies (MLX, transformers) only when needed. Use TYPE_CHECKING for forward references to enable type hints without runtime imports, then conditionally import at runtime based on config flags.

**Audio Buffer Management (VAD-based):**
```python
# Resample with explicit type cast for scipy.signal.resample
if INPUT_SAMPLE_RATE != input_sample_rate:
    audio_frame = np.asarray(resample(
        audio_frame,
        int(len(audio_frame) * INPUT_SAMPLE_RATE / input_sample_rate)
    ), dtype=np.int16)

# Collect audio frames during speech
self._audio_buffer.append(audio_to_int16(audio_float))

# Process complete utterance after silence threshold
if self._silence_frames >= self._silence_threshold_frames:
    audio_data = np.concatenate(self._audio_buffer)
    self._audio_buffer = []
    asyncio.create_task(self._process_utterance(audio_data))
```
Energy-based VAD with RMS threshold (configurable via `LOCAL_S2S_VAD_THRESHOLD`, default 0.02), minimum speech frames (`_min_speech_frames=5`), and silence detection (`_silence_threshold_frames=30` for ~1.25s). Resample operations require explicit `np.asarray` cast since scipy returns Any.

**Headless Audio Lifecycle (Sequential Startup):**
```python
# CRITICAL: Wait for handler startup BEFORE starting audio loops
await handler.start_up()  # 6+ seconds for model loading
logger.info("Handler ready - flushing stale audio buffer")

# Flush stale audio accumulated during model load
for _ in range(100):  # Discard ~4 seconds at 24kHz
    self._robot.media.audio.get_audio_sample()

# NOW start audio loops - handler is ready to process
self._tasks.append(asyncio.create_task(self.record_loop()))
self._tasks.append(asyncio.create_task(self.play_loop()))
```
Sequential startup prevents audio loss: wait for `handler.start_up()` to complete (model loading), flush stale audio from buffer, then start record/play loops. Without this, audio captured during model load is lost or contains zeros.

**Background Task Management (Greeting & Utterances):**
```python
# Greeting as background task (doesn't block emit() startup)
if not config.LOCAL_S2S_SKIP_GREETING:
    self._greeting_task = asyncio.create_task(self.generate_greeting())

# Track tasks for proper shutdown
async def shut_down(self):
    self._shutdown_requested = True
    if self._greeting_task:
        self._greeting_task.cancel()
    if self._utterance_task:
        self._utterance_task.cancel()
```
Greeting generation runs in background to allow `emit()` to start consuming audio immediately (like OpenAI Realtime). Track tasks for clean cancellation on shutdown. Skip greeting via `LOCAL_S2S_SKIP_GREETING=true` for faster Gradio startup.

**Streaming Generation (Low-Latency Response):**
```python
# Prepare inputs - returns dict with input_ids + audio features
model_inputs, _ = prepare_omni_inputs(self.processor, conversation)

# CRITICAL: Extract input_ids, pass remaining features via **kwargs
input_ids = model_inputs.pop("input_ids")
# Remaining: input_features, feature_attention_mask, audio_feature_lengths

# Stream text and audio chunks (~260ms TTFT after warm-up vs ~15s batch)
for chunk_type, chunk_data in self.model.generate_stream(
    input_ids=input_ids,
    speaker=self.speaker,
    thinker_max_new_tokens=512,
    talker_max_new_tokens=2048,
    chunk_size=200,
    **model_inputs,  # Pass audio features - REQUIRED for speech understanding
):
    if chunk_type == "text":
        response_tokens.extend(chunk_data)
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": decoded_text, "partial": True})
        )
    elif chunk_type == "audio":
        audio_int16 = (np.array(chunk_data) * 32767).astype(np.int16)
        await self.output_queue.put((OUTPUT_SAMPLE_RATE, audio_int16))
```
CRITICAL: `prepare_omni_inputs()` returns audio Mel spectrogram features that MUST be passed to `generate_stream()` via `**model_inputs`. Without these, model falls back to text-only mode and speech understanding fails. Use `pop("input_ids")` to extract the required positional argument, then spread remaining features.

**Model Warm-up (JIT Compilation):**
```python
async def _warm_up_model(self):
    """Run dummy inference during start_up() to JIT compile the model."""
    # Type assertions for Pylance/mypy
    assert self.model is not None
    assert self.processor is not None

    warmup_conv = [{"role": "user", "content": "Hi"}]
    inputs, _ = prepare_omni_inputs(self.processor, warmup_conv)
    for _ in self.model.generate_stream(input_ids=inputs["input_ids"], ...):
        break  # Just need first iteration to trigger compilation
    mx.eval()  # Force evaluation
```
Pre-compile model during `start_up()` to ensure first real response is fast (~260ms TTFT instead of ~2s). Type assertions satisfy static checkers for Optional attributes.

**Stream Keepalive Pattern:**
```python
async def emit(self):
    if self._shutdown_requested:
        return None  # Signal stream end
    # Use wait_for_item helper - returns None on timeout (0.1s)
    return await wait_for_item(self.output_queue)
```
Uses fastrtc's `wait_for_item()` helper which returns None after timeout, preventing stream closure while allowing clean shutdown. Both handlers use this pattern for consistency.

**Shutdown Guards:**
```python
# Check shutdown flag throughout async operations
if self._shutdown_requested:
    logger.debug("Skipping processing - shutdown requested")
    return

# In loops: check between iterations
for chunk in self.model.generate_stream(...):
    if self._shutdown_requested:
        break
    await self.output_queue.put(chunk)
```
Prevents race conditions during connection lifecycle by checking `_shutdown_requested` at key points.

**Queue Clearing by Backend:**
```python
if self._robot.media.backend == MediaBackend.GSTREAMER:
    self._robot.media.audio.clear_player()
elif self._robot.media.backend == MediaBackend.DEFAULT:
    self._robot.media.audio.clear_output_buffer()
```
Different audio backends require different clearing methods.

**Debug Audio Logging:**
```python
# Detailed stats for troubleshooting audio issues
duration_sec = len(audio_data) / INPUT_SAMPLE_RATE
audio_float = audio_data.astype(np.float32) / 32768.0
rms = np.sqrt(np.mean(audio_float ** 2))
peak = np.max(np.abs(audio_float))
unique_vals = len(np.unique(audio_data[:1000]))  # Check variation
logger.info(
    "Processing audio: %.2fs, %d samples, RMS=%.4f, peak=%.4f, unique_vals=%d, dtype=%s, range=[%d, %d]",
    duration_sec, len(audio_data), rms, peak, unique_vals,
    audio_data.dtype, audio_data.min(), audio_data.max()
)

# Persistent debug file for manual inspection
debug_path = Path("/tmp/qwen_debug_audio.wav")
sf.write(str(debug_path), audio_float, INPUT_SAMPLE_RATE)
logger.info("Debug audio saved to: %s", debug_path)
```
Log comprehensive audio stats (RMS, peak, unique values, dtype, range) and save to `/tmp/qwen_debug_audio.wav` for offline analysis when troubleshooting VAD or inference issues.

**Camera Worker (Thread-Safe Face Tracking):**
```python
class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(self, reachy_mini: ReachyMini, head_tracker: Any = None):
        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self.face_tracking_offsets: List[float] = [0.0] * 6  # x, y, z, roll, pitch, yaw

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def working_loop(self):
        """30Hz+ camera polling with face tracking integration."""
        # Captures frames in background thread
        # Updates face_tracking_offsets when faces detected
        # Smoothly interpolates back to neutral when face lost (2s delay, 1s interpolation)
```
Runs in dedicated thread at 30Hz+, providing latest frame for tools (camera snapshot) and continuous face tracking. Thread-safe locks protect frame access. Automatically handles face-lost → neutral interpolation with configurable delays.

**Headless Personality REST API:**
```python
# Mount personality routes on FastAPI settings app
mount_personality_routes(
    app,
    handler,
    get_loop,  # Callable returning asyncio loop for cross-thread scheduling
    persist_personality=lambda profile: ...,  # Save to .env
    get_persisted_personality=lambda: ...,  # Read startup choice
)

# Endpoints:
# GET  /personalities - List available personalities + current/startup selection
# GET  /personalities/load?name=X - Load personality details (instructions, tools, voice)
# POST /personalities/save - Save new/updated personality
# POST /personalities/apply - Apply personality at runtime (live session update)
# POST /personalities/voices - List available voices for current handler
```
Enables runtime personality switching in headless mode via REST API. Personality changes trigger live `session.update()` on OpenAI Realtime connection (or model reload for local S2S). Changes persisted to `.env` file when `persist=true`.

**Profile-Based Tool Loading:**
```python
# tools.txt in profile directory
move_head
dance
play_emotion
# camera  # Commented out - not enabled for this profile

# Tool loader searches profile-local first, then shared tools
profile_tool = f"reachy_mini_conversation_app.profiles.{profile}.{tool_name}"
shared_tool = f"reachy_mini_conversation_app.tools.{tool_name}"

# Profile can override shared tools with custom implementations
# e.g., profiles/example/sweep_look.py overrides tools/sweep_look.py
```
Each profile defines enabled tools via `tools.txt`. Tools can be profile-specific (custom implementations) or shared (common library). Comments start with `#`. Tool discovery happens at startup by scanning `Tool` subclasses.
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: dependencies -->
## Dependencies

**Core:**
- `reachy_mini>=1.2.3rc1` - Robot SDK
- `fastrtc>=0.0.34` - Real-time audio streaming
- `gradio==5.50.1.dev1` - Web UI

**OpenAI Mode:**
- `openai>=2.1` - Realtime API client

**Local S2S Mode (optional):**
- `mlx-vlm>=0.3.10` - Qwen3-Omni model loading (Apple Silicon, install from git for latest)
- `soundfile` - Audio I/O
- `scipy` - Resampling

**Vision (optional):**
- `torch`, `transformers` - SmolVLM2 local vision
- `ultralytics`, `supervision` - YOLO detection
- `mediapipe==0.10.14` - Face tracking
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: git-insights -->
## Git Insights

**Headless Personality Management (fc51236 + Recent):**
- NEW: REST API for runtime personality switching in headless mode via `/personalities/*` endpoints
- Profile-based tool loading from `tools.txt` (profile-local tools override shared tools)
- Live personality application via `session.update()` for OpenAI Realtime (instant switch)
- Personality persistence to `.env` file when `persist=true` flag set
- Thread-safe camera worker with face tracking integration (30Hz+ polling)
- Type safety improvements: `TYPE_CHECKING` guards, Union types for dual-mode handlers

**Local S2S Integration (feature/local-s2s-qwen3-omni - COMPLETE):**
- Full working implementation of on-device speech-to-speech using Qwen3-Omni via MLX
- Performance: ~950ms TTFT using `generate_stream()` (down from ~15s batch mode), ~260ms after warm-up
- Handler implements `fastrtc.AsyncStreamHandler` interface matching OpenAI Realtime path
- Config-driven selection: `LOCAL_S2S_ENABLED` chooses between cloud and local inference
- JIT warm-up during `start_up()` eliminates ~2s compilation delay on first inference
- Background greeting generation (doesn't block audio loops), skippable via `LOCAL_S2S_SKIP_GREETING`
- Energy-based VAD with configurable thresholds (`LOCAL_S2S_VAD_THRESHOLD`)
- Sequential startup pattern: wait for handler.start_up() → flush stale buffer → start audio loops
- Future optimizations identified: prefix caching (65% TTFT reduction potential), model persistence, Silero VAD, tool calling integration

**Audio Feature Passing Fix (CRITICAL):**
- BUG FIX: Must pass all `model_inputs` to `generate_stream()` via `**kwargs`
- Previously only `input_ids` was passed, causing model to ignore audio Mel spectrogram features
- Root cause: `prepare_omni_inputs()` returns dict with `input_ids`, `input_features`, `feature_attention_mask`, `audio_feature_lengths`
- Fix: `input_ids = model_inputs.pop("input_ids")` then `**model_inputs` spreads remaining features
- Impact: Without this, model falls back to text-only mode and speech understanding fails completely

**Streaming Generation Refactor:**
- Switched from `model.generate()` to `model.generate_stream()` for low-latency responses
- TTFT improved from ~15s (batch) to ~260ms after warm-up (streaming with 200-token chunks)
- Added `_warm_up_model()` to JIT compile during `start_up()` for fast first response
- Progressive text transcripts (partial updates) and audio chunks queued as available
- Shutdown guards throughout to handle connection lifecycle gracefully

**fc51236 - Improve interruptions:**
- Added initial local S2S support to reduce OpenAI dependency
- Qwen3-Omni via MLX enables on-device inference at 24kHz
- Improved audio queue clearing for DEFAULT backend (not just GSTREAMER)
- Handler selection now config-driven (`LOCAL_S2S_ENABLED`)

**Key Architectural Decisions:**
- **Dual-mode S2S**: Runtime choice between cloud (OpenAI) and local (MLX) inference, identical `AsyncStreamHandler` interface
- **Profile-based tool system**: Tools loaded from `tools.txt`, profile-local override shared, dynamic discovery via `Tool` subclasses
- **Thread-safe camera worker**: Dedicated 30Hz+ polling thread with face tracking, interpolation to neutral on face-lost
- **Headless personality API**: FastAPI endpoints for runtime personality switching with live session updates

**Local S2S Integration Gotchas:**
- mlx-vlm 0.3.9 on PyPI lacks Qwen3-Omni support - MUST install from git main branch
- Requires two monkey patches for transformers/Qwen2TokenizerFast bugs (applied in start_up())
- `generate_stream()` returns `(chunk_type, chunk_data)` tuples, not `(text, audio)` pairs
- Stream keepalive handled by `wait_for_item()` helper - returns None on timeout without closing stream
- Model reloads on each Gradio session because handler is recreated per recording session
- First call after load is slow (~2s) due to JIT compilation - warm-up eliminates this
- Must check `_shutdown_requested` throughout async flows to prevent race conditions
- **Headless audio timing (CRITICAL)**: Audio loops MUST wait for `handler.start_up()` completion, then flush stale buffer before processing input (see "Headless Audio Lifecycle" pattern)
- Greeting generation runs as background task - track `_greeting_task` for clean shutdown cancellation
<!-- END AUTO-MANAGED -->

<!-- MANUAL -->
## Usage Notes

- Use `--gradio` flag for web interface with personality selection
- Headless mode serves settings UI on Reachy Mini Apps runtime
- Local S2S requires Apple Silicon for MLX acceleration
- Speaker voices: Ethan, Chelsie, Aiden (Qwen3-Omni)
<!-- END MANUAL -->
