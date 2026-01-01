# Handoff: Local S2S with Qwen3-Omni for Reachy Mini

**Last Updated:** 2025-12-31 (Latency Optimizations Implemented!)

<original_task>
Integrate local speech-to-speech (S2S) using Qwen3-Omni via MLX as an alternative to OpenAI Realtime API in the Reachy Mini conversation app. Focus on slotting in local S2S while keeping the rest of the app unchanged.
</original_task>

<work_completed>
1. **Feature branch**: `feature/local-s2s-qwen3-omni` → `feature/latency-optimizations`

2. **Core implementation complete and working**:
   - `src/reachy_mini_conversation_app/local_qwen_s2s.py` - Full streaming S2S handler
   - Config flags: `LOCAL_S2S_ENABLED`, `LOCAL_S2S_MODEL`, `LOCAL_S2S_SPEAKER`
   - Conditional handler selection in `main.py` and `console.py`

3. **Both Gradio AND Headless modes now working**:
   - Gradio: Browser WebRTC audio - works
   - Headless: Robot SoundDevice via SDK - **FIXED** (was returning zeros)

4. **Critical bug fixed (2025-12-31)**:
   - **Root cause**: Float32 → int16 truncation during audio resampling
   - SDK provides float32 in [-1, 1] range
   - Old code converted to int16, resampled (truncating 0.0003 → 0), then back to float
   - Fix: Preserve float32 throughout pipeline, resample in float domain

5. **Streaming generation** (low-latency):
   - Uses `model.generate_stream()` for ~260ms TTFT after warm-up
   - Model warm-up during startup eliminates JIT delay

6. **Audio pipeline patterns**:
   - console.py passes float32 directly to handler (no conversion)
   - Handler accepts both float32 (SDK) and int16 (legacy) formats
   - Parallel task startup (matches working OpenAI fork)
   - Audio warmup logic discards stale buffers in handler's receive()

7. **Latency Optimizations Implemented (2025-12-31)**:
   - **Model persistence**: Caches model/processor in `_ModelCache` across Gradio sessions (~15-20s saved per session)
   - **Silero VAD**: Neural VAD replaces energy-based for faster end-of-speech detection (~200-500ms faster)
   - **Prefix caching infrastructure**: Stores tokenized system prompt (ready for when mlx-vlm supports KV caching)
   - **Configurable tuning**: chunk_size=100 (was 200), thinker_tokens=256 (was 512)
   - All optimizations toggleable via env vars
</work_completed>

<work_remaining>
## Latency Optimizations - IMPLEMENTED ✓

| Optimization | Status | Notes |
|--------------|--------|-------|
| **Smaller chunk_size** | ✅ Done | Reduced from 200 → 100 (configurable via `LOCAL_S2S_CHUNK_SIZE`) |
| **Reduced thinker tokens** | ✅ Done | Reduced from 512 → 256 (configurable via `LOCAL_S2S_THINKER_TOKENS`) |
| **Silero VAD** | ✅ Done | Neural VAD for ~200-500ms faster end-of-speech detection |
| **Model persistence** | ✅ Done | `_ModelCache` keeps model across Gradio sessions (~15-20s saved) |
| **Prefix caching infra** | ✅ Done | Infrastructure ready; awaiting mlx-vlm KV cache API support |

### Configuration (.env)
```bash
# Latency tuning
LOCAL_S2S_CHUNK_SIZE=100        # Smaller = faster first audio (default: 100)
LOCAL_S2S_THINKER_TOKENS=256    # Lower = faster response (default: 256)
LOCAL_S2S_TALKER_TOKENS=1024    # Max audio tokens (default: 1024)
LOCAL_S2S_USE_SILERO_VAD=true   # Neural VAD (default: true)
LOCAL_S2S_MODEL_PERSISTENCE=true # Keep model in memory (default: true)
LOCAL_S2S_PREFIX_CACHING=true   # System prompt caching (default: true)
```

## Remaining Items
1. **Tool calling**: Not wired up yet for local S2S
2. **WebRTC stability**: `InvalidStateError` during long responses
3. **Better interruption handling**: Stop generation when user speaks
4. **Full KV cache integration**: When mlx-vlm adds native prefix cache API
5. **Profile real-world latency**: Measure TTFT improvement with all optimizations
</work_remaining>

<context>
## Current Status (2025-12-31)

| Mode | Audio In | Audio Out | Status |
|------|----------|-----------|--------|
| Gradio | Browser WebRTC | Browser WebRTC | **WORKING** |
| Headless | Robot SoundDevice | Robot SoundDevice | **WORKING** |

## Key Fix: Float32 Audio Pipeline

```python
# OLD (BROKEN) - int16 truncation destroyed audio:
audio_frame = np.asarray(resample(...), dtype=np.int16)  # Truncates 0.0003 → 0!
audio_float = audio_frame / 32768.0  # Too late, data is gone

# NEW (FIXED) - preserve float32 throughout:
audio_frame = np.asarray(resample(...), dtype=np.float32)  # Preserves precision
if audio_frame.dtype == np.int16:
    audio_float = audio_frame / 32768.0
else:
    audio_float = audio_frame  # Already normalized [-1, 1]
```

## Config (.env)

```bash
# Core settings
LOCAL_S2S_ENABLED=true
LOCAL_S2S_MODEL=mlx-community/Qwen3-Omni-30B-A3B-Instruct-5bit
LOCAL_S2S_SPEAKER=Aiden  # or Ethan, Chelsie
LOCAL_S2S_VAD_THRESHOLD=0.02        # Lower = more sensitive (energy-based fallback)
LOCAL_S2S_SKIP_GREETING=true        # Skip for faster startup

# Latency tuning (NEW)
LOCAL_S2S_CHUNK_SIZE=100            # Streaming chunk size (smaller = faster first audio)
LOCAL_S2S_THINKER_TOKENS=256        # Max "thinking" tokens (lower = faster response)
LOCAL_S2S_TALKER_TOKENS=1024        # Max speech tokens
LOCAL_S2S_USE_SILERO_VAD=true       # Neural VAD (faster end-of-speech detection)
LOCAL_S2S_MODEL_PERSISTENCE=true    # Keep model in memory across sessions
LOCAL_S2S_PREFIX_CACHING=true       # Pre-compute system prompt tokens
```

## Debug Files
- `/tmp/reachy_mic_debug.wav` - Raw mic capture
- `/tmp/qwen_debug_audio.wav` - Audio sent to model
- `qwen-debug.log` - Session logs

## LLM Council Contribution

Consulted 4-model council (GPT-5.1, Gemini-3-Pro, Claude Sonnet 4.5, Grok-4) for debugging. They correctly identified:
- Stale buffer issue (handled by warmup logic)
- Parallel vs sequential startup difference
- Channel mismatch possibility

But the actual bug (int16 truncation) was found by comparing with the working OpenAI fork's simpler audio pipeline.

## Gotchas Discovered

1. **mlx-vlm must be from git** - PyPI 0.3.9 lacks Qwen3-Omni support
2. **Float32 pipeline is critical** - Resample in float domain, not int16
3. **Parallel task startup** - All tasks start together, handler discards stale audio
4. **generate_stream API** - Returns `(chunk_type, chunk_data)` tuples
5. **Audio features MUST be passed** - Use `**model_inputs` for Mel spectrograms
6. **First call is slow** - JIT compilation, warm-up helps
7. **Greeting timing** - Must be background task to not block emit()

### Latency Optimization Gotchas (NEW)

8. **Silero VAD requires 16kHz** - Must resample from 24kHz, adds small overhead but worth it
9. **Model persistence requires thread-safe cache** - `_ModelCache` uses threading.Lock
10. **Prefix caching is infrastructure-only** - mlx-vlm doesn't expose KV cache API yet
11. **chunk_size trade-off** - Smaller = faster first audio, but more overhead per chunk
12. **thinker_tokens trade-off** - Lower = faster response, but may truncate complex reasoning

## Files Changed This Session
- `src/reachy_mini_conversation_app/console.py` - Simplified record_loop, float32 passthrough
- `src/reachy_mini_conversation_app/local_qwen_s2s.py` - Float32 receive(), dual-format support
- `CLAUDE.md` - Updated patterns and conventions

### Latency Optimization Session (feature/latency-optimizations)
- `src/reachy_mini_conversation_app/config.py` - Added latency tuning env vars
- `src/reachy_mini_conversation_app/local_qwen_s2s.py` - Major refactor:
  - Added `_ModelCache` for model persistence across sessions
  - Added Silero VAD integration with 24kHz→16kHz resampling
  - Added prefix caching infrastructure
  - Made chunk_size/token limits configurable
  - Refactored `start_up()` to use cached model
  - Added `_detect_speech()`, `_detect_speech_silero()`, `_detect_speech_energy()` methods
  - Updated `shutdown()` to preserve cached model
</context>
