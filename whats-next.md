# Handoff: Local S2S with Qwen3-Omni for Reachy Mini

**Last Updated:** 2025-12-31 (Headless Mode Fixed!)

<original_task>
Integrate local speech-to-speech (S2S) using Qwen3-Omni via MLX as an alternative to OpenAI Realtime API in the Reachy Mini conversation app. Focus on slotting in local S2S while keeping the rest of the app unchanged.
</original_task>

<work_completed>
1. **Feature branch**: `feature/local-s2s-qwen3-omni`

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
</work_completed>

<work_remaining>
## Immediate Priority: Latency Optimization

Current performance is functional but slow. Optimization targets:

| Optimization | Potential Impact | Effort | Notes |
|--------------|------------------|--------|-------|
| **Prefix caching** | ~65% TTFT reduction | Medium | Pre-compute KV cache for system prompt |
| **Smaller chunk_size** | Faster first audio | Low | Reduce from 200, trade-off with overhead |
| **Silero VAD** | Faster end-of-speech | Medium | Replace energy-based VAD |
| **Model persistence** | No reload per session | Medium | Keep model in memory across Gradio sessions |

### Quick Wins to Try First:
1. Reduce `chunk_size` in `generate_stream()` from 200 to 100 or 50
2. Reduce `thinker_max_new_tokens` if responses are shorter than needed
3. Profile to identify actual bottlenecks

## Secondary Items
1. **Tool calling**: Not wired up yet for local S2S
2. **WebRTC stability**: `InvalidStateError` during long responses
3. **Better interruption handling**: Stop generation when user speaks
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
LOCAL_S2S_ENABLED=true
LOCAL_S2S_MODEL=mlx-community/Qwen3-Omni-30B-A3B-Instruct-5bit
LOCAL_S2S_SPEAKER=Aiden  # or Ethan, Chelsie
LOCAL_S2S_VAD_THRESHOLD=0.02        # Lower = more sensitive
LOCAL_S2S_SKIP_GREETING=true        # Skip for faster startup
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

## Files Changed This Session
- `src/reachy_mini_conversation_app/console.py` - Simplified record_loop, float32 passthrough
- `src/reachy_mini_conversation_app/local_qwen_s2s.py` - Float32 receive(), dual-format support
- `CLAUDE.md` - Updated patterns and conventions
</context>
