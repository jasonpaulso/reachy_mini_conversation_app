# Handoff: Local S2S with Qwen3-Omni for Reachy Mini

**Last Updated:** 2025-12-31 (Debugging Session)

<original_task>
Integrate local speech-to-speech (S2S) using Qwen3-Omni via MLX as an alternative to OpenAI Realtime API in the Reachy Mini conversation app. Focus on slotting in local S2S while keeping the rest of the app unchanged.
</original_task>

<work_completed>
1. **Feature branch**: `feature/local-s2s-qwen3-omni`

2. **Core implementation complete and working**:
   - `src/reachy_mini_conversation_app/local_qwen_s2s.py` - Full streaming S2S handler
   - Config flags: `LOCAL_S2S_ENABLED`, `LOCAL_S2S_MODEL`, `LOCAL_S2S_SPEAKER`
   - Conditional handler selection in `main.py` and `console.py`

3. **Streaming generation implemented** (was batch before):
   - Uses `model.generate_stream()` for low-latency responses
   - TTFT ~950ms (down from ~15s batch mode)
   - Streams text + audio chunks as they become available

4. **Critical bugs fixed**:
   - **Audio features bug**: Now passes `**model_inputs` to `generate_stream()` (not just `input_ids`)
   - **Stream keepalive**: Returns silent frames when queue empty to prevent Gradio closing
   - **Shutdown guards**: Checks `_shutdown_requested` throughout to handle connection lifecycle

5. **Model warm-up**: JIT compiles on startup for fast first response

6. **Debug tooling added**:
   - Logs microphone sample rate, audio stats (RMS, peak, unique values)
   - Saves debug audio to `/tmp/qwen_debug_audio.wav`

7. **Session 2025-12-31 - Type errors & audio debugging**:
   - Fixed 80+ type errors across 13 source files
   - Added `Optional[Any]` annotations for model/processor (lines 62-63)
   - **Gradio mode now FULLY WORKING** - audio input AND output functional
   - Added config: `LOCAL_S2S_VAD_THRESHOLD`, `LOCAL_S2S_SKIP_GREETING`
   - Added audio warmup logic to discard stale buffers (lines 331-351)
   - Greeting generation moved to background task (line 172-176)
   - Task tracking for proper shutdown cancellation
   - Direct SoundDevice diagnostic test in console.py (lines 511-524)
</work_completed>

<work_remaining>
## Immediate Priority: Headless Mode Audio

**Gradio mode works.** Headless mode does NOT.

**Problem:** Robot SDK's `get_audio_sample()` returns ALL ZEROS
- Direct SoundDevice test: **PASSES** (microphone works)
- Robot SDK wrapper: Returns zeros
- Issue is in `reachy_mini` SDK's audio wrapper, not SoundDevice

**Investigation needed:**
- Check `reachy_mini.media.audio_base` module
- Look at how `get_audio_sample()` captures from SoundDevice
- May need to configure input device explicitly
- Or bypass robot SDK and use SoundDevice directly

**Relevant log pattern:**
```
Direct SoundDevice test: PASSED (nonzero=..., rms=...)
Raw audio: shape=(94500, 2), min=0.000000, max=0.000000, nonzero=0/189000  # ALL ZEROS
```

## Secondary: Performance Optimization
- TTFT: ~1.5-4.5 seconds (goal: <1s)
- Total response: ~7-10 seconds
- Options: prefix caching, model quantization, smaller chunks

## Tertiary: Other Items
1. **Prefix caching**: Pre-compute KV cache for system prompt (~65% TTFT reduction)
2. **WebRTC stability**: `InvalidStateError` during long responses
3. **Model persistence**: Reloads on each Gradio session
4. **Silero VAD**: Replace energy-based VAD
5. **Tool calling**: Not wired up yet
6. **OpenAI Realtime headless regression**: Same audio input issue
</work_remaining>

<context>
## Current Status (2025-12-31)

| Mode | Audio In | Audio Out | Status |
|------|----------|-----------|--------|
| Gradio | Browser WebRTC | Browser WebRTC | **WORKING** |
| Headless | Robot SoundDevice | Robot SoundDevice | **BROKEN** (zeros) |

## Key Architecture

```
Gradio:   Browser Mic → WebRTC → fastrtc → handler.receive() → WORKS
Headless: Robot Mic → SoundDevice → reachy_mini SDK → handler.receive() → ZEROS
```

## Config (.env)

```bash
LOCAL_S2S_ENABLED=true
LOCAL_S2S_MODEL=mlx-community/Qwen3-Omni-30B-A3B-Instruct-5bit
LOCAL_S2S_SPEAKER=Aiden  # or Ethan, Chelsie
LOCAL_S2S_VAD_THRESHOLD=0.02        # Lower = more sensitive
LOCAL_S2S_SKIP_GREETING=true        # Skip for faster Gradio startup
```

## Debug Files
- `/tmp/reachy_mic_debug.wav` - Raw mic capture from console.py
- `/tmp/qwen_debug_audio.wav` - Audio sent to model
- `qwen-debug.log`, `qwen-gradio-debug.log` - Session logs

## Critical Pattern (Bug Fix)

```python
model_inputs, _ = prepare_omni_inputs(processor, conversation)
input_ids = model_inputs.pop("input_ids")  # Extract input_ids
# MUST pass remaining features via **kwargs!
for chunk_type, chunk_data in model.generate_stream(
    input_ids=input_ids,
    **model_inputs,  # Contains input_features, feature_attention_mask, audio_feature_lengths
):
```

Without `**model_inputs`, audio Mel spectrogram features are ignored and speech understanding fails.

## Gotchas Discovered

1. **mlx-vlm must be from git** - PyPI 0.3.9 doesn't have Qwen3-Omni support
2. **generate_stream API** - returns `(type, data)` not `(text, audio)`
3. **Audio features MUST be passed** - model ignores audio without them
4. **Gradio closes connection during inference** - need shutdown guards
5. **First call after load is slow** - JIT compilation, warm-up helps
6. **Stale buffer issue** - Model loading takes ~6s, audio piles up as zeros
7. **WebRTC timeout** - Long operations (>10s) cause channel to die
8. **Shape mismatch** - Device reports 1 channel but SDK returns 2-channel array
9. **Greeting timing** - Must be background task in Gradio mode

## Files Changed This Session
- `src/reachy_mini_conversation_app/local_qwen_s2s.py`
- `src/reachy_mini_conversation_app/console.py`
- `src/reachy_mini_conversation_app/config.py`

## Councly Insight
Used council hearing for debugging. Key finding: Headless audio zeros is a "capture-path boundary mismatch" in the robot SDK, not a model or SoundDevice issue.
</context>
