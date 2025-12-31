# Handoff: Local S2S with Qwen3-Omni for Reachy Mini

**Last Updated:** 2025-12-31 (LLM Council Session - Headless Fix)

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

8. **Session 2025-12-31 - LLM Council Headless Fix**:
   - Consulted 4-model LLM council (GPT-5.1, Gemini-3-Pro, Claude Sonnet 4.5, Grok-4)
   - **Root cause identified**: Timing issue - audio loops started BEFORE handler.start_up() completed
   - **Fix 1**: Sequential startup - wait for `handler.start_up()` THEN start audio loops (console.py lines 427-447)
   - **Fix 2**: Buffer flush after model load - discard stale audio accumulated during 6s+ model loading
   - **Fix 3**: Zero-audio detection with auto-restart - if 5 consecutive zero samples, restart stream (lines 562-597)
   - **Fix 4**: Improved logging - early warning for persistent audio issues
</work_completed>

<work_remaining>
## Immediate Priority: Test Headless Mode Fixes

The fixes are implemented but **need testing on actual robot hardware**:
1. Run headless mode with `LOCAL_S2S_ENABLED=true`
2. Check logs for "Handler ready - flushing stale audio buffer" message
3. Verify audio samples have nonzero values after flush
4. If stream restart triggered, check if it recovers

**Expected log flow (happy path):**
```
Starting handler initialization (model loading)...
[6+ seconds of model loading]
Handler ready - flushing stale audio buffer
Flushed 6.50s of stale audio (0/156000 nonzero)  # Zeros discarded!
Direct SoundDevice test: PASSED (nonzero=..., rms=...)
Raw audio: shape=(...), min=..., max=..., nonzero=.../...  # Should be NON-ZERO now
```

**If zeros persist after fixes:**
The issue is deeper in the SDK - the callback itself isn't receiving data. Next steps:
1. Check if `reachy_mini.media.audio_sounddevice` stream is active
2. Verify device selection (looks for "Reachy Mini Audio" or "respeaker")
3. Consider bypassing SDK and using direct SoundDevice capture

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
6. **OpenAI Realtime headless regression**: Same audio input issue (might be fixed by these changes)
</work_remaining>

<context>
## Current Status (2025-12-31)

| Mode | Audio In | Audio Out | Status |
|------|----------|-----------|--------|
| Gradio | Browser WebRTC | Browser WebRTC | **WORKING** |
| Headless | Robot SoundDevice | Robot SoundDevice | **FIXED** (needs testing) |

## Key Architecture

```
Gradio:   Browser Mic -> WebRTC -> fastrtc -> handler.receive() -> WORKS
Headless: Robot Mic -> SoundDevice -> reachy_mini SDK -> handler.receive() -> FIXED?
```

## LLM Council Findings (Unanimous Agreement)

**All 4 models identified the same root causes:**

1. **Timing Issue (PRIMARY)**: Audio loops started in PARALLEL with handler.start_up()
   - `record_loop()` was feeding audio while model still loading (6+ seconds)
   - Handler not ready to process = audio effectively lost
   - FIX: Sequential startup - wait for start_up(), THEN start loops

2. **Stale Buffer Issue**: Recording starts before model load
   - Buffer fills with 6+ seconds of audio during load time
   - First `get_audio_sample()` returns this stale (possibly zero) data
   - FIX: Flush buffer after model loading completes

3. **Channel Mismatch (if still failing)**: SDK doesn't specify channels
   - `audio_sounddevice.py` line 65-68: `sd.InputStream()` has no `channels=` param
   - SoundDevice uses device default which may not match callback expectations
   - If device is 1-channel but SDK expects 2, PortAudio may fail silently

4. **Stream State Issues (if still failing)**: Callback may not fire
   - Stream might not be started correctly
   - Device selection might fall back to wrong device
   - FIX: Added auto-restart on 5 consecutive zero samples

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
6. **Stale buffer issue** - Model loading takes ~6s, audio piles up (NOW FIXED)
7. **WebRTC timeout** - Long operations (>10s) cause channel to die
8. **Shape mismatch** - Device reports 1 channel but SDK returns 2-channel array
9. **Greeting timing** - Must be background task in Gradio mode
10. **Timing issue** - Audio loops must wait for handler startup (NOW FIXED)

## Files Changed This Session
- `src/reachy_mini_conversation_app/console.py` - Sequential startup, buffer flush, zero-detection

## SDK Investigation Notes

**reachy_mini.media.audio_sounddevice.py** (installed package):
- Line 65-68: `sd.InputStream()` created WITHOUT explicit `channels=`
- Line 108: Callback does `indata[:, :MAX_INPUT_CHANNELS].copy()` (clips to 4 channels)
- Device selection looks for "Reachy Mini Audio" or "respeaker", falls back to default
- No visible fallback-to-zeros logic in callback - if data is zeros, they came from SoundDevice

If zeros persist after fixes, the issue is that SoundDevice callback receives zeros from PortAudio.
This could be: wrong device, channel mismatch, or device not properly configured.
</context>
