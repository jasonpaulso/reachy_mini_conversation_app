# Handoff: Local S2S with Qwen3-Omni for Reachy Mini

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
</work_completed>

<work_remaining>
**Original task is COMPLETE** - Local S2S is working end-to-end.

**Future optimizations to revisit**:
1. **Prefix caching**: Pre-compute KV cache for system prompt (~65% TTFT reduction possible, like cascading pipeline achieved 280ms)
2. **Speculative decoding**: Could further reduce latency
3. **WebRTC stability**: Still seeing `InvalidStateError` when connection closes during response - audio may cut off
4. **Model persistence**: Currently reloads on each Gradio session - could cache globally
5. **Silero VAD**: Replace energy-based VAD with more robust speech detection
6. **Tool calling**: Not wired up in local handler yet
</work_remaining>

<context>
## Key Architecture

- **Handler interface**: `LocalQwenS2SHandler` extends `fastrtc.AsyncStreamHandler`
- **Sample rates**: 24kHz for I/O, Qwen3-Omni feature extractor expects 16kHz (auto-resampled)
- **Streaming API**: `generate_stream()` yields `(chunk_type, chunk_data)` tuples:
  - `"text"`: `chunk_data` is list of token IDs (decode with `processor.decode()`)
  - `"audio"`: `chunk_data` is audio array

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

Without `**model_inputs`, the audio Mel spectrogram features are ignored and speech understanding fails completely.

## Config (.env)

```bash
LOCAL_S2S_ENABLED=true
LOCAL_S2S_MODEL=mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit
LOCAL_S2S_SPEAKER=Ethan  # or Chelsie, Aiden
```

## Gotchas Discovered

1. **mlx-vlm must be from git** - PyPI 0.3.9 doesn't have Qwen3-Omni support
2. **generate_stream API is different** - returns (type, data) not (text, audio)
3. **Audio features MUST be passed** - model ignores audio without them
4. **Gradio closes connection during inference** - need shutdown guards
5. **First call after load is slow** - JIT compilation, warm-up helps
</context>
