# Pulse

Local speech-to-text on Apple Silicon. Record from your mic, get text back — no cloud, no API keys, no latency from network round-trips.

Uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (via [whisper-rs](https://github.com/tazz4843/whisper-rs)) with Metal GPU acceleration for inference. Model downloads automatically on first run and caches locally.

## Quick Start

```bash
cargo build --release
cargo run --release
```

Press Enter to start recording, press Enter again to stop. Transcribed text prints to stdout.

```
$ cargo run --release
Audio device: MacBook Pro Microphone (48000 Hz, 1 ch)
Audio buffers pre-allocated (5.5 MB × 2)
Loading model...
Model loaded (Metal GPU + flash attention)
Inference buffers pre-allocated (3.7 MB)
Ready. Press Enter to start recording.

Recording... Press Enter to stop.
Captured 3.2s of audio. Transcribing...
The quick brown fox jumps over the lazy dog.

Press Enter to record again.
```

### File mode

Transcribe a WAV file directly:

```bash
cargo run --release -- --file recording.wav
```

### Pipe-friendly

All status/progress goes to stderr. Only transcription output goes to stdout, so you can pipe it:

```bash
cargo run --release -- --file recording.wav 2>/dev/null | pbcopy
```

## Architecture

```
STARTUP (one-time cost)
═══════════════════════

  AudioCapture::new()              InferenceEngine::with_sample_rate(48000)
  ┌──────────────────────┐         ┌──────────────────────────────────────┐
  │ cpal stream (paused) │         │ WhisperState        ~240 MB (GPU)    │
  │ live buf   11 MB cap │         │ mono_buf             5.5 MB cap      │
  │ ready buf  11 MB cap │         │ resample_buf         1.9 MB cap      │
  └──────────────────────┘         │ resample_chunk         4 KB cap      │
                                   │ SincFixedIn resampler (pre-built)    │
  All buffers sized for 30s        └──────────────────────────────────────┘
  of audio. No allocations
  after this point.                 Model downloaded from HuggingFace on
                                    first run (~487 MB), cached at
                                    ~/.cache/huggingface/hub/


RECORDING LOOP (zero allocations per iteration)
════════════════════════════════════════════════

  ┌──────────┐       ┌────────────┐       ┌──────────┐       ┌──────────┐
  │  start() │──────►│ cpal fills │──────►│  stop()  │──────►│transcribe│
  │          │       │ live buf   │       │          │       │          │
  │ swap in  │       │ (48kHz     │       │ swap out │       │ mono →   │
  │ empty    │       │  stereo)   │       │ to ready │       │ resample │
  │ buffer   │       │            │       │          │       │ → whisper│
  └──────────┘       └────────────┘       └──────────┘       └──────────┘
       │                                      │                   │
       │            Producer                  │     Consumer      │
       └──────── (mic callback) ──────────────┘  (inference) ─────┘


DATA FLOW
═════════

  Mic (48kHz stereo)
       │
       │  cpal callback pushes f32 samples
       ▼
  ┌─────────────┐
  │  live buffer │  (pre-allocated, receives at device native rate)
  └──────┬──────┘
         │  stop() — mem::swap, no copy
         ▼
  ┌─────────────┐
  │ ready buffer │  (consumer reads via samples())
  └──────┬──────┘
         │  transcribe() reads &[f32] slice
         ▼
  ┌─────────────┐
  │  mono_buf    │  clear + refill (channels → mono, reuses capacity)
  └──────┬──────┘
         │  if sample_rate ≠ 16kHz
         ▼
  ┌──────────────┐
  │ resample_buf  │  clear + refill (rubato SincFixedIn, reuses capacity)
  └──────┬───────┘
         │  16kHz mono PCM
         ▼
  ┌──────────────┐
  │  whisper.cpp  │  Metal GPU encoder + greedy decoder
  └──────┬───────┘
         │
         ▼
       stdout      "The quick brown fox jumps over the lazy dog."


BUFFER LIFECYCLE
════════════════

  Call 1 (startup pre-allocated — already hot):
  ┌──────────────────────────────────────────────────┐
  │ audio live:     with_capacity(max) → len=0       │
  │ audio ready:    with_capacity(max) → len=0       │
  │ mono_buf:       with_capacity(max) → len=0       │
  │ resample_buf:   with_capacity(max) → len=0       │
  └──────────────────────────────────────────────────┘

  Call N (every subsequent transcription):
  ┌──────────────────────────────────────────────────┐
  │ .clear() resets len to 0, capacity unchanged     │
  │ .extend_from_slice() fills within existing cap   │
  │ mem::swap() exchanges pointers, no allocation    │
  │                                                  │
  │ Zero allocator calls. Zero copies.               │
  └──────────────────────────────────────────────────┘
```

## File Structure

```
src/
  main.rs       — CLI entry point, push-to-talk loop and --file mode
  audio.rs      — AudioCapture: mic input via cpal, double-buffer swap
  inference.rs  — InferenceEngine: whisper.cpp bindings, resampling, pre-allocated buffers
  lib.rs        — public API surface, shared read_wav utility
tests/
  e2e.rs        — end-to-end tests using macOS TTS (say + afconvert)
```

## Performance

Optimizations applied for minimal latency from "stop talking" to "text appears":

| Optimization | Effect |
|---|---|
| Metal GPU | Encoder/decoder runs on Apple Silicon GPU via ggml |
| Flash attention | Fused attention kernels, reduced memory bandwidth |
| English-only model | `ggml-small.en.bin` — skips language detection |
| Pre-allocated buffers | Zero heap allocations in the recording loop |
| Double-buffer swap | `mem::swap` instead of clone — pointer exchange, no copy |
| Pre-built resampler | `SincFixedIn` constructed once at startup |
| Greedy decoding | No beam search overhead |
| Single segment mode | Skips segmentation pass for short utterances |
| Persistent WhisperState | KV caches and compute buffers reused across calls |
| All CPU cores | `n_threads` set to `available_parallelism()` |

## Tests

Tests are `#[ignore]` because they download the Whisper model (~487 MB) on first run. The TTS test requires macOS.

```bash
# Run all e2e tests
cargo test --release -- --ignored --nocapture

# Run a single test
cargo test --release test_e2e_with_tts -- --ignored --nocapture
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) for Metal GPU acceleration
- Rust 2024 edition
- Microphone access (grant permission when prompted)
- ~487 MB disk space for the Whisper model (downloaded automatically)

## Dependencies

| Crate | Purpose |
|---|---|
| `whisper-rs` | Rust bindings to whisper.cpp (with `metal` feature) |
| `cpal` | Cross-platform audio capture |
| `rubato` | High-quality audio resampling (device rate → 16kHz) |
| `hf-hub` | HuggingFace Hub API for model downloads |
| `hound` | WAV file reading (`--file` mode and tests) |
| `anyhow` | Error handling |
