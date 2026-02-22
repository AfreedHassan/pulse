use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use std::sync::Arc;

/// Max recording duration in seconds. Buffers are pre-allocated to this size.
const MAX_RECORDING_SECS: usize = 30;

/// Target sample rate for speech. Higher rates waste buffer space and resampling work.
const TARGET_RATE: u32 = 16_000;

pub struct AudioCapture {
    stream: cpal::Stream,
    /// Buffer receiving samples from the cpal callback.
    live: Arc<Mutex<Vec<f32>>>,
    /// Holds captured audio after `stop()`, available via `samples()`.
    ready: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
}

impl Default for AudioCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioCapture {
    pub fn new() -> Self {
        let host = cpal::default_host();
        let device = host.default_input_device().expect("No input device found");

        // Pick the lowest sample rate >= 16kHz. Speech only needs 16kHz;
        // capturing at 96kHz wastes buffer space and resampling work.
        let supported_config = {
            let mut configs: Vec<_> = device
                .supported_input_configs()
                .expect("No supported input configs")
                .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
                .collect();
            assert!(!configs.is_empty(), "No f32 input config found");
            configs.sort_by_key(|c| c.min_sample_rate());
            configs
                .iter()
                .find_map(|c| {
                    let min = c.min_sample_rate();
                    let max = c.max_sample_rate();
                    if max >= TARGET_RATE {
                        let rate = min.max(TARGET_RATE);
                        Some(c.clone().with_sample_rate(rate))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| configs.last().unwrap().clone().with_max_sample_rate())
        };

        let sample_rate = supported_config.sample_rate();
        let channels = supported_config.channels();
        let stream_config = supported_config.config();

        #[allow(deprecated)] // cpal's `name()` — uid-based replacement not yet stable
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        eprintln!(
            "Audio device: {} ({} Hz, {} ch)",
            device_name, sample_rate, channels
        );

        // Buffer is always mono — convert in the callback to avoid
        // downstream issues (VAD energy, sample counts, resampling).
        let max_samples = MAX_RECORDING_SECS * sample_rate as usize;
        let live = Arc::new(Mutex::new(Vec::<f32>::with_capacity(max_samples)));
        let ready = Arc::new(Mutex::new(Vec::<f32>::with_capacity(max_samples)));
        let live_clone = Arc::clone(&live);
        let ch = channels as usize;

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buf = live_clone.lock();
                    if ch == 1 {
                        buf.extend_from_slice(data);
                    } else {
                        // Mix multi-channel frames down to mono.
                        for frame in data.chunks(ch) {
                            buf.push(frame.iter().sum::<f32>() / ch as f32);
                        }
                    }
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )
            .expect("Failed to build input stream");

        stream.pause().expect("Failed to pause stream");

        eprintln!(
            "Audio buffers pre-allocated ({:.1} MB × 2)",
            max_samples as f64 * 4.0 / 1_048_576.0
        );

        Self {
            stream,
            live,
            ready,
            sample_rate,
            channels: 1, // buffer is always mono after callback mixing
        }
    }

    /// Start recording. Clears both buffers.
    pub fn start(&self) {
        self.ready.lock().clear();
        self.live.lock().clear();
        self.stream.play().expect("Failed to start stream");
    }

    /// Stop recording. Moves the filled live buffer into `ready` for reading via `samples()`.
    pub fn stop(&self) {
        self.stream.pause().expect("Failed to pause stream");
        let mut live = self.live.lock();
        let mut ready = self.ready.lock();
        ready.clear();
        std::mem::swap(&mut *live, &mut *ready);
    }

    /// Returns a clone of the captured samples from the last `stop()` call.
    pub fn samples(&self) -> Vec<f32> {
        self.ready.lock().clone()
    }

    /// Returns a snapshot of the live buffer length (for VAD polling without draining).
    pub fn live_len(&self) -> usize {
        self.live.lock().len()
    }

    /// Drain the first `count` samples from the live buffer and return them.
    /// Used by VAD to pull completed utterances while still recording.
    pub fn drain(&self, count: usize) -> Vec<f32> {
        let mut live = self.live.lock();
        let n = count.min(live.len());
        live.drain(..n).collect()
    }

    /// Snapshot the last `window` samples from the live buffer (no drain).
    /// Returns fewer samples if the buffer is shorter than `window`.
    pub fn peek_tail(&self, window: usize) -> Vec<f32> {
        let live = self.live.lock();
        let start = live.len().saturating_sub(window);
        live[start..].to_vec()
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }
}
