//! Audio capture and voice activity detection.

pub mod capture;
pub mod resample;
pub mod vad;

pub use capture::AudioCapture;
pub use vad::{SimpleVad, VoiceActivity, VAD_CHUNK_SIZE, VAD_SAMPLE_RATE};
