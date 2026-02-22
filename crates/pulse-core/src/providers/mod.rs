//! Provider abstraction layer for transcription and completion services.

pub mod completion;
pub mod coreml_whisper;
pub mod local_whisper;
pub mod moonshine;
pub mod openai;
pub mod transcription;

pub use completion::{CompletionProvider, CompletionRequest, CompletionResponse};
pub use transcription::{TranscriptionProvider, TranscriptionRequest, TranscriptionResponse};

use crate::Result;
use coreml_whisper::CoreMLWhisperEngine;
use local_whisper::PulseEngine;
use moonshine::MoonshineEngine;

/// Unified engine dispatching to Whisper, Moonshine, or CoreML.
pub enum Engine {
    Whisper(PulseEngine),
    Moonshine(MoonshineEngine),
    CoreML(CoreMLWhisperEngine),
}

impl Engine {
    pub fn transcribe_sync(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        match self {
            Engine::Whisper(e) => e.transcribe_sync(pcm, sample_rate, channels),
            Engine::Moonshine(e) => e.transcribe_sync(pcm, sample_rate, channels),
            Engine::CoreML(e) => e.transcribe_sync(pcm, sample_rate, channels),
        }
    }
}
