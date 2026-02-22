//! Transcription provider trait and types.

use crate::error::Result;
use async_trait::async_trait;

/// Request to transcribe audio.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// Raw PCM f32 samples (mono, 16kHz preferred).
    pub audio: Vec<f32>,
    /// Sample rate of the audio.
    pub sample_rate: u32,
    /// Number of channels (will be downmixed to mono).
    pub channels: u16,
    /// Language hint (e.g. "en").
    pub language: Option<String>,
    /// Initial prompt for Whisper decoder conditioning.
    pub initial_prompt: Option<String>,
}

/// Response from a transcription provider.
#[derive(Debug, Clone)]
pub struct TranscriptionResponse {
    /// Transcribed text.
    pub text: String,
    /// Confidence score (0.0–1.0), if available.
    pub confidence: Option<f32>,
    /// Processing time in milliseconds.
    pub duration_ms: u64,
}

/// Trait for transcription providers (local or cloud).
#[async_trait]
pub trait TranscriptionProvider: Send + Sync {
    /// Provider name for logging/display.
    fn name(&self) -> &'static str;

    /// Transcribe audio to text.
    async fn transcribe(&self, request: TranscriptionRequest) -> Result<TranscriptionResponse>;

    /// Whether this provider is properly configured and ready to use.
    fn is_configured(&self) -> bool;
}
