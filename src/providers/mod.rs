//! Provider abstraction layer for transcription and completion services.

pub mod completion;
pub mod local_whisper;
pub mod moonshine;
pub mod openai;
pub mod transcription;

pub use completion::{CompletionProvider, CompletionRequest, CompletionResponse};
pub use transcription::{TranscriptionProvider, TranscriptionRequest, TranscriptionResponse};
