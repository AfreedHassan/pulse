//! Typed error hierarchy for Pulse
//!
//! Provides structured errors instead of stringly-typed `anyhow` everywhere.

use thiserror::Error;

/// Result type alias using Pulse's Error type.
pub type Result<T> = std::result::Result<T, Error>;

/// All possible errors in Pulse.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Transcription failed: {0}")]
    Transcription(String),

    #[error("Completion failed: {0}")]
    Completion(String),

    #[error("Storage error: {0}")]
    Storage(#[from] rusqlite::Error),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("VAD error: {0}")]
    Vad(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = Error::Audio("mic not found".into());
        assert_eq!(e.to_string(), "Audio error: mic not found");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let e: Error = io_err.into();
        assert!(matches!(e, Error::Io(_)));
        assert!(e.to_string().contains("file missing"));
    }

    #[test]
    fn test_error_variants() {
        let errors: Vec<Error> = vec![
            Error::Audio("test".into()),
            Error::Transcription("test".into()),
            Error::Completion("test".into()),
            Error::Config("test".into()),
            Error::Vad("test".into()),
        ];
        for e in &errors {
            assert!(!e.to_string().is_empty());
        }
    }
}
