//! Pulse - Local speech-to-text dictation engine.

pub mod audio;
pub mod engine;
pub mod error;
pub mod platform;
pub mod providers;
pub mod storage;
pub mod types;

pub use error::{Error, Result};
pub use types::*;

/// Base data directory: `~/.local/share/pulse`.
pub fn data_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join(".local/share/pulse")
}

/// Read a WAV file into interleaved f32 samples, returning (samples, sample_rate, channels).
pub fn read_wav(path: &str) -> std::result::Result<(Vec<f32>, u32, u16), anyhow::Error> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
    };
    Ok((samples, spec.sample_rate, spec.channels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_result_types() {
        let err = Error::Audio("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }
}
