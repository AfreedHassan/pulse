//! CoreML Whisper provider via WhisperKit (persistent Swift subprocess).
//!
//! Spawns a Swift helper binary (`pulse-whisper-coreml`) that keeps the WhisperKit
//! model loaded across transcriptions. Communication uses a newline-delimited
//! text protocol over stdin/stdout — the same pattern as `indicator.rs`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

use tracing::{debug, info};

use crate::audio::resample::preprocess_audio;
use crate::error::{Error, Result};

/// CoreML model variant for WhisperKit.
#[derive(Debug, Clone, Copy)]
pub enum CoreMLModel {
    /// openai/whisper-large-v3-turbo — best speed/quality tradeoff.
    Turbo,
    /// openai/whisper-large-v3 — highest quality, slower.
    Large,
    /// distil-whisper/distil-large-v3 — distilled, fast.
    Distil,
}

impl CoreMLModel {
    /// Parse a CLI string into a model variant.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "turbo" | "large-v3-turbo" => Some(Self::Turbo),
            "large" | "large-v3" => Some(Self::Large),
            "distil" | "distil-large-v3" => Some(Self::Distil),
            _ => None,
        }
    }

    fn as_protocol_str(&self) -> &'static str {
        match self {
            Self::Turbo => "large-v3-turbo",
            Self::Large => "large-v3",
            Self::Distil => "distil-large-v3",
        }
    }
}

impl std::fmt::Display for CoreMLModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Turbo => write!(f, "large-v3-turbo"),
            Self::Large => write!(f, "large-v3"),
            Self::Distil => write!(f, "distil-large-v3"),
        }
    }
}

/// CoreML Whisper engine backed by a persistent WhisperKit subprocess.
pub struct CoreMLWhisperEngine {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    temp_dir: std::path::PathBuf,
}

impl CoreMLWhisperEngine {
    /// Create a new CoreML engine, spawning the Swift subprocess and loading the model.
    ///
    /// This downloads and compiles the CoreML model on first run (~1-2 min, ~3GB).
    pub fn new(model: CoreMLModel) -> Result<Self> {
        let binary = Self::helper_path().ok_or_else(|| {
            Error::Transcription(
                "pulse-whisper-coreml binary not found. \
                 Build with `cargo build --release` on macOS to compile the Swift helper."
                    .to_string(),
            )
        })?;

        info!("Spawning CoreML whisper helper: {:?}", binary);

        let mut child = Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // download progress visible
            .spawn()
            .map_err(|e| {
                Error::Transcription(format!("Failed to spawn whisper helper: {}", e))
            })?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        let temp_dir = std::env::temp_dir();

        let mut engine = Self {
            child,
            stdin: Some(stdin),
            stdout,
            temp_dir,
        };

        // Send load command and wait for response.
        engine.send_command(&format!("load {}", model.as_protocol_str()))?;
        let response = engine.read_response()?;

        if !response.starts_with("ok") {
            return Err(Error::Transcription(format!(
                "CoreML model load failed: {}",
                response
            )));
        }

        info!("CoreML whisper engine ready (model: {})", model);
        Ok(engine)
    }

    /// Transcribe PCM audio samples to text.
    pub fn transcribe_sync(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        let Some(audio) = preprocess_audio(pcm, sample_rate, channels) else {
            return Ok(String::new());
        };

        let num_samples = audio.len();
        let audio_duration = num_samples as f64 / 16_000.0;
        debug!(
            "[coreml] {} samples ({:.2}s) after preprocessing",
            num_samples, audio_duration
        );

        // Write preprocessed audio to a temp WAV file.
        let wav_path = self
            .temp_dir
            .join(format!("pulse-chunk-{}.wav", std::process::id()));
        write_temp_wav(&wav_path, &audio)?;

        // Send transcribe command.
        let path_str = wav_path.to_string_lossy();
        self.send_command(&format!("transcribe {}", path_str))?;
        let response = self.read_response()?;

        // Clean up temp file.
        let _ = std::fs::remove_file(&wav_path);

        if let Some(text) = response.strip_prefix("ok ") {
            let text = text.trim().to_string();
            debug!("[coreml] transcribed: {:?}", &text[..text.len().min(80)]);
            Ok(text)
        } else if response == "ok" {
            // Empty transcription.
            Ok(String::new())
        } else {
            Err(Error::Transcription(format!(
                "CoreML transcription failed: {}",
                response
            )))
        }
    }

    fn send_command(&mut self, cmd: &str) -> Result<()> {
        let stdin = self.stdin.as_mut().ok_or_else(|| {
            Error::Transcription("Whisper helper stdin closed".to_string())
        })?;
        writeln!(stdin, "{}", cmd).map_err(|e| {
            Error::Transcription(format!("Failed to send command to whisper helper: {}", e))
        })?;
        stdin.flush().map_err(|e| {
            Error::Transcription(format!("Failed to flush whisper helper stdin: {}", e))
        })?;
        Ok(())
    }

    fn read_response(&mut self) -> Result<String> {
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|e| {
            Error::Transcription(format!("Failed to read from whisper helper: {}", e))
        })?;
        Ok(line.trim().to_string())
    }

    /// Locate the helper binary next to the current executable.
    fn helper_path() -> Option<std::path::PathBuf> {
        let exe = std::env::current_exe().ok()?;
        let dir = exe.parent()?;
        let path = dir.join("pulse-whisper-coreml");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }
}

impl Drop for CoreMLWhisperEngine {
    fn drop(&mut self) {
        // Close stdin to signal the subprocess to exit.
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

/// Write preprocessed 16kHz mono f32 PCM to a temporary WAV file (16-bit).
fn write_temp_wav(path: &std::path::Path, audio: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| Error::Transcription(format!("Failed to create temp WAV: {}", e)))?;

    for &sample in audio {
        let s16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(s16)
            .map_err(|e| Error::Transcription(format!("Failed to write WAV sample: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| Error::Transcription(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_model_parse() {
        assert!(matches!(CoreMLModel::parse("turbo"), Some(CoreMLModel::Turbo)));
        assert!(matches!(
            CoreMLModel::parse("large-v3-turbo"),
            Some(CoreMLModel::Turbo)
        ));
        assert!(matches!(CoreMLModel::parse("large"), Some(CoreMLModel::Large)));
        assert!(matches!(CoreMLModel::parse("large-v3"), Some(CoreMLModel::Large)));
        assert!(matches!(CoreMLModel::parse("distil"), Some(CoreMLModel::Distil)));
        assert!(matches!(
            CoreMLModel::parse("distil-large-v3"),
            Some(CoreMLModel::Distil)
        ));
        assert!(CoreMLModel::parse("unknown").is_none());
        assert!(CoreMLModel::parse("").is_none());
    }

    #[test]
    fn test_coreml_model_display() {
        assert_eq!(CoreMLModel::Turbo.to_string(), "large-v3-turbo");
        assert_eq!(CoreMLModel::Large.to_string(), "large-v3");
        assert_eq!(CoreMLModel::Distil.to_string(), "distil-large-v3");
    }

    #[test]
    fn test_coreml_model_protocol_str() {
        assert_eq!(CoreMLModel::Turbo.as_protocol_str(), "large-v3-turbo");
        assert_eq!(CoreMLModel::Large.as_protocol_str(), "large-v3");
        assert_eq!(CoreMLModel::Distil.as_protocol_str(), "distil-large-v3");
    }
}
