//! OpenAI cloud transcription and completion providers.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::{Error, Result};
use crate::providers::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::providers::transcription::{
    TranscriptionProvider, TranscriptionRequest, TranscriptionResponse,
};

// ── OpenAI Transcription API ─────────────────────────────────────

/// Cloud transcription via OpenAI's transcription API.
pub struct OpenAITranscriptionProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAITranscriptionProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let client = Client::new();
        Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".into()),
        }
    }

    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").ok()?;
        let base_url = std::env::var("OPENAI_BASE_URL").ok();
        Some(Self::new(api_key, base_url))
    }
}

#[async_trait]
impl TranscriptionProvider for OpenAITranscriptionProvider {
    fn name(&self) -> &'static str {
        "openai-transcription"
    }

    async fn transcribe(&self, request: TranscriptionRequest) -> Result<TranscriptionResponse> {
        let start = std::time::Instant::now();

        // Convert PCM f32 to WAV bytes for upload.
        let wav_bytes = pcm_to_wav(&request.audio, request.sample_rate, request.channels);

        let part = reqwest::multipart::Part::bytes(wav_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| Error::Transcription(format!("MIME error: {}", e)))?;

        let form = reqwest::multipart::Form::new()
            .text("model", "whisper-1")
            .text(
                "language",
                request.language.unwrap_or_else(|| "en".into()),
            )
            .part("file", part);

        let resp = self
            .client
            .post(format!(
                "{}/audio/transcriptions",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(|e| Error::Network(e))?;

        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Network(e))?;

        if !status.is_success() {
            return Err(Error::Transcription(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        #[derive(Deserialize)]
        struct ApiResponse {
            text: String,
        }

        let response: ApiResponse = serde_json::from_str(&body)?;
        let duration_ms = start.elapsed().as_millis() as u64;

        debug!(
            "[openai] transcribed in {}ms: \"{}\"",
            duration_ms, response.text
        );

        Ok(TranscriptionResponse {
            text: response.text.trim().to_string(),
            confidence: None,
            duration_ms,
        })
    }

    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }
}

// ── OpenAI Completion ──────────────────────────────────────────────

/// Cloud completion via OpenAI-compatible chat API.
pub struct OpenAICompletionProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAICompletionProvider {
    pub fn new(api_key: String, base_url: Option<String>, model: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".into()),
            model: model.unwrap_or_else(|| "gpt-4o-mini".into()),
        }
    }

    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("LLM_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .ok()?;
        let base_url = std::env::var("LLM_BASE_URL").ok();
        let model = std::env::var("LLM_MODEL").ok();
        Some(Self::new(api_key, base_url, model))
    }
}

#[async_trait]
impl CompletionProvider for OpenAICompletionProvider {
    fn name(&self) -> &'static str {
        "openai-completion"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        #[derive(Serialize)]
        struct ChatRequest {
            model: String,
            messages: Vec<Message>,
            temperature: f32,
        }

        #[derive(Serialize)]
        struct Message {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct ChatResponse {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: ResponseMessage,
        }

        #[derive(Deserialize)]
        struct ResponseMessage {
            content: String,
        }

        let body = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: request.system,
                },
                Message {
                    role: "user".into(),
                    content: request.user,
                },
            ],
            temperature: request.temperature,
        };

        let resp = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Network(e))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| Error::Network(e))?;

        if !status.is_success() {
            return Err(Error::Completion(format!(
                "API error ({}): {}",
                status,
                &text[..text.len().min(200)]
            )));
        }

        let response: ChatResponse = serde_json::from_str(&text)?;
        let content = response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        Ok(CompletionResponse {
            text: content.trim().to_string(),
        })
    }

    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }
}

// ── Utility ────────────────────────────────────────────────────────

/// Convert PCM f32 samples to WAV bytes.
fn pcm_to_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let mut buf = Vec::new();
    let cursor = std::io::Cursor::new(&mut buf);
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::new(cursor, spec).expect("WAV writer creation failed");
    for &sample in samples {
        let s16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s16).expect("WAV write failed");
    }
    writer.finalize().expect("WAV finalize failed");
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_transcription_from_env_missing() {
        // Should return None when env vars aren't set.
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        assert!(OpenAITranscriptionProvider::from_env().is_none());
    }

    #[test]
    fn test_openai_completion_is_configured() {
        let provider = OpenAICompletionProvider::new("sk-test".into(), None, None);
        assert!(provider.is_configured());

        let empty = OpenAICompletionProvider::new("".into(), None, None);
        assert!(!empty.is_configured());
    }

    #[test]
    fn test_pcm_to_wav() {
        let samples = vec![0.0f32; 16000]; // 1 second of silence.
        let wav = pcm_to_wav(&samples, 16000, 1);
        assert!(!wav.is_empty());
        // WAV header is 44 bytes + 16000 * 2 bytes (16-bit samples).
        assert_eq!(wav.len(), 44 + 16000 * 2);
    }
}
