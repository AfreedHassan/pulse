//! Local Pulse transcription engine using Candle (pure Rust ML) with Whisper models.
//!
//! Supports multiple model tiers with Metal GPU acceleration on Apple Silicon.

use async_trait::async_trait;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use hf_hub::api::sync::Api;
use parking_lot::Mutex;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::audio::resample::preprocess_audio;
use crate::error::{Error, Result};
use crate::providers::transcription::{
    TranscriptionProvider, TranscriptionRequest, TranscriptionResponse,
};

/// Maximum decoder tokens before stopping.
const MAX_DECODE_TOKENS: usize = 224;

/// Temperature schedule for fallback decoding (OpenAI Whisper default).
/// Try greedy first (0.0), escalate on degenerate output.
const TEMPERATURE_SCHEDULE: &[f32] = &[0.0, 0.2, 0.4, 0.6, 0.8];

/// Compression ratio threshold — if token count / unique token count exceeds this,
/// the output is considered degenerate (excessive repetition).
const MAX_COMPRESSION_RATIO: f32 = 2.4;


// ── Model tiers ────────────────────────────────────────────────────

/// Available Pulse model tiers with different speed/quality tradeoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulseModel {
    /// tiny.en — fast, small (~150MB).
    Fast,
    /// base.en — balanced speed/quality (~290MB).
    Balanced,
    /// small.en — high quality (~950MB).
    Quality,
    /// medium.en — very high quality (~1.5GB).
    Medium,
    /// large-v3 — best quality, multilingual (~3GB).
    Large,
    /// distil-large-v3.5 — distilled large, 1.5x faster than large-v3 (~1.5GB).
    DistilLarge,
}

impl PulseModel {
    /// HuggingFace repo for this model.
    fn repo_id(&self) -> &'static str {
        match self {
            Self::Fast => "openai/whisper-tiny.en",
            Self::Balanced => "openai/whisper-base.en",
            Self::Quality => "openai/whisper-small.en",
            Self::Medium => "openai/whisper-medium.en",
            Self::Large => "openai/whisper-large-v3",
            Self::DistilLarge => "distil-whisper/distil-large-v3.5",
        }
    }

    /// Parse from CLI string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fast" | "tiny" => Some(Self::Fast),
            "balanced" | "base" => Some(Self::Balanced),
            "quality" | "small" => Some(Self::Quality),
            "medium" => Some(Self::Medium),
            "large" | "large-v3" => Some(Self::Large),
            "distil" | "distil-large" | "distil-large-v3.5" => Some(Self::DistilLarge),
            _ => None,
        }
    }
}

impl std::fmt::Display for PulseModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fast => write!(f, "fast (tiny.en)"),
            Self::Balanced => write!(f, "balanced (base.en)"),
            Self::Quality => write!(f, "quality (small.en)"),
            Self::Medium => write!(f, "medium (medium.en)"),
            Self::Large => write!(f, "large (large-v3)"),
            Self::DistilLarge => write!(f, "distil (distil-large-v3.5)"),
        }
    }
}

// ── Engine ─────────────────────────────────────────────────────────

/// Candle-based Pulse inference engine with Metal GPU acceleration.
pub struct PulseEngine {
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
    mel_filters: Vec<f32>,
    multilingual: bool,
}

impl PulseEngine {
    /// Build the engine with the specified model tier.
    pub fn new(model_tier: PulseModel) -> Result<Self> {
        // Prefer Metal GPU on Apple Silicon, fall back to CPU.
        let device = Device::new_metal(0).unwrap_or_else(|e| {
            warn!("Metal device not available ({}), using CPU", e);
            Device::Cpu
        });
        info!("Using device: {:?}", device);

        let repo_id = model_tier.repo_id();

        let api = Api::new()
            .map_err(|e| Error::Transcription(format!("HF API init failed: {}", e)))?;

        info!("Downloading/loading model: {} ({})", model_tier, repo_id);
        let repo = api.model(repo_id.to_string());

        // Download and parse config.json to get model architecture.
        let config_path = repo
            .get("config.json")
            .map_err(|e| Error::Transcription(format!("Failed to download config: {}", e)))?;
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Transcription(format!("Failed to read config: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| Error::Transcription(format!("Failed to parse config: {}", e)))?;

        // Download model weights.
        let model_path = repo
            .get("model.safetensors")
            .map_err(|e| Error::Transcription(format!("Failed to download model: {}", e)))?;
        info!("Model path: {:?}", model_path);

        // Download tokenizer.
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Transcription(format!("Failed to download tokenizer: {}", e)))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::Transcription(format!("Failed to load tokenizer: {}", e)))?;

        // Load model weights.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &device)
                .map_err(|e| Error::Transcription(format!("Failed to load weights: {}", e)))?
        };

        let model = m::model::Whisper::load(&vb, config.clone())
            .map_err(|e| Error::Transcription(format!("Failed to build model: {}", e)))?;

        // Load mel filters (80 bins for tiny/base/small/medium).
        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("../../melfilters.bytes").as_slice(),
            128 => include_bytes!("../../melfilters128.bytes").as_slice(),
            _ => {
                return Err(Error::Transcription(format!(
                    "Unexpected mel bins: {}",
                    config.num_mel_bins
                )));
            }
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        // Multilingual models (large-v3, distil-large) need language + task tokens.
        let multilingual = matches!(model_tier, PulseModel::Large | PulseModel::DistilLarge);

        info!("Model loaded successfully: {}", model_tier);
        eprintln!("  -> decoder_layers: {}", config.decoder_layers);
        eprintln!("  -> num_mel_bins: {}", config.num_mel_bins);
        eprintln!("  -> d_model: {}", config.d_model);
        if multilingual {
            eprintln!("  -> multilingual: true (using en/transcribe prompt)");
        }

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            mel_filters,
            multilingual,
        })
    }

    /// Transcribe PCM audio samples to text.
    pub fn transcribe_sync(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        // Shared preprocessing: mono, resample, silence gate, trim, normalize.
        let trimmed = match preprocess_audio(pcm, sample_rate, channels) {
            Some(t) => t,
            None => return Ok(String::new()),
        };

        // Compute mel spectrogram for the full audio.
        // pcm_to_mel pads the audio internally, so total_frames may exceed
        // the actual content length. Track the real content boundary.
        let content_frames_from_pcm = trimmed.len() / m::HOP_LENGTH;
        let mel_all = m::audio::pcm_to_mel(&self.config, &trimmed, &self.mel_filters);
        let total_frames = mel_all.len() / self.config.num_mel_bins;

        // Build full mel tensor: shape (1, num_mel_bins, total_frames).
        let mel_full = Tensor::from_vec(
            mel_all,
            (1, self.config.num_mel_bins, total_frames),
            &self.device,
        )
        .map_err(|e| Error::Transcription(format!("Mel tensor: {}", e)))?;

        // Pre-compute token IDs used in decoding.
        let sot_token = token_id(&self.tokenizer, m::SOT_TOKEN);
        let eot_token = token_id(&self.tokenizer, m::EOT_TOKEN);
        let transcribe_token = token_id(&self.tokenizer, m::TRANSCRIBE_TOKEN);
        let timestamp_begin = token_id(&self.tokenizer, "<|0.00|>");
        let vocab_size = self.config.vocab_size as u32;

        // Decoder prompt with timestamps enabled (no NO_TIMESTAMPS token).
        // The model will produce timestamp pairs: <|T_start|> text <|T_end|>
        // which we use to compute seek offsets for accurate long-audio handling.
        // We include <|0.00|> to kickstart timestamp generation.
        let prompt_tokens: Vec<u32> = if self.multilingual {
            let en_token = token_id(&self.tokenizer, "<|en|>");
            vec![sot_token, en_token, transcribe_token, timestamp_begin]
        } else {
            vec![sot_token, transcribe_token, timestamp_begin]
        };

        let suppress_config = SuppressionConfig {
            eot_token,
            timestamp_begin,
            vocab_size,
        };

        // input_stride: encoder conv downsamples 3000 mel frames → 1500 encoder tokens.
        // Each timestamp position = 2 mel frames = 0.02 seconds.
        const INPUT_STRIDE: usize = 2;

        // 7. Two paths based on audio length:
        //    - Short audio (≤ 30s): single pass, no padding — Candle's encoder
        //      dynamically slices positional embeddings, so 5s of audio only
        //      processes 500 mel frames instead of 3000.
        //    - Long audio (> 30s): seek-based sliding window with padding, using
        //      timestamp tokens to advance accurately through the audio.
        if content_frames_from_pcm <= m::N_FRAMES {
            // ── Short path: single pass, no seek loop ──
            // Still pad to N_FRAMES — Whisper was trained on 30s windows and the
            // decoder's timestamp/attention patterns expect this fixed size.
            // The win here is avoiding the seek loop overhead, not the padding.
            let mel = pad_or_trim(&mel_full, m::N_FRAMES, &self.device)
                .map_err(|e| Error::Transcription(format!("Mel pad/trim: {}", e)))?;

            self.model.reset_kv_cache();

            let encoder_output = self
                .model
                .encoder
                .forward(&mel, true)
                .map_err(|e| Error::Transcription(format!("Encoder: {}", e)))?;

            let mut result_tokens = Vec::new();
            for &temperature in TEMPERATURE_SCHEDULE {
                result_tokens = greedy_decode(
                    &mut self.model,
                    &encoder_output,
                    &prompt_tokens,
                    eot_token,
                    &suppress_config,
                    &self.device,
                    temperature,
                )?;
                if !is_degenerate(&result_tokens) {
                    break;
                }
            }

            let text_tokens: Vec<u32> = result_tokens
                .iter()
                .copied()
                .filter(|&t| t < timestamp_begin)
                .collect();

            if text_tokens.is_empty() {
                return Ok(String::new());
            }

            let text = self
                .tokenizer
                .decode(&text_tokens, true)
                .map_err(|e| Error::Transcription(format!("Token decode: {}", e)))?;

            Ok(text.trim().to_string())
        } else {
            // ── Long path: seek-based sliding window with padding ──
            let mut all_text_parts: Vec<String> = Vec::new();
            let mut seek: usize = 0;

            const MIN_SEGMENT_FRAMES: usize = 50;

            while seek < content_frames_from_pcm {
                let segment_frames = (content_frames_from_pcm - seek).min(m::N_FRAMES);

                if segment_frames < MIN_SEGMENT_FRAMES {
                    break;
                }

                let mel_segment = mel_full
                    .narrow(2, seek, segment_frames)
                    .map_err(|e| Error::Transcription(format!("Mel narrow: {}", e)))?;

                // Pad to N_FRAMES — required for seek-based timestamp alignment.
                // Whisper was trained on 30s padded windows; timestamps are relative
                // to this fixed window size.
                let mel = pad_or_trim(&mel_segment, m::N_FRAMES, &self.device)
                    .map_err(|e| Error::Transcription(format!("Mel pad/trim: {}", e)))?;

                self.model.reset_kv_cache();

                let encoder_output = self
                    .model
                    .encoder
                    .forward(&mel, true)
                    .map_err(|e| Error::Transcription(format!("Encoder: {}", e)))?;

                let mut result_tokens = Vec::new();
                for &temperature in TEMPERATURE_SCHEDULE {
                    result_tokens = greedy_decode(
                        &mut self.model,
                        &encoder_output,
                        &prompt_tokens,
                        eot_token,
                        &suppress_config,
                        &self.device,
                        temperature,
                    )?;

                    if !is_degenerate(&result_tokens) {
                        break;
                    }
                    debug!(
                        "[decode] degenerate output at temperature {}, retrying",
                        temperature
                    );
                }

                let last_timestamp_pos = result_tokens
                    .iter()
                    .rev()
                    .find(|&&t| t >= timestamp_begin)
                    .map(|&t| (t - timestamp_begin) as usize);

                debug!(
                    "[seek] frame {} ({:.1}s), {} tokens, last_ts_pos={:?}",
                    seek,
                    seek as f64 * m::HOP_LENGTH as f64 / m::SAMPLE_RATE as f64,
                    result_tokens.len(),
                    last_timestamp_pos,
                );

                let seek_advance = match last_timestamp_pos {
                    Some(ts_pos) => (ts_pos * INPUT_STRIDE).max(1),
                    None => segment_frames,
                };

                // For padded segments, filter text tokens that fall beyond real content.
                let content_ts_limit = if segment_frames < m::N_FRAMES {
                    Some((segment_frames / INPUT_STRIDE) as u32 + timestamp_begin)
                } else {
                    None
                };

                let mut text_tokens: Vec<u32> = Vec::new();
                for &t in &result_tokens {
                    if let Some(limit) = content_ts_limit {
                        if t >= limit {
                            break;
                        }
                    }
                    if t < timestamp_begin {
                        text_tokens.push(t);
                    }
                }

                if !text_tokens.is_empty() {
                    let segment_text = self
                        .tokenizer
                        .decode(&text_tokens, true)
                        .map_err(|e| Error::Transcription(format!("Token decode: {}", e)))?;

                    let segment_text = segment_text.trim().to_string();
                    if !segment_text.is_empty() {
                        debug!(
                            "[segment] seek {}: \"{}\"",
                            seek,
                            &segment_text[..segment_text.len().min(60)]
                        );
                        all_text_parts.push(segment_text);
                    }
                }

                if segment_frames < m::N_FRAMES {
                    break;
                }

                seek += seek_advance;
            }

            Ok(all_text_parts.join(" "))
        }
    }
}

// ── Provider implementation ───────────────────────────────────────

/// Local Pulse provider backed by Candle.
pub struct LocalPulseProvider {
    engine: Mutex<PulseEngine>,
}

impl LocalPulseProvider {
    pub fn new(model: PulseModel) -> Result<Self> {
        let engine = PulseEngine::new(model)?;
        Ok(Self {
            engine: Mutex::new(engine),
        })
    }
}

#[async_trait]
impl TranscriptionProvider for LocalPulseProvider {
    fn name(&self) -> &'static str {
        "local-pulse"
    }

    async fn transcribe(&self, request: TranscriptionRequest) -> Result<TranscriptionResponse> {
        let mut engine = self.engine.lock();
        let start = std::time::Instant::now();

        let text =
            engine.transcribe_sync(&request.audio, request.sample_rate, request.channels)?;

        let duration_ms = start.elapsed().as_millis() as u64;
        Ok(TranscriptionResponse {
            text,
            confidence: None,
            duration_ms,
        })
    }

    fn is_configured(&self) -> bool {
        true
    }
}

// ── Helpers ───────────────────────────────────────────────────────

/// Token suppression configuration.
///
/// Whisper's vocab layout: [text tokens] [EOT] [SOT] [languages...] [tasks...] [NO_TIMESTAMPS] [timestamps...]
/// We suppress the special tokens between EOT and timestamp_begin (SOT, languages, tasks, NO_TIMESTAMPS)
/// but allow text tokens, EOT, and timestamp tokens through.
struct SuppressionConfig {
    eot_token: u32,
    timestamp_begin: u32,
    vocab_size: u32,
}

impl SuppressionConfig {
    /// Returns true if this token should be allowed during decoding.
    ///
    /// Allows text tokens (below EOT), EOT itself, and timestamp tokens.
    /// Suppresses everything in between (SOT, language tokens, task tokens, NO_TIMESTAMPS).
    fn is_allowed(&self, id: u32) -> bool {
        id < self.vocab_size && (id <= self.eot_token || id >= self.timestamp_begin)
    }
}

/// Greedy decoding with optional temperature sampling and token suppression.
///
/// At temperature 0, uses argmax (pure greedy). At temperature > 0, applies
/// softmax(logits / temperature) and samples from the distribution. Uses
/// KV-cached incremental decoding for speed — only the prompt is processed
/// in full on the first step, then one token at a time.
fn greedy_decode(
    model: &mut m::model::Whisper,
    encoder_output: &Tensor,
    prompt_tokens: &[u32],
    eot_token: u32,
    suppress: &SuppressionConfig,
    device: &Device,
    temperature: f32,
) -> Result<Vec<u32>> {
    model.reset_kv_cache();

    let mut tokens = prompt_tokens.to_vec();
    let mut result_tokens: Vec<u32> = Vec::new();

    for _ in 0..MAX_DECODE_TOKENS {
        let token_tensor = Tensor::new(tokens.as_slice(), device)
            .map_err(|e| Error::Transcription(format!("Token tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| Error::Transcription(format!("Unsqueeze: {}", e)))?;

        // On the first step, flush=true processes the full prompt and populates the KV cache.
        // On subsequent steps, flush=false uses cached context and only processes the new token.
        let first_step = result_tokens.is_empty();
        let hidden = model
            .decoder
            .forward(&token_tensor, encoder_output, first_step)
            .map_err(|e| Error::Transcription(format!("Decoder: {}", e)))?;

        let logits = model
            .decoder
            .final_linear(&hidden)
            .map_err(|e| Error::Transcription(format!("Final linear: {}", e)))?;

        let last_logits = logits
            .i((0, tokens.len() - 1))
            .map_err(|e| Error::Transcription(format!("Logit indexing: {}", e)))?;

        let logits_vec: Vec<f32> = last_logits
            .to_vec1::<f32>()
            .map_err(|e| Error::Transcription(format!("Logits to vec: {}", e)))?;

        let next_token = if temperature <= 0.0 {
            suppressed_argmax(&logits_vec, suppress)
        } else {
            sample_with_temperature(&logits_vec, suppress, temperature)
        };

        if next_token == eot_token {
            break;
        }

        result_tokens.push(next_token);
        tokens.push(next_token);
    }

    Ok(result_tokens)
}

/// Check if decoder output is degenerate (excessive repetition).
fn is_degenerate(tokens: &[u32]) -> bool {
    if tokens.is_empty() {
        return false;
    }
    let unique: std::collections::HashSet<u32> = tokens.iter().copied().collect();
    let compression = tokens.len() as f32 / unique.len().max(1) as f32;
    compression > MAX_COMPRESSION_RATIO
}

/// Argmax over logits with token suppression.
fn suppressed_argmax(logits: &[f32], suppress: &SuppressionConfig) -> u32 {
    let mut best_token = suppress.eot_token;
    let mut best_logit = f32::NEG_INFINITY;

    for (id, &logit) in logits.iter().enumerate() {
        let id = id as u32;
        if !suppress.is_allowed(id) {
            continue;
        }
        if logit > best_logit {
            best_logit = logit;
            best_token = id;
        }
    }

    best_token
}

/// Sample from logits with temperature scaling and token suppression.
fn sample_with_temperature(
    logits: &[f32],
    suppress: &SuppressionConfig,
    temperature: f32,
) -> u32 {
    // Build scaled probability distribution over valid tokens.
    let valid: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|&(id, _)| suppress.is_allowed(id as u32))
        .map(|(id, &logit)| (id as u32, logit / temperature))
        .collect();

    if valid.is_empty() {
        return suppress.eot_token;
    }

    // Stable softmax.
    let max_logit = valid.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = valid.iter().map(|(_, l)| ((*l - max_logit) as f64).exp()).sum();
    let probs: Vec<(u32, f64)> = valid
        .iter()
        .map(|(id, l)| (*id, ((*l - max_logit) as f64).exp() / sum_exp))
        .collect();

    // Sample using a simple linear scan with a random threshold.
    let seed_bits = logits.first().unwrap_or(&0.0).to_bits() ^ logits.len() as u32;
    let mut rng_state = seed_bits.wrapping_mul(2654435761);
    rng_state ^= rng_state >> 16;
    rng_state = rng_state.wrapping_mul(2246822507);
    rng_state ^= rng_state >> 13;
    let threshold = (rng_state as f64) / (u32::MAX as f64);

    let mut cumulative = 0.0;
    for &(id, prob) in &probs {
        cumulative += prob;
        if cumulative >= threshold {
            return id;
        }
    }

    probs.last().map_or(suppress.eot_token, |&(id, _)| id)
}

/// Pad or trim a mel tensor to exactly `target_frames` frames.
fn pad_or_trim(
    mel: &Tensor,
    target_frames: usize,
    device: &Device,
) -> std::result::Result<Tensor, candle_core::Error> {
    let (_batch, n_mels, n_frames) = mel.dims3()?;
    if n_frames == target_frames {
        Ok(mel.clone())
    } else if n_frames > target_frames {
        mel.narrow(2, 0, target_frames)
    } else {
        let pad_len = target_frames - n_frames;
        let padding = Tensor::zeros((1, n_mels, pad_len), candle_core::DType::F32, device)?;
        Tensor::cat(&[mel, &padding], 2)
    }
}

/// Look up a special token's ID in the tokenizer.
fn token_id(tokenizer: &Tokenizer, token: &str) -> u32 {
    tokenizer
        .token_to_id(token)
        .unwrap_or_else(|| panic!("Missing token: {}", token))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_model_parse() {
        assert_eq!(PulseModel::parse("fast"), Some(PulseModel::Fast));
        assert_eq!(PulseModel::parse("tiny"), Some(PulseModel::Fast));
        assert_eq!(PulseModel::parse("balanced"), Some(PulseModel::Balanced));
        assert_eq!(PulseModel::parse("base"), Some(PulseModel::Balanced));
        assert_eq!(PulseModel::parse("quality"), Some(PulseModel::Quality));
        assert_eq!(PulseModel::parse("small"), Some(PulseModel::Quality));
        assert_eq!(PulseModel::parse("medium"), Some(PulseModel::Medium));
        assert_eq!(PulseModel::parse("large"), Some(PulseModel::Large));
        assert_eq!(PulseModel::parse("distil"), Some(PulseModel::DistilLarge));
        assert_eq!(PulseModel::parse("distil-large"), Some(PulseModel::DistilLarge));
        assert_eq!(PulseModel::parse("distil-large-v3.5"), Some(PulseModel::DistilLarge));
        assert_eq!(PulseModel::parse("unknown"), None);
    }


}
