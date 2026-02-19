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

use crate::error::{Error, Result};
use crate::providers::transcription::{
    TranscriptionProvider, TranscriptionRequest, TranscriptionResponse,
};

/// Model's expected sample rate (16kHz for Whisper).
const MODEL_SAMPLE_RATE: u32 = 16_000;

/// Beam search width (Whisper default is 5).
const BEAM_WIDTH: usize = 5;
/// Maximum decoder tokens before stopping.
const MAX_DECODE_TOKENS: usize = 224;

/// RMS below this threshold is treated as silence — skip inference entirely.
const SILENCE_RMS_THRESHOLD: f32 = 0.003;
/// Samples with absolute value below this are trimmed from leading/trailing edges.
const SILENCE_TRIM_THRESHOLD: f32 = 0.005;
/// Peak amplitude target for normalization.
const NORMALIZE_TARGET: f32 = 0.9;

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

    /// Whether this engine uses a multilingual model.
    fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// Transcribe PCM audio samples to text.
    pub fn transcribe_sync(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        // 1. Convert to mono.
        let mono: Vec<f32> = if channels == 1 {
            pcm.to_vec()
        } else {
            let ch = channels as usize;
            pcm.chunks(ch)
                .map(|c| c.iter().sum::<f32>() / channels as f32)
                .collect()
        };

        // 2. Resample to 16kHz if needed.
        let pcm_16k = if sample_rate == MODEL_SAMPLE_RATE {
            mono
        } else {
            resample_sinc(&mono, sample_rate, MODEL_SAMPLE_RATE)
        };

        if pcm_16k.is_empty() {
            return Ok(String::new());
        }

        // 3. Silence gate.
        let rms = compute_rms(&pcm_16k);
        debug!("[audio] {} samples, RMS={:.6}", pcm_16k.len(), rms);
        if rms < SILENCE_RMS_THRESHOLD {
            debug!("[audio] below silence threshold, skipping");
            return Ok(String::new());
        }

        // 4. Trim leading/trailing silence.
        let start = pcm_16k
            .iter()
            .position(|s| s.abs() > SILENCE_TRIM_THRESHOLD)
            .unwrap_or(pcm_16k.len());
        let end = pcm_16k
            .iter()
            .rposition(|s| s.abs() > SILENCE_TRIM_THRESHOLD)
            .map_or(start, |i| i + 1);

        if start >= end {
            return Ok(String::new());
        }

        let mut trimmed = pcm_16k[start..end].to_vec();
        debug!("[audio] after trim: {} samples", trimmed.len());

        // 5. Peak normalize.
        let peak = trimmed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 0.0 {
            let gain = NORMALIZE_TARGET / peak;
            for s in trimmed.iter_mut() {
                *s *= gain;
            }
        }

        // Reset KV cache before new inference.
        self.model.reset_kv_cache();

        // 6. Compute mel spectrogram.
        let mel = m::audio::pcm_to_mel(&self.config, &trimmed, &self.mel_filters);
        let n_frames = mel.len() / self.config.num_mel_bins;
        let mel = Tensor::from_vec(mel, (1, self.config.num_mel_bins, n_frames), &self.device)
            .map_err(|e| Error::Transcription(format!("Mel tensor: {}", e)))?;

        // Pad or trim to N_FRAMES (3000) frames.
        let mel = pad_or_trim(&mel, m::N_FRAMES, &self.device)
            .map_err(|e| Error::Transcription(format!("Mel pad/trim: {}", e)))?;

        // 7. Run encoder.
        let encoder_output = self
            .model
            .encoder
            .forward(&mel, true)
            .map_err(|e| Error::Transcription(format!("Encoder: {}", e)))?;

        // 8. Beam search decoding with token suppression.
        let sot_token = token_id(&self.tokenizer, m::SOT_TOKEN);
        let eot_token = token_id(&self.tokenizer, m::EOT_TOKEN);
        let no_timestamps_token = token_id(&self.tokenizer, m::NO_TIMESTAMPS_TOKEN);

        // Build the initial decoder prompt.
        // All models get TRANSCRIBE token to signal the task.
        // Multilingual models additionally need a language token.
        let transcribe_token = token_id(&self.tokenizer, m::TRANSCRIBE_TOKEN);
        let prompt_tokens: Vec<u32> = if self.is_multilingual() {
            let en_token = token_id(&self.tokenizer, "<|en|>");
            vec![sot_token, en_token, transcribe_token, no_timestamps_token]
        } else {
            vec![sot_token, transcribe_token, no_timestamps_token]
        };

        let suppress_above = no_timestamps_token;
        let vocab_size = self.config.vocab_size as u32;

        let result_tokens = beam_search_decode(
            &mut self.model,
            &encoder_output,
            &prompt_tokens,
            eot_token,
            suppress_above,
            vocab_size,
            &self.device,
            BEAM_WIDTH,
            MAX_DECODE_TOKENS,
        )?;

        // 9. Decode tokens to text.
        let text = self
            .tokenizer
            .decode(&result_tokens, true)
            .map_err(|e| Error::Transcription(format!("Token decode: {}", e)))?;

        Ok(text.trim().to_string())
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

/// Beam search decoding for Whisper.
///
/// Maintains `beam_width` hypotheses in parallel, expanding each by the top
/// candidates at every step. Since Candle's decoder has a single internal KV
/// cache, we reset it and recompute from the full token sequence for each beam
/// at each step. The encoder output (the expensive part) is computed once and
/// shared across all beams.
fn beam_search_decode(
    model: &mut m::model::Whisper,
    encoder_output: &Tensor,
    prompt_tokens: &[u32],
    eot_token: u32,
    suppress_above: u32,
    vocab_size: u32,
    device: &Device,
    beam_width: usize,
    max_tokens: usize,
) -> Result<Vec<u32>> {
    let prompt_len = prompt_tokens.len();

    // Each beam: (token_sequence, cumulative_log_prob).
    let mut beams: Vec<(Vec<u32>, f64)> = vec![(prompt_tokens.to_vec(), 0.0)];
    let mut completed: Vec<(Vec<u32>, f64)> = Vec::new();

    for _ in 0..max_tokens {
        let mut candidates: Vec<(Vec<u32>, f64)> = Vec::new();

        for (tokens, cum_log_prob) in &beams {
            // Reset KV cache and run the full sequence for this beam.
            model.reset_kv_cache();

            let token_tensor = Tensor::new(tokens.as_slice(), device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::Transcription(format!("Token tensor: {}", e)))?;

            let hidden = model
                .decoder
                .forward(&token_tensor, encoder_output, true)
                .map_err(|e| Error::Transcription(format!("Decoder: {}", e)))?;

            let logits = model
                .decoder
                .final_linear(&hidden)
                .map_err(|e| Error::Transcription(format!("Final linear: {}", e)))?;

            // Extract logits for the last position and compute log-probabilities.
            let last_logits = logits
                .i((0, tokens.len() - 1))
                .map_err(|e| Error::Transcription(format!("Logit indexing: {}", e)))?;

            let logits_vec: Vec<f32> = last_logits
                .to_vec1::<f32>()
                .map_err(|e| Error::Transcription(format!("Logits to vec: {}", e)))?;

            // Stable log-softmax: log_prob = x - log(sum(exp(x - max)))  - max.
            let max_logit = logits_vec
                .iter()
                .take(vocab_size as usize)
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f64 = logits_vec
                .iter()
                .take(vocab_size as usize)
                .map(|&x| ((x - max_logit) as f64).exp())
                .sum::<f64>()
                .ln()
                + max_logit as f64;

            // Collect valid tokens with their log-probabilities, pick top beam_width.
            let mut scored: Vec<(u32, f64)> = logits_vec
                .iter()
                .enumerate()
                .filter(|&(id, _)| {
                    let id = id as u32;
                    id < vocab_size && (id <= suppress_above || id == eot_token)
                })
                .map(|(id, &logit)| (id as u32, logit as f64 - log_sum_exp))
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(beam_width);

            for (token_id, log_prob) in scored {
                let new_cum = cum_log_prob + log_prob;
                let mut new_tokens = tokens.clone();
                new_tokens.push(token_id);

                if token_id == eot_token {
                    // Length-normalize score by number of result tokens.
                    let result_len = (new_tokens.len() - prompt_len) as f64;
                    let normalized = new_cum / result_len.max(1.0);
                    completed.push((new_tokens, normalized));
                } else {
                    candidates.push((new_tokens, new_cum));
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Keep the top beam_width candidates by cumulative log-probability.
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(beam_width);
        beams = candidates;

        // Early exit if we have enough completed beams.
        if completed.len() >= beam_width {
            break;
        }
    }

    // Add any incomplete beams (hit max_tokens) to the completed set.
    for (tokens, cum_log_prob) in beams {
        let result_len = (tokens.len() - prompt_len) as f64;
        let normalized = cum_log_prob / result_len.max(1.0);
        completed.push((tokens, normalized));
    }

    // Pick the best completed beam by length-normalized score.
    completed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    match completed.into_iter().next() {
        Some((tokens, _)) => Ok(tokens[prompt_len..].to_vec()),
        None => Ok(Vec::new()),
    }
}

/// Look up a special token's ID in the tokenizer.
fn token_id(tokenizer: &Tokenizer, token: &str) -> u32 {
    tokenizer
        .token_to_id(token)
        .unwrap_or_else(|| panic!("Missing token: {}", token))
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
        // Pad with zeros.
        let pad_len = target_frames - n_frames;
        let padding = Tensor::zeros((1, n_mels, pad_len), candle_core::DType::F32, device)?;
        Tensor::cat(&[mel, &padding], 2)
    }
}

/// Windowed-sinc (Lanczos) resampling with anti-aliasing.
///
/// Uses a sinc kernel with a Lanczos window to properly low-pass filter
/// the signal before decimation, preventing aliasing artifacts. The kernel
/// half-width of 8 samples provides good quality for speech audio.
fn resample_sinc(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    // Anti-aliasing: when downsampling, scale the cutoff frequency down.
    let cutoff = if to_rate < from_rate {
        ratio // e.g. 16000/48000 = 0.333
    } else {
        1.0
    };

    // Kernel half-width in source samples. 8 lobes is good for speech.
    let half_width: usize = 8;

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let center = src_pos.floor() as i64;
        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for j in (center - half_width as i64 + 1)..=(center + half_width as i64) {
            if j < 0 || j >= samples.len() as i64 {
                continue;
            }
            let x = (src_pos - j as f64) * cutoff;
            let w = lanczos_kernel(x, half_width as f64);
            sum += samples[j as usize] as f64 * w * cutoff;
            weight_sum += w * cutoff;
        }

        let sample = if weight_sum > 0.0 {
            (sum / weight_sum) as f32
        } else {
            0.0
        };
        output.push(sample);
    }

    output
}

/// Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, else 0.
fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else if x.abs() >= a {
        0.0
    } else {
        let px = std::f64::consts::PI * x;
        let pxa = std::f64::consts::PI * x / a;
        (px.sin() / px) * (pxa.sin() / pxa)
    }
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
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

    #[test]
    fn test_resample_sinc_same_rate() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let resampled = resample_sinc(&samples, 44100, 44100);
        assert_eq!(resampled, samples);
    }

    #[test]
    fn test_resample_sinc_downsample() {
        let samples: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.01).sin()).collect();
        let resampled = resample_sinc(&samples, 44100, 16000);
        assert!((resampled.len() as i64 - 16000).abs() < 2);
    }

    #[test]
    fn test_resample_sinc_empty() {
        let resampled = resample_sinc(&[], 44100, 16000);
        assert!(resampled.is_empty());
    }

    #[test]
    fn test_compute_rms_silence() {
        assert_eq!(compute_rms(&vec![0.0; 100]), 0.0);
    }

    #[test]
    fn test_compute_rms_signal() {
        let rms = compute_rms(&vec![0.5; 100]);
        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_pad_or_trim_exact() {
        let device = Device::Cpu;
        let mel = Tensor::zeros((1, 80, 3000), candle_core::DType::F32, &device).unwrap();
        let result = pad_or_trim(&mel, 3000, &device).unwrap();
        assert_eq!(result.dims3().unwrap(), (1, 80, 3000));
    }

    #[test]
    fn test_pad_or_trim_pad() {
        let device = Device::Cpu;
        let mel = Tensor::zeros((1, 80, 1000), candle_core::DType::F32, &device).unwrap();
        let result = pad_or_trim(&mel, 3000, &device).unwrap();
        assert_eq!(result.dims3().unwrap(), (1, 80, 3000));
    }

    #[test]
    fn test_pad_or_trim_trim() {
        let device = Device::Cpu;
        let mel = Tensor::zeros((1, 80, 5000), candle_core::DType::F32, &device).unwrap();
        let result = pad_or_trim(&mel, 3000, &device).unwrap();
        assert_eq!(result.dims3().unwrap(), (1, 80, 3000));
    }
}
