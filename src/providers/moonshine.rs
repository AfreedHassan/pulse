//! Moonshine Base streaming STT provider using ONNX Runtime.
//!
//! Moonshine (UsefulSensors) takes raw waveform with variable-length input — no padding,
//! no mel spectrogram. Expected ~50-250ms latency for 5s chunks vs ~4-5s with Whisper.
//! Uses ONNX Runtime with CoreML for Apple Silicon GPU acceleration.

use hf_hub::api::sync::Api;
use ndarray::Array4;
use ort::{
    execution_providers::CoreMLExecutionProvider,
    session::Session,
    value::Tensor,
};
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::audio::resample::preprocess_audio;
use crate::error::{Error, Result};

/// HuggingFace repo for Moonshine Base ONNX model (community ONNX export).
const MOONSHINE_REPO: &str = "onnx-community/moonshine-base-ONNX";

/// End-of-sequence token ID for Moonshine.
const EOS_TOKEN_ID: i64 = 2;

/// Repetition penalty factor applied to already-generated tokens.
/// >1.0 discourages repetition; 1.2 is a common default.
const REPETITION_PENALTY: f32 = 1.2;

/// If token_count / unique_token_count exceeds this, output is degenerate.
const MAX_COMPRESSION_RATIO: f32 = 2.4;

/// Number of decoder layers in Moonshine Base.
const NUM_LAYERS: usize = 8;
/// Number of attention heads.
const NUM_HEADS: usize = 8;
/// Head dimension (hidden_size=416 / num_heads=8).
const HEAD_DIM: usize = 52;

/// Moonshine Base ONNX inference engine.
pub struct MoonshineEngine {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
}

impl MoonshineEngine {
    /// Build the engine, downloading model files from HuggingFace on first run.
    ///
    /// CoreML execution provider compiles the model on first run (~30-60s), cached after.
    pub fn new() -> Result<Self> {
        let api = Api::new()
            .map_err(|e| Error::Transcription(format!("HF API init failed: {}", e)))?;

        info!("Downloading/loading Moonshine Base model...");
        let repo = api.model(MOONSHINE_REPO.to_string());

        let encoder_path = repo
            .get("onnx/encoder_model.onnx")
            .map_err(|e| Error::Transcription(format!("Failed to download encoder: {}", e)))?;
        let decoder_path = repo
            .get("onnx/decoder_model_merged.onnx")
            .map_err(|e| Error::Transcription(format!("Failed to download decoder: {}", e)))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Transcription(format!("Failed to download tokenizer: {}", e)))?;

        info!("Encoder: {:?}", encoder_path);
        info!("Decoder: {:?}", decoder_path);

        eprintln!("Initializing ONNX Runtime (CoreML compilation may take 30-60s on first run)...");

        let encoder = Session::builder()
            .map_err(|e| Error::Transcription(format!("ORT session builder: {}", e)))?
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(|e| Error::Transcription(format!("CoreML EP: {}", e)))?
            .commit_from_file(&encoder_path)
            .map_err(|e| Error::Transcription(format!("Load encoder: {}", e)))?;

        let decoder = Session::builder()
            .map_err(|e| Error::Transcription(format!("ORT session builder: {}", e)))?
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(|e| Error::Transcription(format!("CoreML EP: {}", e)))?
            .commit_from_file(&decoder_path)
            .map_err(|e| Error::Transcription(format!("Load decoder: {}", e)))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::Transcription(format!("Failed to load tokenizer: {}", e)))?;

        info!("Moonshine Base loaded successfully");

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
        })
    }

    /// Transcribe PCM audio samples to text.
    pub fn transcribe_sync(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        let audio = match preprocess_audio(pcm, sample_rate, channels) {
            Some(a) => a,
            None => return Ok(String::new()),
        };

        let num_samples = audio.len();
        let audio_duration = num_samples as f64 / 16_000.0;
        debug!(
            "[moonshine] {} samples ({:.2}s) after preprocessing",
            num_samples, audio_duration
        );

        let max_tokens = (audio_duration * 8.0).ceil() as usize + 16;

        // --- Encoder ---
        // Input: "input_values" shape [1, num_samples]
        let audio_tensor = Tensor::from_array(([1i64, num_samples as i64], audio))
            .map_err(|e| Error::Transcription(format!("Audio tensor: {}", e)))?;

        let encoder_outputs = self
            .encoder
            .run(ort::inputs![audio_tensor])
            .map_err(|e| Error::Transcription(format!("Encoder forward: {}", e)))?;

        // Output: "last_hidden_state" shape [1, enc_seq_len, 416]
        let (hidden_shape, hidden_data) = encoder_outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Transcription(format!("Extract hidden states: {}", e)))?;
        let enc_seq_len = hidden_shape[1] as usize;
        let hidden_data_vec: Vec<f32> = hidden_data.to_vec();

        debug!(
            "[moonshine] encoder output: [{}, {}, {}]",
            hidden_shape[0], hidden_shape[1], hidden_shape[2]
        );

        // --- Decoder: autoregressive loop with KV cache ---
        // KV cache state: decoder self-attention + encoder cross-attention
        // Each layer has 4 tensors: decoder.key, decoder.value, encoder.key, encoder.value
        // Shape: [1, NUM_HEADS, seq_len, HEAD_DIM]
        let mut decoder_kv: Vec<Vec<f32>> = vec![Vec::new(); NUM_LAYERS * 2]; // key+value per layer
        let mut encoder_kv: Vec<Vec<f32>> = vec![Vec::new(); NUM_LAYERS * 2];
        let mut decoder_kv_seq: usize = 0;
        let mut encoder_kv_seq: usize = 0;

        let mut generated_tokens: Vec<i64> = vec![1]; // BOS token

        for step in 0..max_tokens {
            let use_cache = step > 0;

            // Build named inputs.
            let mut inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> =
                Vec::new();

            // input_ids
            let input_ids: Vec<i64> = if use_cache {
                vec![*generated_tokens.last().unwrap()]
            } else {
                generated_tokens.clone()
            };
            let ids_len = input_ids.len() as i64;
            inputs.push((
                "input_ids".into(),
                Tensor::from_array(([1i64, ids_len], input_ids))
                    .map_err(|e| Error::Transcription(format!("IDs tensor: {}", e)))?
                    .into(),
            ));

            // encoder_hidden_states
            inputs.push((
                "encoder_hidden_states".into(),
                Tensor::from_array((
                    [1i64, enc_seq_len as i64, 416i64],
                    hidden_data_vec.clone(),
                ))
                .map_err(|e| Error::Transcription(format!("Hidden states tensor: {}", e)))?
                .into(),
            ));

            // past_key_values: 8 layers × (decoder.key, decoder.value, encoder.key, encoder.value)
            // Use ndarray::Array4 to create tensors — it allows 0-length dims unlike the (shape, Vec) path.
            for layer in 0..NUM_LAYERS {
                let kv_names = [
                    (format!("past_key_values.{}.decoder.key", layer), &decoder_kv[layer * 2], decoder_kv_seq),
                    (format!("past_key_values.{}.decoder.value", layer), &decoder_kv[layer * 2 + 1], decoder_kv_seq),
                    (format!("past_key_values.{}.encoder.key", layer), &encoder_kv[layer * 2], encoder_kv_seq),
                    (format!("past_key_values.{}.encoder.value", layer), &encoder_kv[layer * 2 + 1], encoder_kv_seq),
                ];

                for (name, data, seq_len) in kv_names {
                    let arr = if seq_len == 0 {
                        Array4::<f32>::zeros((1, NUM_HEADS, 0, HEAD_DIM))
                    } else {
                        Array4::from_shape_vec((1, NUM_HEADS, seq_len, HEAD_DIM), data.clone())
                            .map_err(|e| Error::Transcription(format!("KV array {}: {}", name, e)))?
                    };
                    inputs.push((
                        name.into(),
                        Tensor::from_array(arr)
                            .map_err(|e| Error::Transcription(format!("KV tensor: {}", e)))?
                            .into(),
                    ));
                }
            }

            // use_cache_branch (last input)
            inputs.push((
                "use_cache_branch".into(),
                Tensor::from_array(([1i64], vec![use_cache]))
                    .map_err(|e| Error::Transcription(format!("Cache flag: {}", e)))?
                    .into(),
            ));

            // Run decoder.
            let outputs = self
                .decoder
                .run(inputs)
                .map_err(|e| {
                    Error::Transcription(format!("Decoder forward step {}: {}", step, e))
                })?;

            // Extract logits, apply repetition penalty, then argmax.
            let (logits_shape, logits_data) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Transcription(format!("Extract logits: {}", e)))?;
            let vocab_size = logits_shape[2] as usize;
            let mut last_logits = logits_data[logits_data.len() - vocab_size..].to_vec();

            // Apply repetition penalty: divide logits for already-generated tokens.
            for &prev_token in &generated_tokens {
                let idx = prev_token as usize;
                if idx < last_logits.len() {
                    if last_logits[idx] > 0.0 {
                        last_logits[idx] /= REPETITION_PENALTY;
                    } else {
                        last_logits[idx] *= REPETITION_PENALTY;
                    }
                }
            }

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx as i64)
                .unwrap_or(EOS_TOKEN_ID);

            if next_token == EOS_TOKEN_ID {
                debug!("[moonshine] EOS at step {}", step);
                break;
            }

            generated_tokens.push(next_token);

            // Check for degenerate output (repetition loop) every 20 tokens.
            if generated_tokens.len() > 20 && generated_tokens.len() % 20 == 0 {
                let toks = &generated_tokens[1..]; // skip BOS
                let unique: std::collections::HashSet<i64> = toks.iter().copied().collect();
                let compression = toks.len() as f32 / unique.len().max(1) as f32;
                if compression > MAX_COMPRESSION_RATIO {
                    debug!(
                        "[moonshine] degenerate output at step {} (compression {:.1}), stopping",
                        step, compression
                    );
                    // Truncate to the first non-repeating portion.
                    // Find the last point before repetition started by looking for
                    // the first bigram that repeats excessively.
                    let truncate_to = find_repetition_start(toks);
                    generated_tokens.truncate(1 + truncate_to); // +1 for BOS
                    break;
                }
            }

            // Update KV caches from present.*.* outputs.
            for layer in 0..NUM_LAYERS {
                // Decoder self-attention KV (grows each step).
                let dk_out_name = format!("present.{}.decoder.key", layer);
                let (dk_shape, dk_data) = outputs[dk_out_name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Transcription(format!("KV output: {}", e)))?;
                decoder_kv[layer * 2] = dk_data.to_vec();

                let dv_out_name = format!("present.{}.decoder.value", layer);
                let (_, dv_data) = outputs[dv_out_name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Transcription(format!("KV output: {}", e)))?;
                decoder_kv[layer * 2 + 1] = dv_data.to_vec();

                if layer == 0 {
                    decoder_kv_seq = dk_shape[2] as usize;
                    debug!("[moonshine] step {} decoder KV seq: {}", step, decoder_kv_seq);
                }

                // Encoder cross-attention KV: only read on first step (populated by encoder),
                // reused unchanged on subsequent steps.
                if !use_cache {
                    let ek_out_name = format!("present.{}.encoder.key", layer);
                    let (ek_shape, ek_data) = outputs[ek_out_name.as_str()]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| Error::Transcription(format!("KV output: {}", e)))?;
                    encoder_kv[layer * 2] = ek_data.to_vec();

                    let ev_out_name = format!("present.{}.encoder.value", layer);
                    let (_, ev_data) = outputs[ev_out_name.as_str()]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| Error::Transcription(format!("KV output: {}", e)))?;
                    encoder_kv[layer * 2 + 1] = ev_data.to_vec();

                    if layer == 0 {
                        encoder_kv_seq = ek_shape[2] as usize;
                        debug!("[moonshine] encoder KV seq: {}", encoder_kv_seq);
                    }
                }
            }
        }

        // Decode tokens (skip BOS token at index 0).
        let token_ids: Vec<u32> = generated_tokens
            .iter()
            .skip(1)
            .map(|&t| t as u32)
            .collect();

        if token_ids.is_empty() {
            return Ok(String::new());
        }

        let text = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| Error::Transcription(format!("Token decode: {}", e)))?;

        Ok(text.trim().to_string())
    }
}

/// Find the index where repetition starts in a token sequence.
/// Looks for the first bigram that repeats more than 3 times and returns
/// the position of its first occurrence.
fn find_repetition_start(tokens: &[i64]) -> usize {
    if tokens.len() < 4 {
        return tokens.len();
    }

    let mut bigram_first_pos: std::collections::HashMap<(i64, i64), usize> =
        std::collections::HashMap::new();
    let mut bigram_count: std::collections::HashMap<(i64, i64), usize> =
        std::collections::HashMap::new();

    for i in 0..tokens.len() - 1 {
        let bigram = (tokens[i], tokens[i + 1]);
        bigram_first_pos.entry(bigram).or_insert(i);
        *bigram_count.entry(bigram).or_insert(0) += 1;
    }

    // Find the earliest bigram that repeats excessively.
    let mut earliest_repeat = tokens.len();
    for (bigram, count) in &bigram_count {
        if *count > 3 {
            if let Some(&pos) = bigram_first_pos.get(bigram) {
                earliest_repeat = earliest_repeat.min(pos);
            }
        }
    }

    // Keep at least some tokens before the repetition point.
    earliest_repeat.max(1)
}
