use pulse::providers::coreml_whisper::{CoreMLModel, CoreMLWhisperEngine};
use pulse::providers::local_whisper::{PulseEngine, PulseModel};
use pulse::providers::moonshine::MoonshineEngine;
use pulse::providers::Engine;
use std::process::Command;

/// Generate speech audio using macOS `say` + `afconvert`, return PCM samples.
fn generate_speech(text: &str) -> (Vec<f32>, u32) {
    let tmp = std::env::temp_dir();
    let aiff_path = tmp.join("pulse_test.aiff");
    let wav_path = tmp.join("pulse_test.wav");

    let status = Command::new("say")
        .args(["-o", aiff_path.to_str().unwrap(), text])
        .status()
        .expect("Failed to run `say` -- are you on macOS?");
    assert!(status.success(), "`say` command failed");

    let status = Command::new("afconvert")
        .args([
            aiff_path.to_str().unwrap(),
            wav_path.to_str().unwrap(),
            "-d",
            "LEI16@16000",
            "-f",
            "WAVE",
            "-c",
            "1",
        ])
        .status()
        .expect("Failed to run `afconvert`");
    assert!(status.success(), "`afconvert` command failed");

    let _ = std::fs::remove_file(&aiff_path);

    let (samples, sample_rate, _channels) =
        pulse::read_wav(wav_path.to_str().unwrap()).expect("Failed to read WAV");

    let _ = std::fs::remove_file(&wav_path);

    (samples, sample_rate)
}

/// Full end-to-end: macOS TTS → WAV → Pulse → check output.
///
/// Run with: cargo test --release -- --ignored --nocapture
#[test]
#[ignore] // requires model download (~967MB first run), macOS, and is slow
fn test_e2e_with_tts() {
    let text = "The quick brown fox jumps over the lazy dog";
    eprintln!("Generating speech: \"{}\"", text);
    let (samples, sample_rate) = generate_speech(text);
    let duration = samples.len() as f64 / sample_rate as f64;
    eprintln!("Generated {:.1}s of audio at {} Hz", duration, sample_rate);

    // Debug: check audio has actual content
    let rms: f64 = (samples
        .iter()
        .map(|s| (*s as f64) * (*s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    eprintln!("Audio RMS: {:.6}, max |sample|: {:.6}", rms, max_abs);
    assert!(
        max_abs > 0.01,
        "Audio appears to be silence (max={:.6})",
        max_abs
    );

    eprintln!("Loading model...");
    let mut engine = PulseEngine::new(PulseModel::Fast).expect("Failed to load model");

    eprintln!("Transcribing...");
    let result = engine
        .transcribe_sync(&samples, sample_rate, 1)
        .expect("Transcription failed");

    eprintln!("Expected: \"{}\"", text);
    eprintln!("Got:      \"{}\"", result);

    assert!(!result.is_empty(), "Transcription was empty");

    let lower = result.to_lowercase();
    let key_words = ["quick", "brown", "fox", "lazy", "dog"];
    let matches: Vec<&str> = key_words
        .into_iter()
        .filter(|w| lower.contains(w))
        .collect();
    eprintln!("Matched {}/5 key words: {:?}", matches.len(), matches);
    assert!(
        matches.len() >= 3,
        "Expected at least 3/5 key words, got {}: {:?}",
        matches.len(),
        matches
    );
}

/// Simulates VAD-based chunked transcription: generates two utterances with a
/// silence gap between them, splits at the gap, transcribes each chunk separately,
/// and verifies the joined result matches the original.
#[test]
#[ignore]
fn test_vad_chunked_transcription() {
    let phrase_1 = "The quick brown fox";
    let phrase_2 = "jumps over the lazy dog";

    eprintln!("Generating speech for two utterances...");
    let (samples_1, sr) = generate_speech(phrase_1);
    let (samples_2, _) = generate_speech(phrase_2);

    // Insert a 700ms silence gap between the two utterances (simulates a natural pause).
    let silence_gap = vec![0.0f32; (sr as f64 * 0.7) as usize];

    let mut combined = Vec::with_capacity(samples_1.len() + silence_gap.len() + samples_2.len());
    combined.extend_from_slice(&samples_1);
    combined.extend_from_slice(&silence_gap);
    combined.extend_from_slice(&samples_2);

    let total_dur = combined.len() as f64 / sr as f64;
    eprintln!(
        "Combined audio: {:.1}s ({:.1}s + 0.7s silence + {:.1}s)",
        total_dur,
        samples_1.len() as f64 / sr as f64,
        samples_2.len() as f64 / sr as f64,
    );

    // --- VAD: find the silence gap ---
    let vad_threshold: f32 = 0.005;
    let window_size = sr as usize / 20; // 50ms windows
    let min_silence_windows = 12; // 600ms = 12 × 50ms

    let mut silence_run = 0;
    let mut split_point = None;

    for (i, chunk) in combined.chunks(window_size).enumerate() {
        let rms = compute_rms(chunk);
        if rms < vad_threshold {
            silence_run += 1;
            if silence_run >= min_silence_windows && split_point.is_none() {
                // Split at the start of the silence run.
                split_point = Some((i + 1 - silence_run) * window_size);
            }
        } else {
            silence_run = 0;
        }
    }

    let split = split_point.expect("VAD failed to find a silence gap");
    eprintln!(
        "VAD split at sample {} ({:.2}s)",
        split,
        split as f64 / sr as f64
    );

    // --- Transcribe each chunk separately (simulating what the VAD pipeline does) ---
    eprintln!("Loading model...");
    let mut engine = PulseEngine::new(PulseModel::Fast).expect("Failed to load model");

    let chunk_1 = &combined[..split];
    let chunk_2 = &combined[split..];

    eprintln!(
        "Transcribing chunk 1 ({:.1}s)...",
        chunk_1.len() as f64 / sr as f64
    );
    let text_1 = engine
        .transcribe_sync(chunk_1, sr, 1)
        .expect("Chunk 1 transcription failed");
    eprintln!("Chunk 1: \"{}\"", text_1);

    eprintln!(
        "Transcribing chunk 2 ({:.1}s)...",
        chunk_2.len() as f64 / sr as f64
    );
    let text_2 = engine
        .transcribe_sync(chunk_2, sr, 1)
        .expect("Chunk 2 transcription failed");
    eprintln!("Chunk 2: \"{}\"", text_2);

    // --- Also transcribe the whole thing in one shot for comparison ---
    eprintln!(
        "Transcribing full audio ({:.1}s) for comparison...",
        total_dur
    );
    let text_full = engine
        .transcribe_sync(&combined, sr, 1)
        .expect("Full transcription failed");
    eprintln!("Full:    \"{}\"", text_full);

    // --- Verify ---
    let joined = format!("{} {}", text_1, text_2);
    eprintln!("\nJoined chunks: \"{}\"", joined);
    eprintln!("Full single:   \"{}\"", text_full);

    // Both approaches should capture the key words.
    let key_words = ["quick", "brown", "fox", "lazy", "dog"];
    let joined_lower = joined.to_lowercase();
    let matches: Vec<&str> = key_words
        .iter()
        .filter(|w| joined_lower.contains(**w))
        .copied()
        .collect();
    eprintln!(
        "Matched {}/5 key words in joined: {:?}",
        matches.len(),
        matches
    );
    assert!(
        matches.len() >= 4,
        "Expected at least 4/5 key words in joined chunks, got {}: {:?}",
        matches.len(),
        matches
    );

    // Chunk 1 should have fox-related words, chunk 2 should have dog-related words.
    assert!(
        text_1.to_lowercase().contains("fox") || text_1.to_lowercase().contains("brown"),
        "Chunk 1 should contain words from phrase 1, got: \"{}\"",
        text_1
    );
    assert!(
        text_2.to_lowercase().contains("dog") || text_2.to_lowercase().contains("lazy"),
        "Chunk 2 should contain words from phrase 2, got: \"{}\"",
        text_2
    );
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// Tests the pipeline doesn't crash on silence.
#[test]
#[ignore]
fn test_pipeline_with_silence() {
    let mut engine = PulseEngine::new(PulseModel::Fast).expect("Failed to load model");
    let silence = vec![0.0f32; 16000 * 2];
    let result = engine
        .transcribe_sync(&silence, 16000, 1)
        .expect("Transcription failed on silence");
    eprintln!("Silence transcription: {:?}", result);
}

/// Tests resampling path: 48kHz stereo silence.
#[test]
#[ignore]
fn test_pipeline_with_resampling() {
    let mut engine = PulseEngine::new(PulseModel::Fast).expect("Failed to load model");
    let silence_stereo = vec![0.0f32; 48000 * 2 * 2];
    let result = engine
        .transcribe_sync(&silence_stereo, 48000, 2)
        .expect("Transcription with resampling failed");
    eprintln!("Resampled transcription: {:?}", result);
}

/// Tests different model tiers load successfully.
#[test]
#[ignore]
fn test_model_tiers() {
    // Fast (tiny.en)
    let engine_fast = PulseEngine::new(PulseModel::Fast).expect("Failed to load fast model");
    drop(engine_fast);
    eprintln!("Fast model loaded successfully");

    // Balanced (base.en)
    let engine_balanced =
        PulseEngine::new(PulseModel::Balanced).expect("Failed to load balanced model");
    drop(engine_balanced);
    eprintln!("Balanced model loaded successfully");
}

/// Tests model tier parsing.
#[test]
fn test_pulse_model_parse() {
    use pulse::providers::local_whisper::PulseModel;

    assert_eq!(PulseModel::parse("fast"), Some(PulseModel::Fast));
    assert_eq!(PulseModel::parse("tiny"), Some(PulseModel::Fast));
    assert_eq!(PulseModel::parse("balanced"), Some(PulseModel::Balanced));
    assert_eq!(PulseModel::parse("base"), Some(PulseModel::Balanced));
    assert_eq!(PulseModel::parse("quality"), Some(PulseModel::Quality));
    assert_eq!(PulseModel::parse("small"), Some(PulseModel::Quality));
    assert_eq!(PulseModel::parse("medium"), Some(PulseModel::Medium));
    assert_eq!(PulseModel::parse("large"), Some(PulseModel::Large));
    assert_eq!(PulseModel::parse("large-v3"), Some(PulseModel::Large));
    assert_eq!(PulseModel::parse("unknown"), None);
}

/// ~100-word transcription accuracy test.
///
/// Uses macOS TTS to generate a passage with varied vocabulary, numbers,
/// and natural sentence structure. Measures word-level accuracy by comparing
/// expected vs actual words (case-insensitive, punctuation-stripped).
///
/// Run with: cargo test --release test_100_word_accuracy -- --ignored --nocapture
#[test]
#[ignore]
fn test_100_word_accuracy() {
    let sentences = [
        "Yesterday morning I walked through the park and noticed the leaves were changing color.",
        "The temperature outside was around fifty degrees, which felt perfect for a long walk.",
        "Several children were playing near the fountain while their parents watched from wooden benches.",
        "A small brown dog ran across the path chasing after a bright red ball.",
        "I stopped at the corner bakery and ordered a coffee with two sugars and cream.",
        "The woman behind the counter smiled and said it would be ready in just a minute.",
        "While waiting I noticed a newspaper headline about new technology changing how people communicate.",
        "After finishing my drink I continued walking toward the library on the other side of town.",
    ];

    let full_text = sentences.join(" ");
    let expected_words: Vec<String> = normalize_words(&full_text);
    let word_count = expected_words.len();
    eprintln!("Test passage: {} words", word_count);
    eprintln!("Text: \"{}\"", full_text);

    // Generate speech for each sentence separately to avoid TTS issues with very long text,
    // then concatenate with small pauses.
    eprintln!("Generating speech via macOS TTS...");
    let mut all_samples: Vec<f32> = Vec::new();
    let mut sr = 0u32;

    for (i, sentence) in sentences.iter().enumerate() {
        let (samples, sample_rate) = generate_speech(sentence);
        sr = sample_rate;
        all_samples.extend_from_slice(&samples);
        // Add a 300ms pause between sentences.
        if i < sentences.len() - 1 {
            all_samples.extend(vec![0.0f32; (sr as f64 * 0.3) as usize]);
        }
    }

    let duration = all_samples.len() as f64 / sr as f64;
    eprintln!("Total audio: {:.1}s at {} Hz", duration, sr);

    eprintln!("Loading model...");
    let mut engine = PulseEngine::new(PulseModel::Medium).expect("Failed to load model");

    eprintln!("Transcribing...");
    let start = std::time::Instant::now();
    let result = engine
        .transcribe_sync(&all_samples, sr, 1)
        .expect("Transcription failed");
    let elapsed = start.elapsed();

    eprintln!("Transcribed in {:.1}s (RTF: {:.2}x)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / duration);
    eprintln!("Result: \"{}\"", result);

    assert!(!result.is_empty(), "Transcription was empty");

    let result_words = normalize_words(&result);
    eprintln!("\nExpected {} words, got {} words", expected_words.len(), result_words.len());

    // Calculate word-level accuracy using longest common subsequence.
    let lcs_len = longest_common_subsequence(&expected_words, &result_words);
    let accuracy = lcs_len as f64 / expected_words.len() as f64 * 100.0;

    eprintln!("Word accuracy (LCS): {}/{} = {:.1}%", lcs_len, expected_words.len(), accuracy);

    // Show missed words (unique only).
    let result_set: std::collections::HashSet<&str> =
        result_words.iter().map(|s| s.as_str()).collect();
    let unique_missed: std::collections::BTreeSet<&str> = expected_words
        .iter()
        .map(|s| s.as_str())
        .filter(|w| !result_set.contains(w))
        .collect();
    if !unique_missed.is_empty() {
        eprintln!("Words not found in output: {:?}", unique_missed);
    }

    // Require at least 70% word accuracy for the fast model with TTS input.
    assert!(
        accuracy >= 70.0,
        "Word accuracy too low: {:.1}% (expected >= 70%)",
        accuracy,
    );

    eprintln!("\nPASSED: {:.1}% word accuracy", accuracy);
}

/// Strip punctuation and lowercase for word comparison.
fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Longest common subsequence length (for word-level accuracy).
fn longest_common_subsequence(a: &[String], b: &[String]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    dp[m][n]
}

/// Simulates the interactive recording loop: uses VAD to split audio at silence
/// boundaries, transcribes each utterance independently, and joins the results —
/// matching how the real pipeline chunks audio.
///
/// Run with: cargo test --release test_interactive_chunked -- --ignored --nocapture
#[test]
#[ignore]
fn test_interactive_chunked() {
    let sentences = [
        "Yesterday morning I walked through the park and noticed the leaves were changing color.",
        "The temperature outside was around fifty degrees, which felt perfect for a long walk.",
        "Several children were playing near the fountain while their parents watched from wooden benches.",
        "A small brown dog ran across the path chasing after a bright red ball.",
        "I stopped at the corner bakery and ordered a coffee with two sugars and cream.",
        "The woman behind the counter smiled and said it would be ready in just a minute.",
        "While waiting I noticed a newspaper headline about new technology changing how people communicate.",
        //"After finishing my drink I continued walking toward the library on the other side of town.",
    ];

    let full_text = sentences.join(" ");
    let expected_words = normalize_words(&full_text);
    eprintln!("Test passage: {} words", expected_words.len());

    // Generate audio.
    eprintln!("Generating speech via macOS TTS...");
    let mut all_samples: Vec<f32> = Vec::new();
    let mut sr = 0u32;
    for (i, sentence) in sentences.iter().enumerate() {
        let (samples, sample_rate) = generate_speech(sentence);
        sr = sample_rate;
        all_samples.extend_from_slice(&samples);
        if i < sentences.len() - 1 {
            all_samples.extend(vec![0.0f32; (sr as f64 * 0.3) as usize]);
        }
    }
    let total_duration = all_samples.len() as f64 / sr as f64;
    eprintln!("Total audio: {:.1}s at {} Hz", total_duration, sr);

    // VAD-based splitting: find silence gaps and split there.
    let vad_threshold: f32 = 0.005;
    let window_size = sr as usize / 20; // 50ms windows
    let min_silence_windows = 4; // 200ms of silence triggers a split
    let min_chunk_samples = sr as usize / 2; // 0.5s minimum chunk

    let mut chunks: Vec<&[f32]> = Vec::new();
    let mut chunk_start = 0;
    let mut silence_run = 0;

    for (i, window) in all_samples.chunks(window_size).enumerate() {
        let rms = compute_rms(window);
        if rms < vad_threshold {
            silence_run += 1;
            if silence_run >= min_silence_windows {
                let split = (i + 1 - silence_run) * window_size;
                if split > chunk_start && (split - chunk_start) >= min_chunk_samples {
                    chunks.push(&all_samples[chunk_start..split]);
                    chunk_start = (i + 1) * window_size;
                }
                silence_run = 0;
            }
        } else {
            silence_run = 0;
        }
    }
    // Remaining audio.
    if chunk_start < all_samples.len() && (all_samples.len() - chunk_start) >= min_chunk_samples {
        chunks.push(&all_samples[chunk_start..]);
    }

    eprintln!("VAD split into {} chunks", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        eprintln!("  chunk {}: {:.1}s", i, chunk.len() as f64 / sr as f64);
    }

    eprintln!("Loading model...");
    let mut engine = PulseEngine::new(PulseModel::Medium).expect("Failed to load model");

    // Transcribe each chunk independently, measuring per-chunk latency.
    let mut transcripts: Vec<String> = Vec::new();
    let mut chunk_latencies: Vec<(f64, f64)> = Vec::new(); // (audio_dur, transcribe_dur)
    let start = std::time::Instant::now();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_dur = chunk.len() as f64 / sr as f64;
        let chunk_start = std::time::Instant::now();
        match engine.transcribe_sync(chunk, sr, 1) {
            Ok(text) if !text.is_empty() => {
                let latency = chunk_start.elapsed().as_secs_f64();
                eprintln!("[chunk {}] {:.1}s audio -> {:.0}ms latency -> \"{}\"",
                    i, chunk_dur, latency * 1000.0, &text[..text.len().min(60)]);
                chunk_latencies.push((chunk_dur, latency));
                transcripts.push(text);
            }
            Ok(_) => {
                let latency = chunk_start.elapsed().as_secs_f64();
                eprintln!("[chunk {}] {:.1}s audio -> {:.0}ms latency -> (empty)", i, chunk_dur, latency * 1000.0);
                chunk_latencies.push((chunk_dur, latency));
            }
            Err(e) => eprintln!("[chunk {}] error: {}", i, e),
        }
    }

    let elapsed = start.elapsed();
    let result = transcripts.join(" ");

    // Per-chunk latency summary.
    if let Some(&(last_audio, last_latency)) = chunk_latencies.last() {
        eprintln!("\n── Latency (speech-end to text-ready) ──");
        eprintln!("  Final chunk: {:.1}s audio transcribed in {:.0}ms", last_audio, last_latency * 1000.0);
    }
    let avg_latency: f64 = chunk_latencies.iter().map(|(_, l)| l).sum::<f64>() / chunk_latencies.len() as f64;
    let max_latency = chunk_latencies.iter().map(|(_, l)| *l).fold(0.0f64, f64::max);
    eprintln!("  Avg chunk latency: {:.0}ms", avg_latency * 1000.0);
    eprintln!("  Max chunk latency: {:.0}ms", max_latency * 1000.0);

    eprintln!("\nTotal transcription time: {:.1}s (RTF: {:.2}x)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / total_duration);
    eprintln!("Result: \"{}\"", result);

    assert!(!result.is_empty(), "Transcription was empty");

    let result_words = normalize_words(&result);
    eprintln!("\nExpected {} words, got {} words", expected_words.len(), result_words.len());

    let lcs_len = longest_common_subsequence(&expected_words, &result_words);
    let accuracy = lcs_len as f64 / expected_words.len() as f64 * 100.0;
    eprintln!("Word accuracy (LCS): {}/{} = {:.1}%", lcs_len, expected_words.len(), accuracy);

    let result_set: std::collections::HashSet<&str> =
        result_words.iter().map(|s| s.as_str()).collect();
    let unique_missed: std::collections::BTreeSet<&str> = expected_words
        .iter()
        .map(|s| s.as_str())
        .filter(|w| !result_set.contains(w))
        .collect();
    if !unique_missed.is_empty() {
        eprintln!("Words not found in output: {:?}", unique_missed);
    }

    assert!(
        accuracy >= 70.0,
        "Word accuracy too low: {:.1}% (expected >= 70%)",
        accuracy,
    );

    eprintln!("\nPASSED: {:.1}% word accuracy", accuracy);
}

/// Same as test_interactive_chunked but using CoreML (WhisperKit) with Large v3.
///
/// Run with: cargo test --release test_coreml_chunked -- --ignored --nocapture
#[test]
#[ignore]
fn test_coreml_chunked() {
    let sentences = [
        "Yesterday morning I walked through the park and noticed the leaves were changing color.",
        "The temperature outside was around fifty degrees, which felt perfect for a long walk.",
        "Several children were playing near the fountain while their parents watched from wooden benches.",
        "A small brown dog ran across the path chasing after a bright red ball.",
        "I stopped at the corner bakery and ordered a coffee with two sugars and cream.",
        "The woman behind the counter smiled and said it would be ready in just a minute.",
        "While waiting I noticed a newspaper headline about new technology changing how people communicate.",
        //"After finishing my drink I continued walking toward the library on the other side of town.",
    ];

    let full_text = sentences.join(" ");
    let expected_words = normalize_words(&full_text);
    eprintln!("Test passage: {} words", expected_words.len());

    eprintln!("Generating speech via macOS TTS...");
    let mut all_samples: Vec<f32> = Vec::new();
    let mut sr = 0u32;
    for (i, sentence) in sentences.iter().enumerate() {
        let (samples, sample_rate) = generate_speech(sentence);
        sr = sample_rate;
        all_samples.extend_from_slice(&samples);
        if i < sentences.len() - 1 {
            all_samples.extend(vec![0.0f32; (sr as f64 * 0.3) as usize]);
        }
    }
    let total_duration = all_samples.len() as f64 / sr as f64;
    eprintln!("Total audio: {:.1}s at {} Hz", total_duration, sr);

    // VAD-based splitting.
    let vad_threshold: f32 = 0.005;
    let window_size = sr as usize / 20;
    let min_silence_windows = 4;
    let min_chunk_samples = sr as usize / 2;

    let mut chunks: Vec<&[f32]> = Vec::new();
    let mut chunk_start = 0;
    let mut silence_run = 0;

    for (i, window) in all_samples.chunks(window_size).enumerate() {
        let rms = compute_rms(window);
        if rms < vad_threshold {
            silence_run += 1;
            if silence_run >= min_silence_windows {
                let split = (i + 1 - silence_run) * window_size;
                if split > chunk_start && (split - chunk_start) >= min_chunk_samples {
                    chunks.push(&all_samples[chunk_start..split]);
                    chunk_start = (i + 1) * window_size;
                }
                silence_run = 0;
            }
        } else {
            silence_run = 0;
        }
    }
    if chunk_start < all_samples.len() && (all_samples.len() - chunk_start) >= min_chunk_samples {
        chunks.push(&all_samples[chunk_start..]);
    }

    eprintln!("VAD split into {} chunks", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        eprintln!("  chunk {}: {:.1}s", i, chunk.len() as f64 / sr as f64);
    }

    eprintln!("Loading CoreML model (Large v3)...");
    let coreml = CoreMLWhisperEngine::new(CoreMLModel::Large).expect("Failed to load CoreML model");
    let mut engine = Engine::CoreML(coreml);

    let mut transcripts: Vec<String> = Vec::new();
    let mut chunk_latencies: Vec<(f64, f64)> = Vec::new();
    let start = std::time::Instant::now();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_dur = chunk.len() as f64 / sr as f64;
        let chunk_start = std::time::Instant::now();
        match engine.transcribe_sync(chunk, sr, 1) {
            Ok(text) if !text.is_empty() => {
                let latency = chunk_start.elapsed().as_secs_f64();
                eprintln!("[chunk {}] {:.1}s audio -> {:.0}ms latency -> \"{}\"",
                    i, chunk_dur, latency * 1000.0, &text[..text.len().min(60)]);
                chunk_latencies.push((chunk_dur, latency));
                transcripts.push(text);
            }
            Ok(_) => {
                let latency = chunk_start.elapsed().as_secs_f64();
                eprintln!("[chunk {}] {:.1}s audio -> {:.0}ms latency -> (empty)", i, chunk_dur, latency * 1000.0);
                chunk_latencies.push((chunk_dur, latency));
            }
            Err(e) => eprintln!("[chunk {}] error: {}", i, e),
        }
    }

    let elapsed = start.elapsed();
    let result = transcripts.join(" ");

    if let Some(&(last_audio, last_latency)) = chunk_latencies.last() {
        eprintln!("\n── Latency (speech-end to text-ready) ──");
        eprintln!("  Final chunk: {:.1}s audio transcribed in {:.0}ms", last_audio, last_latency * 1000.0);
    }
    let avg_latency: f64 = chunk_latencies.iter().map(|(_, l)| l).sum::<f64>() / chunk_latencies.len() as f64;
    let max_latency = chunk_latencies.iter().map(|(_, l)| *l).fold(0.0f64, f64::max);
    eprintln!("  Avg chunk latency: {:.0}ms", avg_latency * 1000.0);
    eprintln!("  Max chunk latency: {:.0}ms", max_latency * 1000.0);

    eprintln!("\nTotal transcription time: {:.1}s (RTF: {:.2}x)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / total_duration);
    eprintln!("Result: \"{}\"", result);

    assert!(!result.is_empty(), "Transcription was empty");

    let result_words = normalize_words(&result);
    eprintln!("\nExpected {} words, got {} words", expected_words.len(), result_words.len());

    let lcs_len = longest_common_subsequence(&expected_words, &result_words);
    let accuracy = lcs_len as f64 / expected_words.len() as f64 * 100.0;
    eprintln!("Word accuracy (LCS): {}/{} = {:.1}%", lcs_len, expected_words.len(), accuracy);

    let result_set: std::collections::HashSet<&str> =
        result_words.iter().map(|s| s.as_str()).collect();
    let unique_missed: std::collections::BTreeSet<&str> = expected_words
        .iter()
        .map(|s| s.as_str())
        .filter(|w| !result_set.contains(w))
        .collect();
    if !unique_missed.is_empty() {
        eprintln!("Words not found in output: {:?}", unique_missed);
    }

    assert!(
        accuracy >= 70.0,
        "Word accuracy too low: {:.1}% (expected >= 70%)",
        accuracy,
    );

    eprintln!("\nPASSED: {:.1}% word accuracy", accuracy);
}

/// Tests WAV file reading utility.
#[test]
fn test_read_wav_invalid() {
    let result = pulse::read_wav("/nonexistent/file.wav");
    assert!(result.is_err());
}

// ── Moonshine tests ───────────────────────────────────────────────

/// Full end-to-end: macOS TTS → WAV → Moonshine → check output.
///
/// Run with: cargo test --release test_moonshine_e2e_with_tts -- --ignored --nocapture
#[test]
#[ignore] // requires model download, macOS, and ONNX Runtime
fn test_moonshine_e2e_with_tts() {
    let text = "The quick brown fox jumps over the lazy dog";
    eprintln!("Generating speech: \"{}\"", text);
    let (samples, sample_rate) = generate_speech(text);
    let duration = samples.len() as f64 / sample_rate as f64;
    eprintln!("Generated {:.1}s of audio at {} Hz", duration, sample_rate);

    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs > 0.01,
        "Audio appears to be silence (max={:.6})",
        max_abs
    );

    eprintln!("Loading Moonshine Base model...");
    let mut engine = MoonshineEngine::new().expect("Failed to load Moonshine model");

    eprintln!("Transcribing...");
    let start = std::time::Instant::now();
    let result = engine
        .transcribe_sync(&samples, sample_rate, 1)
        .expect("Transcription failed");
    let elapsed = start.elapsed();

    eprintln!("Transcribed in {:.2}s (RTF: {:.3}x)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / duration);
    eprintln!("Expected: \"{}\"", text);
    eprintln!("Got:      \"{}\"", result);

    assert!(!result.is_empty(), "Transcription was empty");

    let lower = result.to_lowercase();
    let key_words = ["quick", "brown", "fox", "lazy", "dog"];
    let matches: Vec<&str> = key_words
        .into_iter()
        .filter(|w| lower.contains(w))
        .collect();
    eprintln!("Matched {}/5 key words: {:?}", matches.len(), matches);
    assert!(
        matches.len() >= 3,
        "Expected at least 3/5 key words, got {}: {:?}",
        matches.len(),
        matches
    );
}

/// ~200-word transcription accuracy test with Moonshine Base using chunked decoding.
///
/// Moonshine Base is designed for short audio segments (~5-10s). For longer audio,
/// we chunk into 10s segments (same as interactive mode's CHUNK_INTERVAL) and
/// transcribe each independently, joining the results.
///
/// Run with: cargo test --release test_moonshine_200_word_accuracy -- --ignored --nocapture
#[test]
#[ignore]
fn test_moonshine_200_word_accuracy() {
    let sentences = [
        "Yesterday morning I walked through the park and noticed the leaves were changing color.",
        "The temperature outside was around fifty degrees, which felt perfect for a long walk.",
        "Several children were playing near the fountain while their parents watched from wooden benches.",
        "A small brown dog ran across the path chasing after a bright red ball.",
        "I stopped at the corner bakery and ordered a coffee with two sugars and cream.",
        "The woman behind the counter smiled and said it would be ready in just a minute.",
        "While waiting I noticed a newspaper headline about new technology changing how people communicate.",
        "After finishing my drink I continued walking toward the library on the other side of town.",
        "The library was quiet except for a few students studying at the long wooden tables near the windows.",
        "I found a comfortable chair in the corner and opened the book I had been meaning to read for weeks.",
        "The story was about a young scientist who discovered a way to convert sunlight into clean drinking water.",
        "Her invention could help millions of people in remote villages who lack access to safe water supplies.",
        "The first chapter described her childhood growing up on a farm where water was always scarce.",
        "She remembered watching her grandmother carry heavy buckets from the well every single morning.",
        "That memory inspired her to study chemistry and engineering at the state university.",
        "After graduating she spent three years working in a small laboratory funded by a research grant.",
    ];

    let full_text = sentences.join(" ");
    let expected_words: Vec<String> = normalize_words(&full_text);
    let word_count = expected_words.len();
    eprintln!("Test passage: {} words", word_count);

    eprintln!("Generating speech via macOS TTS...");
    let mut all_samples: Vec<f32> = Vec::new();
    let mut sr = 0u32;

    for (i, sentence) in sentences.iter().enumerate() {
        let (samples, sample_rate) = generate_speech(sentence);
        sr = sample_rate;
        all_samples.extend_from_slice(&samples);
        if i < sentences.len() - 1 {
            all_samples.extend(vec![0.0f32; (sr as f64 * 0.3) as usize]);
        }
    }

    let total_duration = all_samples.len() as f64 / sr as f64;
    eprintln!("Total audio: {:.1}s at {} Hz", total_duration, sr);

    // Chunk into 10-second segments (Moonshine is optimized for short audio).
    let chunk_size = sr as usize * 10;
    let chunks: Vec<&[f32]> = all_samples.chunks(chunk_size).collect();
    eprintln!("Split into {} chunks of ~10s each", chunks.len());

    eprintln!("Loading Moonshine Base model...");
    let mut engine = MoonshineEngine::new().expect("Failed to load Moonshine model");

    let mut transcripts: Vec<String> = Vec::new();
    let start = std::time::Instant::now();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_dur = chunk.len() as f64 / sr as f64;
        eprint!("[chunk {}] {:.1}s -> ", i, chunk_dur);
        match engine.transcribe_sync(chunk, sr, 1) {
            Ok(text) if !text.is_empty() => {
                eprintln!("\"{}\"", &text[..text.len().min(80)]);
                transcripts.push(text);
            }
            Ok(_) => eprintln!("(empty)"),
            Err(e) => eprintln!("error: {}", e),
        }
    }

    let elapsed = start.elapsed();
    let result = transcripts.join(" ");

    eprintln!(
        "\nTotal transcription time: {:.1}s (RTF: {:.2}x)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / total_duration,
    );
    eprintln!("Result: \"{}\"", result);

    assert!(!result.is_empty(), "Transcription was empty");

    let result_words = normalize_words(&result);
    eprintln!(
        "\nExpected {} words, got {} words",
        expected_words.len(),
        result_words.len()
    );

    let lcs_len = longest_common_subsequence(&expected_words, &result_words);
    let accuracy = lcs_len as f64 / expected_words.len() as f64 * 100.0;

    eprintln!(
        "Word accuracy (LCS): {}/{} = {:.1}%",
        lcs_len,
        expected_words.len(),
        accuracy
    );

    let result_set: std::collections::HashSet<&str> =
        result_words.iter().map(|s| s.as_str()).collect();
    let unique_missed: std::collections::BTreeSet<&str> = expected_words
        .iter()
        .map(|s| s.as_str())
        .filter(|w| !result_set.contains(w))
        .collect();
    if !unique_missed.is_empty() {
        eprintln!("Words not found in output: {:?}", unique_missed);
    }

    assert!(
        accuracy >= 70.0,
        "Word accuracy too low: {:.1}% (expected >= 70%)",
        accuracy,
    );

    eprintln!("\nPASSED: {:.1}% word accuracy", accuracy);
}

/// ~100-word transcription accuracy test with Moonshine Base.
///
/// Run with: cargo test --release test_moonshine_100_word_accuracy -- --ignored --nocapture
#[test]
#[ignore]
fn test_moonshine_100_word_accuracy() {
    let sentences = [
        "Yesterday morning I walked through the park and noticed the leaves were changing color.",
        "The temperature outside was around fifty degrees, which felt perfect for a long walk.",
        "Several children were playing near the fountain while their parents watched from wooden benches.",
        "A small brown dog ran across the path chasing after a bright red ball.",
        "I stopped at the corner bakery and ordered a coffee with two sugars and cream.",
        "The woman behind the counter smiled and said it would be ready in just a minute.",
        "While waiting I noticed a newspaper headline about new technology changing how people communicate.",
        "After finishing my drink I continued walking toward the library on the other side of town.",
    ];

    let full_text = sentences.join(" ");
    let expected_words: Vec<String> = normalize_words(&full_text);
    let word_count = expected_words.len();
    eprintln!("Test passage: {} words", word_count);
    eprintln!("Text: \"{}\"", full_text);

    eprintln!("Generating speech via macOS TTS...");
    let mut all_samples: Vec<f32> = Vec::new();
    let mut sr = 0u32;

    for (i, sentence) in sentences.iter().enumerate() {
        let (samples, sample_rate) = generate_speech(sentence);
        sr = sample_rate;
        all_samples.extend_from_slice(&samples);
        if i < sentences.len() - 1 {
            all_samples.extend(vec![0.0f32; (sr as f64 * 0.3) as usize]);
        }
    }

    let duration = all_samples.len() as f64 / sr as f64;
    eprintln!("Total audio: {:.1}s at {} Hz", duration, sr);

    eprintln!("Loading Moonshine Base model...");
    let mut engine = MoonshineEngine::new().expect("Failed to load Moonshine model");

    eprintln!("Transcribing...");
    let start = std::time::Instant::now();
    let result = engine
        .transcribe_sync(&all_samples, sr, 1)
        .expect("Transcription failed");
    let elapsed = start.elapsed();

    eprintln!("Transcribed in {:.1}s (RTF: {:.2}x)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / duration);
    eprintln!("Result: \"{}\"", result);

    assert!(!result.is_empty(), "Transcription was empty");

    let result_words = normalize_words(&result);
    eprintln!("\nExpected {} words, got {} words", expected_words.len(), result_words.len());

    let lcs_len = longest_common_subsequence(&expected_words, &result_words);
    let accuracy = lcs_len as f64 / expected_words.len() as f64 * 100.0;

    eprintln!("Word accuracy (LCS): {}/{} = {:.1}%", lcs_len, expected_words.len(), accuracy);

    let result_set: std::collections::HashSet<&str> =
        result_words.iter().map(|s| s.as_str()).collect();
    let unique_missed: std::collections::BTreeSet<&str> = expected_words
        .iter()
        .map(|s| s.as_str())
        .filter(|w| !result_set.contains(w))
        .collect();
    if !unique_missed.is_empty() {
        eprintln!("Words not found in output: {:?}", unique_missed);
    }

    assert!(
        accuracy >= 70.0,
        "Word accuracy too low: {:.1}% (expected >= 70%)",
        accuracy,
    );

    eprintln!("\nPASSED: {:.1}% word accuracy", accuracy);
}

/// Tests Moonshine doesn't crash on silence.
#[test]
#[ignore]
fn test_moonshine_silence() {
    let mut engine = MoonshineEngine::new().expect("Failed to load Moonshine model");
    let silence = vec![0.0f32; 16000 * 2];
    let result = engine
        .transcribe_sync(&silence, 16000, 1)
        .expect("Transcription failed on silence");
    eprintln!("Moonshine silence transcription: {:?}", result);
    assert!(result.is_empty(), "Expected empty result for silence, got: {:?}", result);
}
