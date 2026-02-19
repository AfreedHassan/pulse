use pulse::providers::local_whisper::{PulseEngine, PulseModel};
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

/// Tests WAV file reading utility.
#[test]
fn test_read_wav_invalid() {
    let result = pulse::read_wav("/nonexistent/file.wav");
    assert!(result.is_err());
}
