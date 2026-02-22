//! Pulse — speech-to-text dictation engine.
//!
//! This is the CLI entry point. Use `--file path.wav` for file mode,
//! or run without arguments for interactive recording mode.
//! Paste mode is on by default; add `--no-paste` to disable.
//! Add `--model fast|balanced|quality` to pick model tier (default: large).
//! Add `--provider whisper` to use Candle Whisper with Metal GPU.
//! Add `--provider moonshine` to use Moonshine Base with ONNX/CoreML.

use pulse::audio::AudioCapture;
use pulse::engine::formatter::{self, Formatter, FormatterConfig};
use pulse::engine::{ContactEngine, LearningEngine, ModeEngine, ShortcutEngine};
use pulse::platform::{accessibility, apps, hotkey, paste};
use pulse::platform::indicator::Indicator;
use pulse::providers::coreml_whisper::CoreMLModel;
use pulse::providers::local_whisper::PulseModel;
use pulse::providers::Engine;
use pulse::read_wav;
use pulse::storage::Storage;
use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// How often to drain the audio buffer and dispatch for transcription.
const CHUNK_INTERVAL: Duration = Duration::from_secs(5);

/// Minimum samples before dispatching a chunk (0.5s at 16kHz).
const MIN_CHUNK_SAMPLES: usize = 8_000;

/// How often to check for stop triggers while recording.
const POLL_INTERVAL: Duration = Duration::from_millis(32);

fn audio_duration_secs(num_samples: usize, sample_rate: u32, channels: u16) -> f64 {
    num_samples as f64 / (sample_rate as f64 * channels as f64)
}

fn main() {
    let _ = dotenvy::dotenv();
    let args: Vec<String> = std::env::args().collect();

    let no_paste = args.iter().any(|a| a == "--no-paste");
    let paste_mode = !no_paste;
    if paste_mode {
        eprintln!("Paste mode enabled (will paste into frontmost app)");
    }

    // Parse provider (default: coreml).
    let provider = args
        .windows(2)
        .find(|w| w[0] == "--provider")
        .map(|w| w[1].as_str())
        .unwrap_or("coreml");

    // Parse model tier (default: large). Only used for Whisper.
    let model_tier = args
        .windows(2)
        .find(|w| w[0] == "--model")
        .and_then(|w| PulseModel::parse(&w[1]))
        .unwrap_or(PulseModel::Large);

    let formatter = FormatterConfig::from_env().map(|cfg| {
        eprintln!("LLM formatter enabled (model: {})", cfg.model);
        Formatter::new(cfg)
    });

    // Load engine based on provider selection.
    let engine = match provider {
        "coreml" => {
            let coreml_model = args
                .windows(2)
                .find(|w| w[0] == "--model")
                .and_then(|w| CoreMLModel::parse(&w[1]))
                .unwrap_or(CoreMLModel::Large);
            eprintln!("Loading CoreML whisper model ({})...", coreml_model);
            match pulse::providers::coreml_whisper::CoreMLWhisperEngine::new(coreml_model) {
                Ok(e) => {
                    eprintln!("✓ CoreML model loaded successfully.");
                    Arc::new(Mutex::new(Engine::CoreML(e)))
                }
                Err(e) => {
                    eprintln!("✗ Failed to load CoreML model: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "moonshine" => {
            eprintln!("Loading Moonshine Base model...");
            match pulse::providers::moonshine::MoonshineEngine::new() {
                Ok(e) => {
                    eprintln!("✓ Moonshine model loaded successfully.");
                    Arc::new(Mutex::new(Engine::Moonshine(e)))
                }
                Err(e) => {
                    eprintln!("✗ Failed to load Moonshine model: {}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Loading Pulse model ({})...", model_tier);
            match pulse::providers::local_whisper::PulseEngine::new(model_tier) {
                Ok(e) => {
                    eprintln!("✓ Model loaded successfully.");
                    Arc::new(Mutex::new(Engine::Whisper(e)))
                }
                Err(e) => {
                    eprintln!("✗ Failed to load model: {}", e);
                    std::process::exit(1);
                }
            }
        }
    };

    // Initialize persistent storage.
    let db_path = pulse::data_dir().join("pulse.db");
    std::fs::create_dir_all(db_path.parent().unwrap()).ok();
    let storage = Storage::open(&db_path).expect("Failed to open storage");

    let file_arg = args
        .windows(2)
        .find(|w| w[0] == "--file")
        .map(|w| w[1].as_str());

    match file_arg {
        Some(path) => run_file_mode(path, &engine, formatter.as_ref(), paste_mode, &storage),
        None => run_interactive_mode(&engine, formatter.as_ref(), paste_mode, &storage),
    }
}

fn emit_text(text: &str, fmt: Option<&Formatter>, paste_mode: bool, storage: &Storage) {
    // 1. Apply learned corrections (fixes ASR word errors).
    let learning = LearningEngine::new(storage);
    let text = learning.apply(text);

    // 2. Expand shortcuts (brb → be right back, etc.).
    let shortcuts = ShortcutEngine::new(storage);
    let text = shortcuts.expand(&text);

    // 3. Fix contact name capitalization.
    let contacts = ContactEngine::new(storage);
    let text = contacts.apply_names(&text);

    // 4. Detect writing mode + text field context (paste mode only).
    let (context, writing_mode) = if paste_mode {
        let ctx = accessibility::read_focused_text_field();
        let mode_engine = ModeEngine::new(storage);
        let mode = apps::detect_frontmost_app()
            .map(|app| mode_engine.resolve_mode(&app));
        (ctx, mode)
    } else {
        (None, None)
    };

    // 5. LLM formatter.
    let output = match fmt {
        Some(f) => {
            match f.format_with_mode(&text, writing_mode.as_ref(), context.as_deref()) {
                Ok(formatted) if formatter::passes_guardrails(&text, &formatted) => formatted,
                Ok(_) => {
                    eprintln!("Formatter produced bad output, using raw text");
                    text.to_string()
                }
                Err(e) => {
                    eprintln!("Formatter error (using raw text): {}", e);
                    text.to_string()
                }
            }
        }
        None => text.to_string(),
    };

    // 6. Paste or print.
    if paste_mode {
        if let Err(e) = paste::paste_text(&output) {
            eprintln!("Paste failed (printing to stdout instead): {}", e);
            println!("{}", output);
        }
    } else {
        println!("{}", output);
    }
}

fn run_file_mode(
    path: &str,
    engine: &Arc<Mutex<Engine>>,
    fmt: Option<&Formatter>,
    paste_mode: bool,
    storage: &Storage,
) {
    let (samples, sample_rate, channels) = read_wav(path).expect("Failed to read WAV file");
    let duration = audio_duration_secs(samples.len(), sample_rate, channels);
    eprintln!(
        "Transcribing {} ({:.1}s, {} Hz, {} ch)...",
        path, duration, sample_rate, channels,
    );

    let start = std::time::Instant::now();
    match engine.lock().unwrap().transcribe_sync(&samples, sample_rate, channels) {
        Ok(text) => {
            let elapsed = start.elapsed();
            eprintln!("Transcribed in {:.1}s", elapsed.as_secs_f64());
            if text.is_empty() {
                eprintln!("(no speech detected)");
            } else {
                emit_text(&text, fmt, paste_mode, storage);
            }
        }
        Err(e) => {
            eprintln!("Transcription error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_interactive_mode(
    engine: &Arc<Mutex<Engine>>,
    fmt: Option<&Formatter>,
    paste_mode: bool,
    storage: &Storage,
) {
    let audio = AudioCapture::new();
    let sample_rate = audio.sample_rate();
    let channels = audio.channels();

    // Spawn the visual recording indicator (paste mode only).
    let mut indicator = if paste_mode {
        match Indicator::spawn() {
            Some(ind) => {
                eprintln!("Recording indicator ready.");
                Some(ind)
            }
            None => {
                eprintln!("Recording indicator not available (helper binary not found).");
                None
            }
        }
    } else {
        None
    };

    // In paste mode, use a global hotkey (Fn/Globe) so it works from any app.
    // Otherwise, use Enter in the terminal.
    let trigger_rx = if paste_mode {
        eprintln!("Ready. Hold Fn (🌐) to record, release to stop.\n");
        hotkey::listen()
    } else {
        eprintln!("Ready. Press Enter to start recording.\n");
        let (enter_tx, enter_rx) = mpsc::channel::<()>();
        thread::spawn(move || {
            let stdin = io::stdin();
            let mut buf = [0u8; 1];
            loop {
                if stdin.lock().read(&mut buf).is_ok() {
                    let _ = enter_tx.send(());
                }
            }
        });
        enter_rx
    };

    // Scale minimum chunk size to the device's native sample rate.
    // MIN_CHUNK_SAMPLES (8000) is 0.5s at 16kHz.
    // At 48kHz we need 3x more samples for the same duration.
    let rate_scale = (sample_rate as usize / 16_000).max(1);
    let min_chunk = MIN_CHUNK_SAMPLES * rate_scale;

    loop {
        // Wait for trigger to start.
        let _ = trigger_rx.recv();
        if paste_mode {
            eprintln!("🔴 Recording... Release Fn to stop.");
        } else {
            eprintln!("🔴 Recording... Press Enter to stop.");
        }
        // Clear any queued triggers.
        while trigger_rx.try_recv().is_ok() {}
        audio.start();

        // Show recording indicator.
        if let Some(ref mut ind) = indicator {
            ind.show();
        }

        // Timer-based chunking: drain the audio buffer every CHUNK_INTERVAL
        // and dispatch to a background transcription thread.
        let (result_tx, result_rx) = mpsc::channel::<String>();
        let (work_tx, work_rx) = mpsc::channel::<(Vec<f32>, u32, u16)>();
        let engine_ref = Arc::clone(engine);

        let transcribe_thread = thread::spawn(move || {
            let mut eng = engine_ref.lock().unwrap();
            while let Ok((chunk, sr, ch)) = work_rx.recv() {
                let dur = chunk.len() as f64 / (sr as f64 * ch as f64);
                eprintln!("[chunk] Transcribing {:.1}s...", dur);
                match eng.transcribe_sync(&chunk, sr, ch) {
                    Ok(text) if !text.is_empty() => {
                        eprintln!("[chunk] -> \"{}\"", &text[..text.len().min(80)]);
                        let _ = result_tx.send(text);
                    }
                    Ok(_) => eprintln!("[chunk] (silence/empty)"),
                    Err(e) => eprintln!("[chunk] Transcription error: {}", e),
                }
            }
        });

        let mut samples_consumed: usize = 0;
        let mut last_drain = Instant::now();

        loop {
            // Check for trigger to stop.
            if trigger_rx.try_recv().is_ok() {
                break;
            }

            thread::sleep(POLL_INTERVAL);

            // Every CHUNK_INTERVAL, drain the buffer and dispatch for transcription.
            if last_drain.elapsed() >= CHUNK_INTERVAL {
                let available = audio.live_len();
                if available > min_chunk {
                    let chunk = audio.drain(available);
                    eprintln!(
                        "[chunk] Dispatching {:.1}s of audio...",
                        audio_duration_secs(chunk.len(), sample_rate, channels)
                    );
                    samples_consumed += chunk.len();
                    let _ = work_tx.send((chunk, sample_rate, channels));
                }
                last_drain = Instant::now();
            }
        }

        // User released Fn / pressed Enter — stop recording and process remaining.
        audio.stop();

        // Hide recording indicator.
        if let Some(ref mut ind) = indicator {
            ind.hide();
        }

        let remaining = audio.samples();
        let total_samples = samples_consumed + remaining.len();
        let total_duration = audio_duration_secs(total_samples, sample_rate, channels);
        eprintln!("⏹ Stopped. Captured {:.1}s total.", total_duration);

        if !remaining.is_empty() && remaining.len() > min_chunk {
            let _ = work_tx.send((remaining, sample_rate, channels));
        }

        // Signal the transcription thread to finish and wait for it.
        drop(work_tx);
        let _ = transcribe_thread.join();

        // Collect all transcription results.
        let mut transcripts: Vec<String> = Vec::new();
        while let Ok(text) = result_rx.try_recv() {
            transcripts.push(text);
        }

        // Combine all utterance transcripts and format.
        let full_text = transcripts.join(" ");
        if full_text.is_empty() {
            eprintln!("(no speech detected)");
        } else {
            emit_text(&full_text, fmt, paste_mode, storage);
        }

        if paste_mode {
            eprintln!("\nReady. Hold Fn (🌐) to record again.");
        } else {
            eprintln!("\nPress Enter to record again.");
        }
    }
}
