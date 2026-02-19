//! Pulse — speech-to-text dictation engine.
//!
//! This is the CLI entry point. Use `--file path.wav` for file mode,
//! or run without arguments for interactive recording mode.
//! Add `--paste` to paste transcribed text into the frontmost app.
//! Add `--model fast|balanced|quality` to pick model tier (default: fast).

use pulse::audio::{AudioCapture, SimpleVad, VoiceActivity, VAD_CHUNK_SIZE};
use pulse::engine::formatter::{self, Formatter, FormatterConfig};
use pulse::platform::{hotkey, paste};
use pulse::providers::local_whisper::{PulseEngine, PulseModel};
use pulse::read_wav;
use std::io::{self, Read};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Minimum utterance length before we bother transcribing.
const MIN_CHUNK_SAMPLES: usize = 32_000; // 2s at 16kHz

/// How often to poll the audio buffer for VAD.
const VAD_POLL_INTERVAL: Duration = Duration::from_millis(32);

fn audio_duration_secs(num_samples: usize, sample_rate: u32, channels: u16) -> f64 {
    num_samples as f64 / (sample_rate as f64 * channels as f64)
}

fn main() {
    let _ = dotenvy::dotenv();
    let args: Vec<String> = std::env::args().collect();

    let paste_mode = args.iter().any(|a| a == "--paste");
    if paste_mode {
        eprintln!("Paste mode enabled (will paste into frontmost app)");
    }

    // Parse model tier (default: fast).
    let model_tier = args
        .windows(2)
        .find(|w| w[0] == "--model")
        .and_then(|w| PulseModel::parse(&w[1]))
        .unwrap_or(PulseModel::Fast);

    let formatter = FormatterConfig::from_env().map(|cfg| {
        eprintln!("LLM formatter enabled (model: {})", cfg.model);
        Formatter::new(cfg)
    });

    // Load engine.
    eprintln!("Loading Pulse model ({})...", model_tier);
    let engine = match PulseEngine::new(model_tier) {
        Ok(e) => {
            eprintln!("✓ Model loaded successfully.");
            Arc::new(Mutex::new(e))
        }
        Err(e) => {
            eprintln!("✗ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    // Find --file argument
    let file_arg = args
        .windows(2)
        .find(|w| w[0] == "--file")
        .map(|w| w[1].clone());

    match file_arg {
        Some(path) => run_file_mode(&path, &engine, formatter.as_ref(), paste_mode),
        None => run_interactive_mode(&engine, formatter.as_ref(), paste_mode),
    }
}

fn emit_text(text: &str, fmt: Option<&Formatter>, paste_mode: bool) {
    let output = match fmt {
        Some(f) => match f.format(text) {
            Ok(formatted) if formatter::passes_guardrails(text, &formatted) => formatted,
            Ok(_) => {
                eprintln!("Formatter produced bad output, using raw text");
                text.to_string()
            }
            Err(e) => {
                eprintln!("Formatter error (using raw text): {}", e);
                text.to_string()
            }
        },
        None => text.to_string(),
    };

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
    engine: &Arc<Mutex<PulseEngine>>,
    fmt: Option<&Formatter>,
    paste_mode: bool,
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
                emit_text(&text, fmt, paste_mode);
            }
        }
        Err(e) => {
            eprintln!("Transcription error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_interactive_mode(engine: &Arc<Mutex<PulseEngine>>, fmt: Option<&Formatter>, paste_mode: bool) {
    let audio = AudioCapture::new();
    let sample_rate = audio.sample_rate();
    let channels = audio.channels();

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

    let mut vad = SimpleVad::new();

    // Scale VAD chunk size and minimum utterance length to the device's native sample rate.
    // VAD_CHUNK_SIZE (512) is calibrated for 16kHz (~32ms window).
    // MIN_CHUNK_SAMPLES (32000) is 2s at 16kHz.
    // At 48kHz we need 3x more samples for the same duration.
    let rate_scale = sample_rate as usize / 16_000;
    let vad_chunk = VAD_CHUNK_SIZE * rate_scale.max(1);
    let min_chunk = MIN_CHUNK_SAMPLES * rate_scale.max(1);

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
        vad.reset();

        // VAD loop: monitor audio while recording, transcribe completed utterances.
        // Transcription runs on a background thread so VAD polling continues
        // uninterrupted — this prevents missing words right after boundaries.
        let (chunk_tx, result_rx) = mpsc::channel::<String>();
        let engine_ref = Arc::clone(&engine);
        let transcribe_handle = {
            let chunk_tx = chunk_tx;
            let (work_tx, work_rx) = mpsc::channel::<(Vec<f32>, u32, u16)>();

            let handle = thread::spawn(move || {
                let mut eng = engine_ref.lock().unwrap();
                while let Ok((chunk, sr, ch)) = work_rx.recv() {
                    let dur = chunk.len() as f64 / (sr as f64 * ch as f64);
                    eprintln!("[vad] Transcribing utterance ({:.1}s)...", dur);
                    match eng.transcribe_sync(&chunk, sr, ch) {
                        Ok(text) if !text.is_empty() => {
                            eprintln!("[vad] → \"{}\"", &text[..text.len().min(80)]);
                            let _ = chunk_tx.send(text);
                        }
                        Ok(_) => eprintln!("[vad] (silence/empty)"),
                        Err(e) => eprintln!("[vad] Transcription error: {}", e),
                    }
                }
            });

            // Send chunks via work_tx, results come back via result_rx.
            (work_tx, handle)
        };
        let (work_tx, transcribe_thread) = transcribe_handle;

        let mut samples_consumed: usize = 0;

        loop {
            // Check for trigger to stop.
            if trigger_rx.try_recv().is_ok() {
                break;
            }

            thread::sleep(VAD_POLL_INTERVAL);

            // Peek at the tail of the live buffer for VAD.
            let tail = audio.peek_tail(vad_chunk);
            if tail.is_empty() {
                continue;
            }

            let (state, changed) = vad.update(&tail);

            if changed && state == VoiceActivity::Silence {
                // Utterance boundary! Drain and send to background transcriber.
                let current_len = audio.live_len();
                let drain_count = current_len.saturating_sub(vad_chunk);
                if drain_count > min_chunk {
                    let chunk = audio.drain(drain_count);
                    eprintln!("[vad] Utterance boundary, dispatching...");
                    let _ = work_tx.send((chunk, sample_rate, channels));
                    samples_consumed += drain_count;
                }
            }
        }

        // User released Fn / pressed Enter — stop recording and process remaining.
        audio.stop();
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
            emit_text(&full_text, fmt, paste_mode);
        }

        if paste_mode {
            eprintln!("\nReady. Hold Fn (🌐) to record again.");
        } else {
            eprintln!("\nPress Enter to record again.");
        }
    }
}
